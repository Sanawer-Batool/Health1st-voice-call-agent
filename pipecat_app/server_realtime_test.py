"""
Phase 1 of the streaming-pipeline rebuild: prove that cloud streaming
STT/LLM/TTS (Deepgram + OpenAI + Cartesia) actually fixes the latency
problem, before adding ANY complexity back in — no tools, no LangGraph,
no RAG, no safety guardrail. Just: does this respond fast.

Same Twilio transport/webhook setup as before (server.py) — only the
STT/LLM/TTS internals change. Point your ngrok-fronted Twilio number's
webhook at THIS file's /twiml when testing this phase.

Run:
    uvicorn server_realtime_test:app --port 8000 --reload

Requires (on top of what earlier phases needed):
    pip install "pipecat-ai[deepgram,openai,cartesia,silero]"

Env vars needed (add to .env):
    DEEPGRAM_API_KEY
    OPENAI_API_KEY
    CARTESIA_API_KEY
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN

Watch the terminal during a real call — the [latency] lines show
per-stage TTFB (time-to-first-byte) for STT, LLM, and TTS. That's your
real evidence for whether this phase actually fixed anything.
"""
import os
import json
import time
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import MetricsFrame, TTSSpeakFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService

load_dotenv()

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY")

# A default Cartesia voice for now — swap once you've picked a real one
# from their voice library. Not the focus of this phase.
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")

app = FastAPI()


class LatencyMonitor(FrameProcessor):
    """
    Logs per-stage TTFB (time-to-first-byte) metrics as they flow through
    the pipeline. This is your actual evidence for Phase 1 — not a guess,
    a real per-call number for STT, LLM, and TTS latency.
    """

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for metric in frame.data:
                if isinstance(metric, TTFBMetricsData):
                    ms = metric.value * 1000
                    logger.info(f"[latency] {metric.processor}: {ms:.0f}ms")
        await self.push_frame(frame, direction)


@app.post("/twiml")
async def twiml_endpoint(request: Request):
    host = request.url.hostname
    stream_url = f"wss://{host}/ws"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.post("/status")
async def status_callback(request: Request):
    data = await request.form()
    logger.info(f"[status] Call {data.get('CallSid')}: {data.get('CallStatus')}")
    return PlainTextResponse(content="", status_code=204)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    timing_start = time.monotonic()

    stream_sid = None
    call_sid = None
    while stream_sid is None:
        message = await websocket.receive_text()
        data = json.loads(message)
        if data.get("event") == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]

            logger.info(f"[timing] start event parsed: {time.monotonic() - timing_start:.2f}s")
    logger.info(f"Call started — stream_sid={stream_sid}, call_sid={call_sid}")

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
        ),
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        ttfs_p99_latency=0.9,
        settings=DeepgramSTTService.Settings(model="nova-3", language="en"),
    )
    logger.info(f"[timing] stt constructed: {time.monotonic() - timing_start:.2f}s")

    llm_settings = OpenAILLMService.Settings(
        model="gpt-5.4-nano",
        extra={"reasoning_effort": "none"},
    )
    logger.info(f"[llm-config] {llm_settings}")
    llm = OpenAILLMService(settings=llm_settings, api_key=OPENAI_API_KEY)
    
    logger.info(f"[timing] llm constructed: {time.monotonic() - timing_start:.2f}s")

    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        settings=CartesiaTTSService.Settings(voice=CARTESIA_VOICE_ID, model="sonic-2"),
    )
    logger.info(f"[timing] tts constructed: {time.monotonic() - timing_start:.2f}s")

    # Deliberately bare system prompt for Phase 1 — no booking logic, no
    # tools, no guardrails yet. We're testing raw pipeline speed only.
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly phone assistant for a clinic. "
                    "Keep responses to 1-2 short sentences. This is a phone "
                    "call, not a chat window — sound natural and brief."
                ),
            }
        ]
    )

    # Standard cascade wiring: user_aggregator sits directly before the
    # LLM (it's what turns accumulated transcript + VAD turn-stop into the
    # LLMContextFrame that actually triggers inference); assistant_aggregator
    # sits directly after (captures the LLM's reply back into context for
    # the next turn's history).
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.5, min_volume=0.3, start_secs=0.1, stop_secs=0.5)
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.5)],
            ),
        ),
    )
    logger.info(f"[timing] vad/aggregator constructed: {time.monotonic() - timing_start:.2f}s")

    latency_monitor = LatencyMonitor()

    pipeline = Pipeline([
        transport.input(),
        stt,
        latency_monitor,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=8000,
            enable_metrics=True,  # required for MetricsFrame/TTFB data to be emitted at all
        ),
    )
    logger.info(f"[timing] pipeline built, about to run: {time.monotonic() - timing_start:.2f}s")

    # Speaking first absorbs Deepgram's websocket setup window before the caller responds.
    await task.queue_frames([
        TTSSpeakFrame(text="Hi, thanks for calling Health1st Clinic! How can I help you today?")
    ])

    runner = PipelineRunner()
    await runner.run(task)
