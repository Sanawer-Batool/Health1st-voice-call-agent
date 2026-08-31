"""
Step 3b of the build plan: swap the "repeat it back" bridge for a real call
into langgraph_app/graph.py. STT (Whisper) and TTS (Piper) are UNCHANGED
from the confirmed-working Step 3a file (server_voice_test.py) — see that
file's own docstring for the full history of what was wrong with VAD/STT/TTS
and how each was found and fixed. Don't re-litigate those here; if audio
input/output ever seems broken again, the fastest diagnostic is running
server_voice_test.py side by side to check whether the same symptom
reproduces there — if it does, it's not a LangGraph regression.

The only functional change from server_voice_test.py:
  FinalTranscriptToTTS (echoed the transcript) -> LangGraphBridge (calls
  langgraph_app/graph.py, speaks the agent's reply instead).

Run:
    uvicorn server_voice_langgraph:app --port 8000 --reload

Same ngrok/Twilio webhook setup as before, pointed at THIS file's /twiml
and /ws endpoints.

Requires (on top of what server_voice_test.py needed):
    Your langgraph_app/graph.py must be importable and expose a compiled
    graph named `graph`, invocable as graph.invoke(state) -> state, where
    state["messages"] is a list of langchain_core message objects.
    OPENAI_API_KEY must be set in .env (loaded via python-dotenv, same as
    the rest of this project) since graph.py's LLM calls read it from
    there.

EXPECTED LATENCY CHANGE: the first reply after this swap will feel
noticeably slower than the Step 3a echo test, because every turn now
includes a real OpenAI API round trip (potentially multi-hop, if the LLM
calls a tool like check_availability before replying) sitting in the
middle of the STT -> TTS chain. That's expected, not a bug — measure and
optimize it only once this is confirmed working end to end.
"""
import os
import sys
import json
import time
import random
import asyncio
from contextlib import asynccontextmanager
import numpy as np
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
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.frames.frames import (
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TextFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.whisper.stt import WhisperSTTService, Model as WhisperModel
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transcriptions.language import Language

# Make langgraph_app/graph.py importable from this file's location.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "langgraph_app"))
from graph import graph as langgraph_graph
from langchain_core.messages import HumanMessage

load_dotenv()

# DEBUG logging so you can see exactly what STT is (or isn't) transcribing
# in real time — this is what would have surfaced the no_speech_prob issue
# immediately instead of just hearing dead air.
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Belt-and-suspenders fix for the print()-buffering issue: force stdout to
# line-buffer even when it's not attached to a real terminal (which is the
# case under uvicorn --reload on Windows). Not strictly needed anymore since
# every print() below has been replaced with logger calls, but harmless to
# keep in case you add more print() debugging later.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


class VADLogger(FrameProcessor):
    """
    Confirms VAD is actually firing on the incoming Twilio audio, completely
    independent of whether Whisper ever produces a transcript. If you don't
    see these log lines while talking, the problem is upstream of STT.

    Listens for VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame —
    the raw signal straight out of Silero VAD, and the same frames
    SegmentedSTTService itself listens for to know when to buffer/flush
    audio. (Note: these are NOT the same classes as the plain
    UserStartedSpeakingFrame/UserStoppedSpeakingFrame — there's no
    inheritance relationship between them, so checking for the wrong one
    silently never fires, regardless of whether VAD is actually working.)

    Also logs a running count of raw InputAudioRawFrame chunks once a
    second, so you can see — independent of VAD entirely — whether audio
    is even still flowing from Twilio at each point in the call. Alongside
    that, logs the peak sample amplitude seen in that window (0.0-1.0,
    where 1.0 is digital full-scale) as a sanity check on whether the
    decoded audio actually has real signal in it, independent of Silero's
    own volume gate.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._audio_frame_count = 0
        self._last_log_time = 0.0
        self._peak_amplitude = 0.0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.info("VAD: user started speaking")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.info("VAD: user stopped speaking")
        elif isinstance(frame, InputAudioRawFrame):
            self._audio_frame_count += 1
            samples = np.frombuffer(frame.audio, dtype=np.int16)
            if samples.size:
                peak = float(np.abs(samples).max()) / 32768.0
                self._peak_amplitude = max(self._peak_amplitude, peak)
            now = time.monotonic()
            if now - self._last_log_time >= 1.0:
                logger.info(
                    f"Audio in: {self._audio_frame_count} chunks so far, "
                    f"peak amplitude this window: {self._peak_amplitude:.4f}"
                )
                self._last_log_time = now
                self._peak_amplitude = 0.0
        else:
            # DIAGNOSTIC: we've now confirmed twice that no
            # VADUserStartedSpeakingFrame ever reaches this processor, even
            # with loud, unambiguous speech (peak amplitude 0.93) hitting
            # the transport. Rather than guess a third blind fix, log the
            # class name of literally every non-audio frame that passes
            # through here. This tells us the ground truth for this specific
            # pipecat version: whether VAD frames are (a) never generated at
            # all, (b) generated under a different/renamed class, or (c)
            # generated but consumed internally by the transport's turn
            # controller before ever reaching the pipeline's processor
            # chain. Whichever of those it is changes the fix completely,
            # so don't skip reading these logs.
            logger.debug(f"[frame-tap] {frame.__class__.__name__}")
        await self.push_frame(frame, direction)

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# Shared, preloaded STT/TTS instances. Populated once in `lifespan` below,
# before Uvicorn ever starts accepting connections, so a caller's first
# word never waits on model loading. See DEBUGGING NOTES #6 at the top of
# this file for why, and the concurrency caveat if you need multiple
# simultaneous calls later.
_stt_service: WhisperSTTService | None = None
_tts_service: PiperTTSService | None = None


def _build_stt() -> WhisperSTTService:
    settings = WhisperSTTService.Settings(
        # UPGRADED from Model.BASE: DISTIL_MEDIUM_EN is actually pipecat's
        # own built-in default model (confirmed in WhisperSTTService.__init__
        # source), not base. It's English-only and distilled from medium —
        # since we're pinning language=EN anyway (see below), an
        # English-only model doesn't waste capacity representing other
        # languages, so it's meaningfully more accurate on accented English
        # than a multilingual model of similar size, while running faster
        # than a full (non-distilled) medium model would. Trade-off: larger
        # download (first startup will take longer to fetch/cache it) and
        # somewhat higher per-utterance inference latency than base — worth
        # timing on your actual hardware; drop back to Model.SMALL if the
        # latency cost isn't worth the accuracy gain for your case.
        model=WhisperModel.DISTIL_MEDIUM_EN,
        # CONFIRMED FIX (source-verified against pipecat-ai==1.7.0):
        # run_stt() calls self._model.transcribe(audio, language=language)
        # using self._settings.language, which defaults to None (i.e.
        # auto-detect) if never set. Per-utterance language auto-detection
        # on short, noisy, 8kHz telephone audio — especially with a
        # non-native accent — is unreliable and can occasionally produce
        # wildly unrelated-sounding text (e.g. "checkup" transcribed as
        # "jacob") when detection drifts. Pinning language=Language.EN
        # removes that entire axis of uncertainty at zero cost. Note: this
        # pipecat wrapper does NOT forward initial_prompt, beam_size, or
        # best_of to faster-whisper's transcribe() — only language is
        # passed through — so vocabulary-biasing isn't available at this
        # layer; see the chat history for other mitigation options
        # (bigger model, LLM-side error tolerance) if accuracy is still
        # not good enough after this.
        language=Language.EN,
        # CONFIRMED BUG (verified by downloading pipecat-ai==1.7.0 and
        # reading services/whisper/stt.py directly, not guessed):
        # run_stt()'s filter is:
        #   if no_speech_prob_threshold is not None and
        #      segment.no_speech_prob < no_speech_prob_threshold:
        #          text += segment.text
        # Setting no_speech_prob=None does NOT disable filtering — it
        # makes the "is not None" check always False, so NO segment's
        # text is ever appended, for any call, ever. That's the exact
        # total-silence-no-error symptom we've been chasing. Since
        # segment.no_speech_prob is always in [0, 1], setting the
        # threshold above that range makes "segment.no_speech_prob <
        # threshold" true for every real segment, which is what
        # actually achieves "accept everything" with this code.
        no_speech_prob=1.01,
    )
    logger.debug(f"WhisperSTTService.Settings actually built: {vars(settings)}")
    return WhisperSTTService(settings=settings)


def _build_tts() -> PiperTTSService:
    return PiperTTSService(settings=PiperTTSService.Settings(voice="en_US-lessac-medium"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stt_service, _tts_service
    logger.info("Preloading Whisper + Piper models before accepting connections...")
    start = time.monotonic()
    # Still off the event loop via to_thread — these are the same blocking
    # constructors as before, just called once at startup instead of once
    # per call.
    _stt_service = await asyncio.to_thread(_build_stt)
    _tts_service = await asyncio.to_thread(_build_tts)
    logger.info(f"Models preloaded in {time.monotonic() - start:.1f}s — ready for calls.")
    yield
    # (no teardown needed for these two)


app = FastAPI(lifespan=lifespan)


class LangGraphBridge(FrameProcessor):
    """
    Step 3b: real reasoning hookup, replacing Step 3a's echo bridge
    (FinalTranscriptToTTS in server_voice_test.py). Same TTSSpeakFrame
    lesson still applies here — see that file's docstring note #5 for why
    a plain TextFrame silently never gets synthesized without an LLM turn
    bracketing it (this bridge has no such issue since it always emits
    TTSSpeakFrame directly, same as before).

    One instance per call (created fresh in websocket_endpoint below), so
    conversation state is naturally scoped to that single call — no global
    dict, no cross-call bleed. This is also the correct pattern once
    concurrent calls are supported later, not just a shortcut for now.

    CONFIRMED BUG (source-verified against pipecat-ai==1.7.0): LangGraph
    replied correctly and Piper logged "Finished TTS [...]" with the right
    text, but no audio was ever heard — an InterruptionFrame showed up
    right around the same moment, and TTSService._handle_interruption()
    explicitly discards all pending/in-flight audio on interruption. Root
    cause traced to pipecat/turns/user_stop/speech_timeout_user_turn_stop_
    strategy.py: SpeechTimeoutUserTurnStopStrategy._maybe_trigger_user_turn
    _stopped() requires `self._text` to be non-empty (populated only by a
    real TranscriptionFrame reaching it) before it will cleanly signal
    turn-stop. This bridge was consuming the TranscriptionFrame entirely —
    replacing it with a TTSSpeakFrame — so user_aggregator (positioned
    downstream, at the end of the pipeline) never saw it, starving the
    turn-stop strategy and pushing it onto some fallback/timeout path
    instead, which is what threw the stray InterruptionFrame. Fix: forward
    the ORIGINAL TranscriptionFrame downstream too, unmodified, alongside
    the TTSSpeakFrame reply. Confirmed safe to route through `tts`:
    TTSService.process_frame explicitly excludes TranscriptionFrame /
    InterimTranscriptionFrame from its own "speak this text" logic, so
    Piper won't echo the caller's own words back.

    LATENCY MITIGATION (Bluejay's "12 Ways to Reduce Voice Agent Latency" —
    getbluejay.ai — #1 pick, "Thinking Phrases"): graph.invoke() is a
    blocking round trip to OpenAI, potentially multi-hop if a tool
    (check_availability / find_patient_appointments / search_clinic_faq)
    gets called before the final answer. That whole time was previously
    dead air. A short, hardcoded acknowledgment is now spoken via
    TTSSpeakFrame IMMEDIATELY on receiving the transcript, before
    graph.invoke() is even called — this requires NO changes to
    langgraph_app/graph.py, since it's purely about what the voice layer
    says while waiting, not about the reasoning itself.
    THIS IS THE MVP VERSION: one fixed line, spoken on every turn,
    regardless of whether this turn will actually be slow (e.g. a quick
    FAQ answer will also get the filler, even though it didn't need one).
    A smarter, context-aware version — only speaking a filler when a tool
    call is actually about to happen, or picking a filler that matches
    what's being checked — would require langgraph_app/graph.py to expose
    a signal for that (e.g. via graph.stream() intermediate state, or an
    early lightweight intent-classification node). That's a separate,
    optional follow-up for whoever owns graph.py — not needed for this
    MVP version.
    """

    # MVP thinking phrases: one fixed pool, chosen at random per turn to
    # reduce repetition (per Bluejay's suggestion). Kept short and generic
    # since this version has no idea yet whether a tool call is coming.
    THINKING_PHRASES = [
        "Let me check that for you.",
        "One moment, please.",
        "Let me look into that for you.",
        "Give me just a second.",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.turn_count = 0
        self.state = {
            "messages": [], "intent": None, "patient_name": None, "provider": None,
            "appointment_type": None, "requested_datetime": None, "contact_info": None,
            "target_appointment_id": None, "missing_slots": [], "awaiting_confirmation": False,
            "out_of_scope_flag": False,
        }

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            return
        elif isinstance(frame, TranscriptionFrame):
            self.turn_count += 1
            logger.info(f"[turn {self.turn_count}] [transcript] {frame.text!r}")
            # Forward the original transcript downstream FIRST, unmodified
            # — user_aggregator at the end of the pipeline needs to see
            # this to correctly resolve turn-stop. See class docstring:
            # withholding this was the actual cause of TTS getting
            # interrupted/discarded before you ever heard it.
            await self.push_frame(frame, direction)

            # THINKING PHRASE — COMMENTED OUT FOR NOW (per request): in
            # practice this was being heard right before the real answer
            # instead of filling the silence during it, on turns where
            # graph.invoke() latency was high. Rest of the pipeline
            # (transcript forwarding, turn counter, latency logging,
            # endpointing) is untouched — this is the only line disabled.
            # filler = random.choice(self.THINKING_PHRASES)
            # logger.debug(f"[thinking-phrase] {filler!r}")
            # await self.push_frame(TTSSpeakFrame(text=filler, append_to_context=False), direction)

            self.state["messages"].append(HumanMessage(content=frame.text))
            # graph.invoke is a blocking sync call (real network round-trip
            # to OpenAI) — MUST run off the event loop via to_thread, same
            # reasoning as the STT/TTS preloading in server_voice_test.py,
            # or it freezes the whole pipeline (audio, VAD, everything) for
            # the duration of the LLM call.
            invoke_start = time.monotonic()
            self.state = await asyncio.to_thread(langgraph_graph.invoke, self.state)
            logger.info(
                f"[turn {self.turn_count}] [latency] graph.invoke took "
                f"{time.monotonic() - invoke_start:.2f}s"
            )
            reply_text = self.state["messages"][-1].content
            logger.info(f"[agent reply] {reply_text!r}")
            await self.push_frame(
                TTSSpeakFrame(text=reply_text, append_to_context=False), direction
            )
        else:
            logger.debug(f"[stt-out] {frame.__class__.__name__}")
            await self.push_frame(frame, direction)


class TTSOutputTap(FrameProcessor):
    """
    Sits between tts and transport.output(). If TTSSpeakFrame still doesn't
    produce sound, this tells us definitively whether Piper generated any
    audio frames at all (TTSAudioRawFrame / TTSStartedFrame /
    TTSStoppedFrame) versus the problem being downstream in
    transport.output()'s delivery to Twilio.
    """

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        logger.debug(f"[tts-out] {frame.__class__.__name__}")
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

    stream_sid = None
    call_sid = None
    while stream_sid is None:
        message = await websocket.receive_text()
        data = json.loads(message)
        if data.get("event") == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]

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
            # NOTE: no vad_analyzer here anymore. As of pipecat 1.0,
            # TransportParams.vad_analyzer was removed. Pydantic silently
            # drops unknown fields, so passing it here used to look fine
            # and do absolutely nothing — no error, just permanent silence.
            # VAD now lives on the user turn aggregator, built below.
            serializer=serializer,
        ),
    )

    # stt/tts are preloaded once at server startup (see `lifespan` above) —
    # no model-loading latency here anymore, just reusing the warm
    # instances. See DEBUGGING NOTES #6 for the reasoning and the
    # single-call-at-a-time caveat.
    if _stt_service is None or _tts_service is None:
        logger.error("STT/TTS not preloaded yet — lifespan startup hasn't finished")
        await websocket.close(code=1011)
        return
    stt = _stt_service
    tts = _tts_service

    # THE ACTUAL FIX (verified against Pipecat's 1.0 migration docs, not
    # guessed): VAD moved off the transport entirely and onto the user turn
    # aggregator. Building this — and including it anywhere in the
    # pipeline — is what causes a SpeechControlParamsFrame to be sent
    # upstream to the transport at pipeline start, which is what actually
    # configures and activates VAD on the transport's input side. Without
    # this aggregator present, the transport's VAD analyzer is simply never
    # set, which is exactly why your frame-tap logs never showed a single
    # VAD-related frame, under any name, despite loud unambiguous speech.
    #
    # We have no LLM in this "dumb" test bot, so we only use the user half
    # of the pair and never touch the assistant half. We also override the
    # default turn-stop strategy: pipecat 1.x defaults to
    # TurnAnalyzerUserTurnStopStrategy (the "smart-turn" semantic model),
    # which would silently download and load an extra ONNX model you didn't
    # ask for. SpeechTimeoutUserTurnStopStrategy gives you the same simple
    # "N seconds of silence = turn over" behavior your original stop_secs
    # was going for, with no extra model.
    # REVERTED: stop_secs=0.4 / user_speech_timeout=0.3 (the tightened
    # values from the previous latency pass) are the suspected cause of a
    # single short utterance ("Hello") firing TWO separate turns — visible
    # as a real reply's "Finished TTS" immediately followed by a NEW
    # filler's "Generating TTS" in the logs, which is consistent with two
    # full LangGraphBridge invocations stacking serially (explaining both
    # the reordering AND the 15-20s latency on a trivial no-tool-call
    # message — that's roughly two slow turns back to back, not one).
    # Reverted to the values confirmed clean back in Step 3a
    # (server_voice_test.py) rather than guessing new numbers. If dupe
    # turns are still confirmed in the logs (see LangGraphBridge's new
    # turn-count logging below) even after this revert, endpointing isn't
    # the actual cause and this should be investigated further rather than
    # tightened again blindly.
    turn_context = LLMContext(messages=[])
    user_aggregator, _assistant_aggregator = LLMContextAggregatorPair(
        turn_context,
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

    pipeline = Pipeline([
        transport.input(),
        VADLogger(),
        stt,
        LangGraphBridge(),   # was FinalTranscriptToTTS() in server_voice_test.py
        tts,
        TTSOutputTap(),
        transport.output(),
        # Positioned at the end deliberately: it never needs to see our
        # transcript/TTS frames, it just needs to exist in the pipeline so
        # its startup handshake with the transport happens. Frame direction
        # for that handshake is upstream, so position here doesn't stop it
        # from reaching the transport.
        user_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=8000,  # Twilio needs 8kHz mu-law on the way out
            # audio_in_sample_rate intentionally left unset (defaults to
            # 16000) so TwilioFrameSerializer's 8kHz->16kHz upsampling stays
            # active for VAD/Whisper. allow_interruptions was also removed
            # in 1.0 (silently ignored) — turn/interruption behavior is now
            # configured via user_turn_strategies above instead.
        ),
    )

    runner = PipelineRunner()
    await runner.run(task)