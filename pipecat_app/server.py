"""
Step 2 of the build plan: Twilio + Pipecat plumbing in isolation, with a
dummy/echo bot. No LangGraph, no STT/TTS/LLM here on purpose — the only
goal is proving the audio transport itself works: does a call connect,
does audio flow both ways cleanly, does VAD fire.

The pipeline here is deliberately trivial: input audio is routed straight
to output, so you should hear your own voice echoed back with a short
delay when you call in. That delay you hear IS real network round-trip
latency — worth listening for, since it's the same floor you'll be
fighting once STT/LLM/TTS are added on top in later steps.

Run from repo root:
    uvicorn pipecat_app.server:app --port 8000 --reload

Or run from this folder:
    uvicorn server:app --port 8000 --reload

Then in a separate terminal:
    ngrok http 8000

Then set your Twilio number's "A call comes in" webhook (Voice, HTTP POST)
to: https://<your-ngrok-domain>/twiml

Requires environment variables (see .env.example):
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer

load_dotenv()

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

app = FastAPI()


@app.post("/twiml")
async def twiml_endpoint(request: Request):
    """
    Twilio hits this when a call comes in. We respond with TwiML telling
    it to open a Media Stream WebSocket back to our /ws endpoint.
    """
    host = request.url.hostname
    # request.url.hostname will be the ngrok domain when accessed through the tunnel
    stream_url = f"wss://{host}/ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""

    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Twilio sends a 'connected' event, then a 'start' event containing
    # streamSid and callSid — we need both before building the serializer.
    stream_sid = None
    call_sid = None

    import json
    while stream_sid is None:
        message = await websocket.receive_text()
        data = json.loads(message)
        if data.get("event") == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]

    print(f"Call started — stream_sid={stream_sid}, call_sid={call_sid}")

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
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    # Echo-only pipeline: audio in -> straight to audio out. No STT, no LLM,
    # no TTS. This proves the transport itself works before adding any
    # reasoning layer on top.
    pipeline = Pipeline([transport.input(), transport.output()])

    task = PipelineTask(pipeline, params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000))

    runner = PipelineRunner()
    await runner.run(task)
