"""
Step 3a of the build plan: add real STT (Whisper) + TTS (Piper) to the
pipeline, but keep it "dumb" — it just repeats back what you said, no
LangGraph, no reasoning. This isolates STT/TTS correctness from reasoning
correctness, same principle as the echo-bot step: one new variable at a
time. If something sounds wrong here, you know it's STT or TTS, not your
conversation logic.

Only once this sounds right do we move to Step 3b: replacing the "repeat
it back" bridge with an actual call into langgraph_app/graph.py.

Run:
    uvicorn server_voice_test:app --port 8000 --reload

Same ngrok/Twilio webhook setup as before, pointed at THIS file's /twiml
and /ws endpoints instead of server.py's.

Requires (on top of what server.py needed):
    pip install "pipecat-ai[whisper]" "pipecat-ai[piper]"

First real call will be slow to connect — Whisper's model and Piper's
voice model both download on first use and get cached locally afterward.

DEBUGGING NOTES:
1. CORRECTED: the original assumption here was that setting
   no_speech_prob=None would disable filtering, since Twilio's noisy 8kHz
   audio can score every segment above the default threshold (0.4) and
   WhisperSTTService.run_stt() silently yields nothing in that case — no
   TranscriptionFrame, no ErrorFrame, no exception. That diagnosis of the
   *symptom* was right, but the fix was backwards. Verified by downloading
   pipecat-ai==1.7.0 and reading services/whisper/stt.py directly: the
   filter is `if threshold is not None and segment.no_speech_prob <
   threshold: text += segment.text`. Passing None makes the `is not None`
   check always False, so NO segment is ever included, for any call, ever
   — the opposite of "disable filtering." Since segment.no_speech_prob is
   always in [0, 1], the actual fix is a threshold ABOVE that range
   (no_speech_prob=1.01 below), which makes every real segment satisfy
   "no_speech_prob < threshold" and get included. See the STT construction
   below for the corrected value.
2. print() output can get block-buffered (rather than line-buffered) when
   running under `uvicorn --reload` on Windows, since the reload subprocess
   isn't attached to a real terminal — meaning print() lines can silently
   sit in a buffer and never actually appear while the process is running.
   Every status line below uses logger.info()/logger.debug() instead, which
   loguru flushes immediately, so nothing you need to see gets stuck.
3. A VADLogger processor is added right after transport.input() to log
   every UserStartedSpeakingFrame / UserStoppedSpeakingFrame. This lets you
   confirm VAD is actually firing on the Twilio audio at all — the earliest
   possible point of failure, and one entirely independent of Whisper/Piper.
   If you never see "VAD: user started speaking" while talking, the issue
   is upstream of STT entirely (mic audio/VAD), not the no_speech_prob filter.
4. FIXED BUG: PipelineParams previously had audio_in_sample_rate=8000 set
   alongside audio_out_sample_rate=8000. That was the actual cause of total
   silence on every call. TwilioFrameSerializer normally upsamples Twilio's
   8kHz mu-law audio to 16kHz before it reaches the rest of the pipeline,
   which is the rate SileroVADAnalyzer and WhisperSTTService are built to
   consume. Forcing audio_in_sample_rate=8000 disabled that upsampling, so
   VAD received 16kHz-window-sized chunks of audio that was actually 8kHz —
   its speech probability score never crossed the threshold, no matter how
   loud you talked (confirmed by peak amplitude logs showing real signal
   while zero "VAD: user started speaking" lines ever printed). Since
   WhisperSTTService is a SegmentedSTTService, it does nothing until VAD
   tells it a speech segment just ended — so VAD never firing meant Whisper
   never ran, no transcript was ever produced, and Piper never had anything
   to speak. Only audio_out_sample_rate needs to be pinned to 8000 for
   Twilio's outbound leg; audio_in_sample_rate is left at its default
   (16000) so the serializer's upsampling — and therefore VAD/STT — works
   correctly. SileroVADAnalyzer is also left at its default 16000 sample
   rate to match.
   CORRECTION: this alone did not fix it (see note 5) — kept here because
   it's still correct and necessary, just not sufficient on its own.
5. THE ACTUAL ROOT CAUSE (confirmed against Pipecat's official 1.0
   migration guide, not guessed): you're on pipecat-ai 1.7.0. As of
   Pipecat 1.0, `vad_analyzer` was REMOVED as a transport parameter
   (`FastAPIWebsocketParams`/`TransportParams`). Removed params are
   dropped silently by Pydantic — no error, no warning. The pipeline
   linked up fine, audio flowed fine, but the transport's VAD analyzer was
   simply never set, so it never ran, no matter how loud you talked. A
   frame-tap of every non-audio frame confirmed this directly: after 25+
   seconds of loud speech, only StartFrame / ClientConnectedFrame /
   STTMetadataFrame / OutputTransportReadyFrame ever appeared — never
   anything VAD-related, under any name.
   VAD now lives on the *user turn aggregator*
   (`LLMUserAggregatorParams.vad_analyzer`, from
   `pipecat.processors.aggregators.llm_response_universal`). Constructing
   that aggregator and including it anywhere in the pipeline is what sends
   a `SpeechControlParamsFrame` upstream to the transport at pipeline
   start — that's the actual mechanism that configures and activates VAD
   on the transport's input side. Without it, nothing configures VAD, ever.
   Since this test bot has no LLM, only the "user" half of the pair is
   used, positioned at the end of the pipeline (position doesn't matter
   for this upstream handshake — direction does, not order). The default
   turn-stop strategy in 1.x is also overridden
   (`SpeechTimeoutUserTurnStopStrategy` instead of the default
   `TurnAnalyzerUserTurnStopStrategy`), to avoid silently pulling in and
   loading the extra "smart-turn" semantic ONNX model neither asked for
   nor needed here — a plain silence-timeout matches what stop_secs was
   already doing conceptually.
6. LATENCY FIX: stt/tts were previously built fresh inside
   websocket_endpoint — meaning Whisper's ~3s model load and Piper's ~4-5s
   voice load both happened AFTER the caller was already connected and
   waiting, adding that whole delay before the first response could even
   start. WhisperSTTService._load() (confirmed in source) always
   constructs a brand-new faster_whisper.WhisperModel from scratch — there
   is no "reuse an already-loaded model" hook — so this cost is unavoidable
   per construction, but avoidable per-CALL. Both services are now built
   once in a FastAPI `lifespan` startup block, before Uvicorn ever starts
   accepting connections, and reused for the call. Since your local
   ngrok/Twilio webhook only becomes reachable once the server has fully
   started, this moves the entire load delay to before any real call can
   possibly arrive — callers now hit an already-warm pipeline.
   CAVEAT: this reuses the same two service instances across calls, which
   is fine for one call at a time (which is all this test targets), but a
   FrameProcessor is generally only safe to belong to one active Pipeline
   at once. If you later need to handle multiple SIMULTANEOUS calls, these
   would need to become a small pool of pre-warmed instances (or per-call
   instances wrapping a shared underlying model object) rather than a
   single shared pair — flag this before Step 3b if concurrent calls are a
   real requirement.
"""
import os
import sys
import json
import time
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
        model=WhisperModel.BASE,
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


class FinalTranscriptToTTS(FrameProcessor):
    """
    STT emits BOTH InterimTranscriptionFrame (partial, updates as the
    caller speaks) and TranscriptionFrame (final, once they pause) — and
    both are TextFrame subclasses, so TTS would try to speak every partial
    fragment if wired directly. This bridge only forwards final
    transcripts, and DIAGNOSTIC HISTORY below explains why it forwards a
    TTSSpeakFrame rather than a plain TextFrame.

    CONFIRMED BUG #2 (source-verified, not guessed): Whisper was fixed and
    is now transcribing correctly ("Hello, can you hear me?" showed up
    clean in the logs), but Piper still produced no audible output and no
    errors. Read pipecat/services/tts_service.py's TTSService.process_frame
    directly: a plain TextFrame only gets flushed to run_tts() once an
    LLMFullResponseEndFrame (or EndFrame) arrives — that's the signal the
    base class uses to know a "turn" of text is complete and should
    actually be synthesized. Since this bot has no LLM, no
    LLMFullResponseStartFrame/EndFrame pair is ever sent, so a bare
    TextFrame just sits in Piper's internal aggregation buffer forever,
    with nothing wrong enough to log an error about.
    TTSSpeakFrame exists specifically for this case: a standalone, one-off
    TTS request that creates its own context_id and synthesizes
    immediately, with no LLM turn bracketing required.
    """

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            logger.debug(f"[stt-out] InterimTranscriptionFrame: {frame.text!r}")
            return  # drop partials entirely, don't forward
        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"[transcript] {frame.text!r}")
            # append_to_context=False: our LLMContext only exists to host
            # VAD on the user-turn aggregator (see websocket_endpoint) —
            # it's not wired to an assistant aggregator, so there's nothing
            # meaningful to append these utterances to.
            await self.push_frame(
                TTSSpeakFrame(text=frame.text, append_to_context=False), direction
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
        FinalTranscriptToTTS(),
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




# """
# Step 3a of the build plan: add real STT (Whisper) + TTS (Piper) to the
# pipeline, but keep it "dumb" — it just repeats back what you said, no
# LangGraph, no reasoning. This isolates STT/TTS correctness from reasoning
# correctness, same principle as the echo-bot step: one new variable at a
# time. If something sounds wrong here, you know it's STT or TTS, not your
# conversation logic.

# Only once this sounds right do we move to Step 3b: replacing the "repeat
# it back" bridge with an actual call into langgraph_app/graph.py.

# Run:
#     uvicorn server_voice_test:app --port 8000 --reload

# Same ngrok/Twilio webhook setup as before, pointed at THIS file's /twiml
# and /ws endpoints instead of server.py's.

# Requires (on top of what server.py needed):
#     pip install "pipecat-ai[whisper]" "pipecat-ai[piper]"

# First real call will be slow to connect — Whisper's model and Piper's
# voice model both download on first use and get cached locally afterward.

# DEBUGGING NOTES:
# 1. CORRECTED: the original assumption here was that setting
#    no_speech_prob=None would disable filtering, since Twilio's noisy 8kHz
#    audio can score every segment above the default threshold (0.4) and
#    WhisperSTTService.run_stt() silently yields nothing in that case — no
#    TranscriptionFrame, no ErrorFrame, no exception. That diagnosis of the
#    *symptom* was right, but the fix was backwards. Verified by downloading
#    pipecat-ai==1.7.0 and reading services/whisper/stt.py directly: the
#    filter is `if threshold is not None and segment.no_speech_prob <
#    threshold: text += segment.text`. Passing None makes the `is not None`
#    check always False, so NO segment is ever included, for any call, ever
#    — the opposite of "disable filtering." Since segment.no_speech_prob is
#    always in [0, 1], the actual fix is a threshold ABOVE that range
#    (no_speech_prob=1.01 below), which makes every real segment satisfy
#    "no_speech_prob < threshold" and get included. See the STT construction
#    below for the corrected value.
# 2. print() output can get block-buffered (rather than line-buffered) when
#    running under `uvicorn --reload` on Windows, since the reload subprocess
#    isn't attached to a real terminal — meaning print() lines can silently
#    sit in a buffer and never actually appear while the process is running.
#    Every status line below uses logger.info()/logger.debug() instead, which
#    loguru flushes immediately, so nothing you need to see gets stuck.
# 3. A VADLogger processor is added right after transport.input() to log
#    every UserStartedSpeakingFrame / UserStoppedSpeakingFrame. This lets you
#    confirm VAD is actually firing on the Twilio audio at all — the earliest
#    possible point of failure, and one entirely independent of Whisper/Piper.
#    If you never see "VAD: user started speaking" while talking, the issue
#    is upstream of STT entirely (mic audio/VAD), not the no_speech_prob filter.
# 4. FIXED BUG: PipelineParams previously had audio_in_sample_rate=8000 set
#    alongside audio_out_sample_rate=8000. That was the actual cause of total
#    silence on every call. TwilioFrameSerializer normally upsamples Twilio's
#    8kHz mu-law audio to 16kHz before it reaches the rest of the pipeline,
#    which is the rate SileroVADAnalyzer and WhisperSTTService are built to
#    consume. Forcing audio_in_sample_rate=8000 disabled that upsampling, so
#    VAD received 16kHz-window-sized chunks of audio that was actually 8kHz —
#    its speech probability score never crossed the threshold, no matter how
#    loud you talked (confirmed by peak amplitude logs showing real signal
#    while zero "VAD: user started speaking" lines ever printed). Since
#    WhisperSTTService is a SegmentedSTTService, it does nothing until VAD
#    tells it a speech segment just ended — so VAD never firing meant Whisper
#    never ran, no transcript was ever produced, and Piper never had anything
#    to speak. Only audio_out_sample_rate needs to be pinned to 8000 for
#    Twilio's outbound leg; audio_in_sample_rate is left at its default
#    (16000) so the serializer's upsampling — and therefore VAD/STT — works
#    correctly. SileroVADAnalyzer is also left at its default 16000 sample
#    rate to match.
#    CORRECTION: this alone did not fix it (see note 5) — kept here because
#    it's still correct and necessary, just not sufficient on its own.
# 5. THE ACTUAL ROOT CAUSE (confirmed against Pipecat's official 1.0
#    migration guide, not guessed): you're on pipecat-ai 1.7.0. As of
#    Pipecat 1.0, `vad_analyzer` was REMOVED as a transport parameter
#    (`FastAPIWebsocketParams`/`TransportParams`). Removed params are
#    dropped silently by Pydantic — no error, no warning. The pipeline
#    linked up fine, audio flowed fine, but the transport's VAD analyzer was
#    simply never set, so it never ran, no matter how loud you talked. A
#    frame-tap of every non-audio frame confirmed this directly: after 25+
#    seconds of loud speech, only StartFrame / ClientConnectedFrame /
#    STTMetadataFrame / OutputTransportReadyFrame ever appeared — never
#    anything VAD-related, under any name.
#    VAD now lives on the *user turn aggregator*
#    (`LLMUserAggregatorParams.vad_analyzer`, from
#    `pipecat.processors.aggregators.llm_response_universal`). Constructing
#    that aggregator and including it anywhere in the pipeline is what sends
#    a `SpeechControlParamsFrame` upstream to the transport at pipeline
#    start — that's the actual mechanism that configures and activates VAD
#    on the transport's input side. Without it, nothing configures VAD, ever.
#    Since this test bot has no LLM, only the "user" half of the pair is
#    used, positioned at the end of the pipeline (position doesn't matter
#    for this upstream handshake — direction does, not order). The default
#    turn-stop strategy in 1.x is also overridden
#    (`SpeechTimeoutUserTurnStopStrategy` instead of the default
#    `TurnAnalyzerUserTurnStopStrategy`), to avoid silently pulling in and
#    loading the extra "smart-turn" semantic ONNX model neither asked for
#    nor needed here — a plain silence-timeout matches what stop_secs was
#    already doing conceptually.
# """
# import os
# import sys
# import json
# import time
# import asyncio
# import numpy as np
# from dotenv import load_dotenv
# from loguru import logger
# from fastapi import FastAPI, Request, WebSocket
# from fastapi.responses import PlainTextResponse

# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.pipeline.runner import PipelineRunner
# from pipecat.pipeline.task import PipelineTask, PipelineParams
# from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
# from pipecat.serializers.twilio import TwilioFrameSerializer
# from pipecat.audio.vad.silero import SileroVADAnalyzer
# from pipecat.audio.vad.vad_analyzer import VADParams
# from pipecat.processors.aggregators.llm_context import LLMContext
# from pipecat.processors.aggregators.llm_response_universal import (
#     LLMContextAggregatorPair,
#     LLMUserAggregatorParams,
# )
# from pipecat.turns.user_start import VADUserTurnStartStrategy
# from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
# from pipecat.turns.user_turn_strategies import UserTurnStrategies
# from pipecat.frames.frames import (
#     TranscriptionFrame,
#     InterimTranscriptionFrame,
#     TextFrame,
#     TTSSpeakFrame,
#     VADUserStartedSpeakingFrame,
#     VADUserStoppedSpeakingFrame,
#     InputAudioRawFrame,
# )
# from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
# from pipecat.services.whisper.stt import WhisperSTTService, Model as WhisperModel
# from pipecat.services.piper.tts import PiperTTSService

# load_dotenv()

# # DEBUG logging so you can see exactly what STT is (or isn't) transcribing
# # in real time — this is what would have surfaced the no_speech_prob issue
# # immediately instead of just hearing dead air.
# logger.remove()
# logger.add(sys.stderr, level="DEBUG")

# # Belt-and-suspenders fix for the print()-buffering issue: force stdout to
# # line-buffer even when it's not attached to a real terminal (which is the
# # case under uvicorn --reload on Windows). Not strictly needed anymore since
# # every print() below has been replaced with logger calls, but harmless to
# # keep in case you add more print() debugging later.
# try:
#     sys.stdout.reconfigure(line_buffering=True)
# except AttributeError:
#     pass


# class VADLogger(FrameProcessor):
#     """
#     Confirms VAD is actually firing on the incoming Twilio audio, completely
#     independent of whether Whisper ever produces a transcript. If you don't
#     see these log lines while talking, the problem is upstream of STT.

#     Listens for VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame —
#     the raw signal straight out of Silero VAD, and the same frames
#     SegmentedSTTService itself listens for to know when to buffer/flush
#     audio. (Note: these are NOT the same classes as the plain
#     UserStartedSpeakingFrame/UserStoppedSpeakingFrame — there's no
#     inheritance relationship between them, so checking for the wrong one
#     silently never fires, regardless of whether VAD is actually working.)

#     Also logs a running count of raw InputAudioRawFrame chunks once a
#     second, so you can see — independent of VAD entirely — whether audio
#     is even still flowing from Twilio at each point in the call. Alongside
#     that, logs the peak sample amplitude seen in that window (0.0-1.0,
#     where 1.0 is digital full-scale) as a sanity check on whether the
#     decoded audio actually has real signal in it, independent of Silero's
#     own volume gate.
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._audio_frame_count = 0
#         self._last_log_time = 0.0
#         self._peak_amplitude = 0.0

#     async def process_frame(self, frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#         if isinstance(frame, VADUserStartedSpeakingFrame):
#             logger.info("VAD: user started speaking")
#         elif isinstance(frame, VADUserStoppedSpeakingFrame):
#             logger.info("VAD: user stopped speaking")
#         elif isinstance(frame, InputAudioRawFrame):
#             self._audio_frame_count += 1
#             samples = np.frombuffer(frame.audio, dtype=np.int16)
#             if samples.size:
#                 peak = float(np.abs(samples).max()) / 32768.0
#                 self._peak_amplitude = max(self._peak_amplitude, peak)
#             now = time.monotonic()
#             if now - self._last_log_time >= 1.0:
#                 logger.info(
#                     f"Audio in: {self._audio_frame_count} chunks so far, "
#                     f"peak amplitude this window: {self._peak_amplitude:.4f}"
#                 )
#                 self._last_log_time = now
#                 self._peak_amplitude = 0.0
#         else:
#             # DIAGNOSTIC: we've now confirmed twice that no
#             # VADUserStartedSpeakingFrame ever reaches this processor, even
#             # with loud, unambiguous speech (peak amplitude 0.93) hitting
#             # the transport. Rather than guess a third blind fix, log the
#             # class name of literally every non-audio frame that passes
#             # through here. This tells us the ground truth for this specific
#             # pipecat version: whether VAD frames are (a) never generated at
#             # all, (b) generated under a different/renamed class, or (c)
#             # generated but consumed internally by the transport's turn
#             # controller before ever reaching the pipeline's processor
#             # chain. Whichever of those it is changes the fix completely,
#             # so don't skip reading these logs.
#             logger.debug(f"[frame-tap] {frame.__class__.__name__}")
#         await self.push_frame(frame, direction)

# TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# app = FastAPI()


# class FinalTranscriptToTTS(FrameProcessor):
#     """
#     STT emits BOTH InterimTranscriptionFrame (partial, updates as the
#     caller speaks) and TranscriptionFrame (final, once they pause) — and
#     both are TextFrame subclasses, so TTS would try to speak every partial
#     fragment if wired directly. This bridge only forwards final
#     transcripts, and DIAGNOSTIC HISTORY below explains why it forwards a
#     TTSSpeakFrame rather than a plain TextFrame.

#     CONFIRMED BUG #2 (source-verified, not guessed): Whisper was fixed and
#     is now transcribing correctly ("Hello, can you hear me?" showed up
#     clean in the logs), but Piper still produced no audible output and no
#     errors. Read pipecat/services/tts_service.py's TTSService.process_frame
#     directly: a plain TextFrame only gets flushed to run_tts() once an
#     LLMFullResponseEndFrame (or EndFrame) arrives — that's the signal the
#     base class uses to know a "turn" of text is complete and should
#     actually be synthesized. Since this bot has no LLM, no
#     LLMFullResponseStartFrame/EndFrame pair is ever sent, so a bare
#     TextFrame just sits in Piper's internal aggregation buffer forever,
#     with nothing wrong enough to log an error about.
#     TTSSpeakFrame exists specifically for this case: a standalone, one-off
#     TTS request that creates its own context_id and synthesizes
#     immediately, with no LLM turn bracketing required.
#     """

#     async def process_frame(self, frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         if isinstance(frame, InterimTranscriptionFrame):
#             logger.debug(f"[stt-out] InterimTranscriptionFrame: {frame.text!r}")
#             return  # drop partials entirely, don't forward
#         elif isinstance(frame, TranscriptionFrame):
#             logger.info(f"[transcript] {frame.text!r}")
#             # append_to_context=False: our LLMContext only exists to host
#             # VAD on the user-turn aggregator (see websocket_endpoint) —
#             # it's not wired to an assistant aggregator, so there's nothing
#             # meaningful to append these utterances to.
#             await self.push_frame(
#                 TTSSpeakFrame(text=frame.text, append_to_context=False), direction
#             )
#         else:
#             logger.debug(f"[stt-out] {frame.__class__.__name__}")
#             await self.push_frame(frame, direction)


# class TTSOutputTap(FrameProcessor):
#     """
#     Sits between tts and transport.output(). If TTSSpeakFrame still doesn't
#     produce sound, this tells us definitively whether Piper generated any
#     audio frames at all (TTSAudioRawFrame / TTSStartedFrame /
#     TTSStoppedFrame) versus the problem being downstream in
#     transport.output()'s delivery to Twilio.
#     """

#     async def process_frame(self, frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#         logger.debug(f"[tts-out] {frame.__class__.__name__}")
#         await self.push_frame(frame, direction)


# @app.post("/twiml")
# async def twiml_endpoint(request: Request):
#     host = request.url.hostname
#     stream_url = f"wss://{host}/ws"
#     twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
# <Response>
#     <Connect>
#         <Stream url="{stream_url}" />
#     </Connect>
# </Response>"""
#     return PlainTextResponse(content=twiml, media_type="application/xml")


# @app.post("/status")
# async def status_callback(request: Request):
#     data = await request.form()
#     logger.info(f"[status] Call {data.get('CallSid')}: {data.get('CallStatus')}")
#     return PlainTextResponse(content="", status_code=204)


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     stream_sid = None
#     call_sid = None
#     while stream_sid is None:
#         message = await websocket.receive_text()
#         data = json.loads(message)
#         if data.get("event") == "start":
#             stream_sid = data["start"]["streamSid"]
#             call_sid = data["start"]["callSid"]

#     logger.info(f"Call started — stream_sid={stream_sid}, call_sid={call_sid}")

#     serializer = TwilioFrameSerializer(
#         stream_sid=stream_sid,
#         call_sid=call_sid,
#         account_sid=TWILIO_ACCOUNT_SID,
#         auth_token=TWILIO_AUTH_TOKEN,
#     )

#     transport = FastAPIWebsocketTransport(
#         websocket=websocket,
#         params=FastAPIWebsocketParams(
#             audio_in_enabled=True,
#             audio_out_enabled=True,
#             # NOTE: no vad_analyzer here anymore. As of pipecat 1.0,
#             # TransportParams.vad_analyzer was removed. Pydantic silently
#             # drops unknown fields, so passing it here used to look fine
#             # and do absolutely nothing — no error, just permanent silence.
#             # VAD now lives on the user turn aggregator, built below.
#             serializer=serializer,
#         ),
#     )

#     # stt/tts construction blocks on synchronous model loading (Whisper ~3s,
#     # Piper ~4s in your logs). Building them directly in this coroutine
#     # freezes the entire asyncio event loop for that whole window — which
#     # means it can't service the Twilio WebSocket either, right as the call
#     # is starting. Running the constructors in a thread via asyncio.to_thread
#     # keeps the event loop free to keep reading Twilio's audio the whole time.
#     def _build_stt():
#         settings = WhisperSTTService.Settings(
#             model=WhisperModel.BASE,
#             # CONFIRMED BUG (verified by downloading pipecat-ai==1.7.0 and
#             # reading services/whisper/stt.py directly, not guessed):
#             # run_stt()'s filter is:
#             #   if no_speech_prob_threshold is not None and
#             #      segment.no_speech_prob < no_speech_prob_threshold:
#             #          text += segment.text
#             # Setting no_speech_prob=None does NOT disable filtering — it
#             # makes the "is not None" check always False, so NO segment's
#             # text is ever appended, for any call, ever. That's the exact
#             # total-silence-no-error symptom we've been chasing. Since
#             # segment.no_speech_prob is always in [0, 1], setting the
#             # threshold above that range makes "segment.no_speech_prob <
#             # threshold" true for every real segment, which is what
#             # actually achieves "accept everything" with this code.
#             no_speech_prob=1.01,
#         )
#         logger.debug(f"WhisperSTTService.Settings actually built: {vars(settings)}")
#         return WhisperSTTService(settings=settings)

#     def _build_tts():
#         return PiperTTSService(settings=PiperTTSService.Settings(voice="en_US-lessac-medium"))

#     stt = await asyncio.to_thread(_build_stt)
#     tts = await asyncio.to_thread(_build_tts)

#     # THE ACTUAL FIX (verified against Pipecat's 1.0 migration docs, not
#     # guessed): VAD moved off the transport entirely and onto the user turn
#     # aggregator. Building this — and including it anywhere in the
#     # pipeline — is what causes a SpeechControlParamsFrame to be sent
#     # upstream to the transport at pipeline start, which is what actually
#     # configures and activates VAD on the transport's input side. Without
#     # this aggregator present, the transport's VAD analyzer is simply never
#     # set, which is exactly why your frame-tap logs never showed a single
#     # VAD-related frame, under any name, despite loud unambiguous speech.
#     #
#     # We have no LLM in this "dumb" test bot, so we only use the user half
#     # of the pair and never touch the assistant half. We also override the
#     # default turn-stop strategy: pipecat 1.x defaults to
#     # TurnAnalyzerUserTurnStopStrategy (the "smart-turn" semantic model),
#     # which would silently download and load an extra ONNX model you didn't
#     # ask for. SpeechTimeoutUserTurnStopStrategy gives you the same simple
#     # "N seconds of silence = turn over" behavior your original stop_secs
#     # was going for, with no extra model.
#     turn_context = LLMContext(messages=[])
#     user_aggregator, _assistant_aggregator = LLMContextAggregatorPair(
#         turn_context,
#         user_params=LLMUserAggregatorParams(
#             vad_analyzer=SileroVADAnalyzer(
#                 params=VADParams(confidence=0.5, min_volume=0.3, start_secs=0.1, stop_secs=0.5)
#             ),
#             user_turn_strategies=UserTurnStrategies(
#                 start=[VADUserTurnStartStrategy()],
#                 stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.5)],
#             ),
#         ),
#     )

#     pipeline = Pipeline([
#         transport.input(),
#         VADLogger(),
#         stt,
#         FinalTranscriptToTTS(),
#         tts,
#         TTSOutputTap(),
#         transport.output(),
#         # Positioned at the end deliberately: it never needs to see our
#         # transcript/TTS frames, it just needs to exist in the pipeline so
#         # its startup handshake with the transport happens. Frame direction
#         # for that handshake is upstream, so position here doesn't stop it
#         # from reaching the transport.
#         user_aggregator,
#     ])

#     task = PipelineTask(
#         pipeline,
#         params=PipelineParams(
#             audio_out_sample_rate=8000,  # Twilio needs 8kHz mu-law on the way out
#             # audio_in_sample_rate intentionally left unset (defaults to
#             # 16000) so TwilioFrameSerializer's 8kHz->16kHz upsampling stays
#             # active for VAD/Whisper. allow_interruptions was also removed
#             # in 1.0 (silently ignored) — turn/interruption behavior is now
#             # configured via user_turn_strategies above instead.
#         ),
#     )

#     runner = PipelineRunner()
#     await runner.run(task)