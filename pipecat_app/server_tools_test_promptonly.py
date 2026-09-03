"""
LATENCY-TEST VARIANT — NOT FOR PRODUCTION.

Same as server_tools_test.py, but with SafetyGate's separate LLM
classification call removed entirely. Medical-content refusal is folded
into the main system prompt instead, with a few-shot block, so the model
has to catch it inline while also juggling tools/booking/history.

This exists ONLY to A/B the latency delta against the SafetyGate version.
Known tradeoff (see prior discussion): a prompt-only instruction competing
with tool-calling context, full history, and reasoning_effort="none" is
easier for the model to drift off than an isolated classifier call. Do not
ship this as-is — if the latency win is confirmed, re-add a fixed version
of SafetyGate (context-aware, race-not-gate, with interruption handling)
rather than leaving this permanently.
"""
import os
import sys
import json
import asyncio
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
from pipecat.frames.frames import MetricsFrame, TTSSpeakFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
from pipecat.processors.aggregators.llm_response_universal import UserTurnStrategies
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# NOTE: SafetyGate import intentionally removed for this test variant.
# from safety_gate import SafetyGate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "langgraph_app"))
from tools import (
    check_availability,
    book_appointment,
    list_providers,
    list_appointment_types,
    list_providers_for_appointment_type,
    find_patient_appointments,
    reschedule_appointment,
    cancel_appointment,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from retrieval import search_clinic_faq

load_dotenv(override=True)

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "cbaf8084-f009-4838-a096-07ee2e6612b1")

app = FastAPI()
from dashboard import router as dashboard_router
app.include_router(dashboard_router)


TOOLS = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="check_availability",
        description="Check available appointment slots for a provider, appointment type, and date.",
        properties={
            "provider_name": {"type": "string", "description": "Provider's full name, e.g. 'Dr. Ahmed Raza'"},
            "appointment_type": {"type": "string", "description": "e.g. 'Routine checkup', 'Follow-up', 'New patient consult'"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
        },
        required=["provider_name", "appointment_type", "date"],
    ),
    FunctionSchema(
        name="book_appointment",
        description="Book a new appointment. Only call after confirming details with the caller.",
        properties={
            "provider_name": {"type": "string"},
            "appointment_type": {"type": "string"},
            "patient_name": {"type": "string"},
            "patient_contact": {"type": "string"},
            "requested_datetime": {"type": "string", "description": "e.g. '2026-08-16 14:00'"},
        },
        required=["provider_name", "appointment_type", "patient_name", "patient_contact", "requested_datetime"],
    ),
    FunctionSchema(
        name="list_providers",
        description="List all clinic providers and their specialties.",
        properties={},
        required=[],
    ),
    FunctionSchema(
        name="list_appointment_types",
        description="List appointment types, optionally filtered to what a specific provider offers.",
        properties={"provider_name": {"type": "string", "description": "Optional — omit to list all types"}},
        required=[],
    ),
    FunctionSchema(
        name="list_providers_for_appointment_type",
        description="List only the providers who actually offer a given appointment type. Call this BEFORE offering provider choices once you know the appointment type.",
        properties={"appointment_type": {"type": "string"}},
        required=["appointment_type"],
    ),
    FunctionSchema(
        name="find_patient_appointments",
        description="Find a caller's existing appointments by name, for reschedule/cancel flows.",
        properties={"patient_name": {"type": "string"}},
        required=["patient_name"],
    ),
    FunctionSchema(
        name="reschedule_appointment",
        description="Reschedule an existing appointment to a new date/time. Confirm details first.",
        properties={
            "appointment_id": {"type": "integer"},
            "new_datetime": {"type": "string"},
        },
        required=["appointment_id", "new_datetime"],
    ),
    FunctionSchema(
        name="cancel_appointment",
        description="Cancel an existing appointment. Confirm which one first.",
        properties={"appointment_id": {"type": "integer"}},
        required=["appointment_id"],
    ),
    FunctionSchema(
        name="search_clinic_faq",
        description="Search clinic FAQ/policy knowledge base for factual questions — insurance, hours, location, referrals, what to bring, fasting, cancellation policy, walk-ins, telehealth, prescriptions, etc. Always call this for such questions rather than answering from general knowledge.",
        properties={"query": {"type": "string", "description": "The caller's question, in their own words"}},
        required=["query"],
    ),
])


def _make_handler(fn):
    async def handler(params: FunctionCallParams):
        logger.info(f"[tool-call] {params.function_name}({params.arguments})")
        result = await asyncio.to_thread(fn, **params.arguments)
        logger.info(f"[tool-result] {params.function_name} -> {result}")
        await params.result_callback(result)

    return handler


TOOL_FUNCTIONS = {
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "list_providers": list_providers,
    "list_appointment_types": list_appointment_types,
    "list_providers_for_appointment_type": list_providers_for_appointment_type,
    "find_patient_appointments": find_patient_appointments,
    "reschedule_appointment": reschedule_appointment,
    "cancel_appointment": cancel_appointment,
    "search_clinic_faq": search_clinic_faq,
}


class LatencyMonitor(FrameProcessor):
    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for metric in frame.data:
                if isinstance(metric, TTFBMetricsData):
                    logger.info(f"[latency] {metric.processor}: {metric.value * 1000:.0f}ms")
        await self.push_frame(frame, direction)


class InterruptionGate(FrameProcessor):
    def __init__(self, turn_strategy, **kwargs):
        super().__init__(**kwargs)
        self._strategy = turn_strategy
        self._strategy._enable_interruptions = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._strategy._enable_interruptions = True
            logger.debug("[interruption-gate] armed — bot is speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._strategy._enable_interruptions = False
            logger.debug("[interruption-gate] disarmed — bot is silent")
        await self.push_frame(frame, direction)


@app.post("/twiml")
async def twiml_endpoint(request: Request):
    host = request.url.hostname
    stream_url = f"wss://{host}/ws"
    form = await request.form()
    caller_number = form.get("From", "")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}">
            <Parameter name="callerNumber" value="{caller_number}" />
        </Stream>
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
            caller_number = data["start"].get("customParameters", {}).get("callerNumber", "")

    logger.info(f"Call started — stream_sid={stream_sid}, call_sid={call_sid}")

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid, call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID, auth_token=TWILIO_AUTH_TOKEN,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, serializer=serializer),
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        ttfs_p99_latency=0.9,
        settings=DeepgramSTTService.Settings(model="nova-3", language="en"),
    )

    llm = OpenAILLMService(
        settings=OpenAILLMService.Settings(model="gpt-5.4-nano", extra={"reasoning_effort": "none"}),
        api_key=OPENAI_API_KEY,
    )

    for name, fn in TOOL_FUNCTIONS.items():
        llm.register_function(name, _make_handler(fn))

    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        settings=CartesiaTTSService.Settings(voice=CARTESIA_VOICE_ID, model="sonic-2"),
    )

    # --- Safety folded into the main prompt, few-shot, no separate gate. ---
    # TEST VARIANT ONLY — see module docstring for the tradeoff being measured.
    system_prompt = """You are a phone receptionist for Health1st Clinic.

You do NOT give medical advice, diagnoses, or symptom guidance under any
circumstance. If a caller describes symptoms, asks whether something is
serious, or asks about medication (dosage, interactions, side effects),
say you can't advise on that and offer to connect them to clinic staff,
then continue with booking if they still want that.

Do NOT treat these as medical content — respond normally:
- "I need to see a cardiologist" -> fine, that's picking a provider type.
- "My knee's been bothering me, can I still come in for my checkup" ->
  fine, that's booking context, not a request for advice.
- "Do you take my insurance" / "what should I bring" -> fine, clinic
  policy question, use search_clinic_faq.

DO treat these as medical content requiring the refusal above:
- "Should I be worried about this pain" -> refuse, offer to connect to staff.
- "What dosage of X should I take" -> refuse, offer to connect to staff.
- "Is this serious / do I need an ER" -> refuse, offer to connect to staff,
  and if it sounds urgent, mention emergency services.

You help callers book, reschedule, or cancel appointments. Ask for missing
details one or two at a time — never list out three or four questions at
once. To book, you need: patient name, provider or specialty, appointment
type, and a requested date/time. Callers won't use exact clinic terms — if
unsure of exact provider/appointment-type names, call list_providers and/or
list_appointment_types first rather than guessing. If the caller names an
appointment type before a provider, call list_providers_for_appointment_type
first and only offer providers who actually offer that type.

Use check_availability before booking to confirm the slot is open. For
reschedule/cancel, use find_patient_appointments to find their existing
appointment(s) — if there's more than one, ask which one before acting.

Always restate the details and get explicit confirmation before calling
book_appointment, reschedule_appointment, or cancel_appointment.

For factual/policy questions about the clinic (insurance, hours, location,
referrals, what to bring, fasting, cancellation policy, walk-ins, telehealth,
prescriptions, etc.) — always call search_clinic_faq first rather than
answering from your own general knowledge. Base your answer only on what
the tool returns. If it doesn't return anything relevant, say you don't
have that info and offer to connect them to clinic staff — never guess.

If a caller states two conflicting values for the same detail close
together in one turn (e.g. "three AM — three PM", "Tuesday, no, Wednesday"),
that's a normal self-correction, not real ambiguity — treat the LATER value
as the correct one and proceed, don't ask them to clarify which they meant.
Only ask for clarification if genuinely unclear which value is the
correction (e.g. they're said far apart, or with no correcting tone like
"or").

Keep responses to 1-2 short sentences — this is a phone call, not a chat window."""

    if caller_number:
        system_prompt += (
            f"\n\nThe caller's phone number, from caller ID, is {caller_number}. "
            f"If they ask you to use 'the number I'm calling from' or don't offer a "
            f"different contact number, use this one — don't ask them to repeat it."
        )

    context = LLMContext(messages=[{"role": "system", "content": system_prompt}], tools=TOOLS)

    start_strategy = VADUserTurnStartStrategy()

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.7, min_volume=0.6, start_secs=0.1, stop_secs=0.7)),
            user_turn_strategies=UserTurnStrategies(
                start=[start_strategy],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.7)],
            ),
        ),
    )

    latency_monitor = LatencyMonitor()
    interruption_gate = InterruptionGate(start_strategy)

    # NOTE: safety_gate removed from this pipeline for the latency test.
    # Compare timestamps in LatencyMonitor output against the SafetyGate
    # version to measure the delta this one processor was adding.
    pipeline = Pipeline([
        transport.input(), stt, latency_monitor, user_aggregator,
        llm, tts, transport.output(), interruption_gate, assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(audio_out_sample_rate=8000, enable_metrics=True))

    await task.queue_frames([TTSSpeakFrame(text="Hi, thanks for calling Health1st Clinic! How can I help you today?")])

    runner = PipelineRunner()
    await runner.run(task)
