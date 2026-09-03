"""
Direct port of the LangGraph safety_check node's design into the
Pipecat-native pipeline: a SEPARATE, narrow LLM call that classifies only
the caller's latest message — no tool-calling context, no conversation
history, one job only. Deliberately NOT reusing the main conversational
`llm` object, for the same reason as the original: keeping this call's
only responsibility simple and hard to derail via prompt injection or
conversational context bleed.
"""
import os
from openai import AsyncOpenAI

# Lazy client construction — deliberately NOT built at module import time.
# Constructing it eagerly would read os.environ before the importing
# module's load_dotenv() has necessarily run yet (import order dependent,
# confirmed to break this exact way when SafetyGate is imported before
# load_dotenv() executes). Building it on first real use sidesteps that
# entirely.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client

SAFETY_CLASSIFIER_PROMPT = """You are a safety classifier for a clinic phone system.
Look ONLY at the caller's latest message below and decide: does it contain medical
content that should NOT be handled by a booking assistant? This includes: describing
symptoms, asking for a diagnosis, asking whether something is serious, asking about
medication (dosage, interactions, side effects), or asking for medical advice of any kind.

Do NOT flag: normal booking requests, names of specialties/appointment types
(e.g. "I need to see a cardiologist" is fine — that's just picking a provider type,
not asking for medical advice), scheduling questions, or general clinic questions
(hours, location, insurance).

Reply with exactly one word: YES if it should be flagged, NO if it's fine.

Caller's latest message: "{message}"
"""

OUT_OF_SCOPE_RESPONSE = (
    "I'm not able to give medical advice or discuss symptoms — for anything like that, "
    "please contact clinic staff directly, or if this is urgent, please call emergency "
    "services. I'm happy to help you book, reschedule, or cancel an appointment if you'd like."
)

# Bare confirmations/negations carry NO inherent medical content — a
# classifier judging "Yes" in isolation (no conversation context reaches
# it, by design) has essentially no real signal to work from, which is
# exactly why it intermittently misfired on plain booking confirmations
# in testing. This is a fixed, deliberately small, safe list of pure
# acknowledgment words that can never by themselves be medical content —
# NOT a length-based shortcut, since short medical phrases are real
# ("chest pain", "any painkillers?" are both short and must still be
# classified normally). Anything not an exact match here still goes
# through the real classifier, unchanged.
SAFE_BYPASS_PHRASES = {
    "yes", "yeah", "yep", "yup", "correct", "that's correct", "that's right",
    "sounds good", "sounds right", "ok", "okay", "sure", "go ahead",
    "no", "nope", "nah", "not correct", "that's wrong", "incorrect",
}


def _is_safe_bypass(message: str) -> bool:
    normalized = message.strip().lower().rstrip(".!?")
    return normalized in SAFE_BYPASS_PHRASES


async def classify_message(message: str) -> bool:
    """Returns True if the message should be flagged as out-of-scope medical content."""
    if _is_safe_bypass(message):
        return False  # deterministic — never sends these to the LLM classifier at all

    client = _get_client()
    response = await client.chat.completions.create(
        model="gpt-5.4-nano",  # cheap, fast — this is a narrow yes/no classification, not conversation
        messages=[{"role": "user", "content": SAFETY_CLASSIFIER_PROMPT.format(message=message)}],
        max_completion_tokens=5,
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")