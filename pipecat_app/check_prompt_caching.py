"""
Standalone diagnostic - does NOT touch the live pipeline. Sends 4
sequential requests simulating a real booking conversation, using the
actual system prompt and tool schemas, and prints OpenAI's own
cached_tokens field from each response.

Run:
    python check_prompt_caching.py

Requires OPENAI_API_KEY in your environment or .env file.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-5.4-nano"

SYSTEM_PROMPT = """You are a phone receptionist for Health1st Clinic.

You do NOT give medical advice, diagnoses, or symptom guidance under any
circumstance - if a caller describes symptoms or asks a medical question,
tell them you can't advise on that and offer to connect them to clinic
staff, then continue with booking if they still want that. (Note: a
dedicated safety check already runs before messages reach you - this
instruction is a backup layer, not the primary defense.)

For factual/policy questions about the clinic (insurance, hours, location,
referrals, what to bring, fasting, cancellation policy, walk-ins, telehealth,
prescriptions, etc.) - always call search_clinic_faq first rather than
answering from your own general knowledge. Base your answer only on what
the tool returns. If it doesn't return anything relevant, say you don't
have that info and offer to connect them to clinic staff - never guess.

If a caller states two conflicting values for the same detail close
together in one turn treat the LATER value as the correct one and proceed.

You help callers book, reschedule, or cancel appointments. Ask for missing
details one or two at a time - never list out three or four questions at
once. To book, you need: patient name, provider or specialty, appointment
type, and a requested date/time. Callers won't use exact clinic terms - if
unsure of exact provider/appointment-type names, call list_providers and/or
list_appointment_types first rather than guessing. If the caller names an
appointment type before a provider, call list_providers_for_appointment_type
first and only offer providers who actually offer that type.

Use check_availability before booking to confirm the slot is open. For
reschedule/cancel, use find_patient_appointments to find their existing
appointment(s) - if there's more than one, ask which one before acting.

Always restate the details and get explicit confirmation before calling
book_appointment, reschedule_appointment, or cancel_appointment.

Keep responses to 1-2 short sentences - this is a phone call, not a chat window.

# --- TEST PADDING ONLY, remove before shipping ---
""" + ("Reference filler line for cache-threshold testing only. " * 250)

TOOLS = [
    {"type": "function", "function": {
        "name": "check_availability",
        "description": "Check available appointment slots for a provider, appointment type, and date.",
        "parameters": {"type": "object", "properties": {
            "provider_name": {"type": "string", "description": "Provider's full name, e.g. 'Dr. Ahmed Raza'"},
            "appointment_type": {"type": "string", "description": "e.g. 'Routine checkup', 'Follow-up', 'New patient consult'"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
        }, "required": ["provider_name", "appointment_type", "date"]},
    }},
    {"type": "function", "function": {
        "name": "book_appointment",
        "description": "Book a new appointment. Only call after confirming details with the caller.",
        "parameters": {"type": "object", "properties": {
            "provider_name": {"type": "string"}, "appointment_type": {"type": "string"},
            "patient_name": {"type": "string"}, "patient_contact": {"type": "string"},
            "requested_datetime": {"type": "string", "description": "e.g. '2026-08-16 14:00'"},
        }, "required": ["provider_name", "appointment_type", "patient_name", "patient_contact", "requested_datetime"]},
    }},
    {"type": "function", "function": {
        "name": "list_providers", "description": "List all clinic providers and their specialties.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "list_appointment_types",
        "description": "List appointment types, optionally filtered to what a specific provider offers.",
        "parameters": {"type": "object", "properties": {
            "provider_name": {"type": "string", "description": "Optional - omit to list all types"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "list_providers_for_appointment_type",
        "description": "List only the providers who actually offer a given appointment type. Call this BEFORE offering provider choices once you know the appointment type.",
        "parameters": {"type": "object", "properties": {"appointment_type": {"type": "string"}}, "required": ["appointment_type"]},
    }},
    {"type": "function", "function": {
        "name": "find_patient_appointments",
        "description": "Find a caller's existing appointments by name, for reschedule/cancel flows.",
        "parameters": {"type": "object", "properties": {"patient_name": {"type": "string"}}, "required": ["patient_name"]},
    }},
    {"type": "function", "function": {
        "name": "reschedule_appointment",
        "description": "Reschedule an existing appointment to a new date/time. Confirm details first.",
        "parameters": {"type": "object", "properties": {
            "appointment_id": {"type": "integer"}, "new_datetime": {"type": "string"},
        }, "required": ["appointment_id", "new_datetime"]},
    }},
    {"type": "function", "function": {
        "name": "cancel_appointment", "description": "Cancel an existing appointment. Confirm which one first.",
        "parameters": {"type": "object", "properties": {"appointment_id": {"type": "integer"}}, "required": ["appointment_id"]},
    }},
    {"type": "function", "function": {
        "name": "search_clinic_faq",
        "description": "Search clinic FAQ/policy knowledge base for factual questions - insurance, hours, location, referrals, what to bring, fasting, cancellation policy, walk-ins, telehealth, prescriptions, etc. Always call this for such questions rather than answering from general knowledge.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The caller's question, in their own words"},
        }, "required": ["query"]},
    }},
]

conversation_turns = [
    "I want to book an appointment.",
    "A routine checkup with Dr. Ahmed Raza.",
    "Tomorrow at 2pm, my name is Test Caller.",
    "Yes that's correct.",
]


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, user_msg in enumerate(conversation_turns, start=1):
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_completion_tokens=150,
        )

        usage = response.usage
        cached = 0
        if usage.prompt_tokens_details is not None:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        total_prompt = usage.prompt_tokens

        pct = f"{100*cached/total_prompt:.0f}%" if total_prompt else "n/a"
        print(f"Turn {i}: prompt_tokens={total_prompt}, cached_tokens={cached}, cache_hit_rate={pct}")

        assistant_msg = response.choices[0].message
        messages.append({"role": "assistant", "content": assistant_msg.content or ""})


if __name__ == "__main__":
    main()
