"""
Skeleton graph — booking happy-path only for now (Step 4 of the build plan).
Reschedule, cancel, safety guardrail, and RAG/FAQ are added on top of this
in later steps, one at a time.

Run as a terminal chatbot via run_text.py — no voice, no Pipecat yet.
"""
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

from state import ConversationState
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
from retrieval import search_clinic_faq

# --- LLM setup ---
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

# Separate, narrow-purpose LLM call for the safety classifier — deliberately
# NOT the same call as the main conversational LLM, and deliberately not
# given any tools. Its only job is one yes/no judgment.
safety_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

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
    "please contact clinic staff directly, or if this is urgent, please call emergency services. "
    "I'm happy to help you book, reschedule, or cancel an appointment if you'd like."
)

tools = [
    check_availability,
    book_appointment,
    list_providers,
    list_appointment_types,
    list_providers_for_appointment_type,
    find_patient_appointments,
    reschedule_appointment,
    cancel_appointment,
    search_clinic_faq,
]
llm_with_tools = llm.bind_tools(tools)
tools_by_name = {t.__name__: t for t in tools}

SYSTEM_PROMPT = """You are a phone receptionist for Health1st Clinic.

You help callers book appointments. You do NOT give medical advice, diagnoses,
or symptom guidance under any circumstance — if a caller describes symptoms or
asks a medical question, tell them you can't advise on that and offer to connect
them to clinic staff, then continue with booking if they still want that.

The caller's speech was transcribed automatically by a speech-to-text
system and may contain transcription errors — misheard words, wrong
homophone, or garbled phrases. If something doesn't make sense in context,
infer the most likely intended clinic-related term (appointment types,
provider names, common phrases) before responding, rather than taking the
transcript completely literally or asking the caller to repeat themselves
unnecessarily. If the ambiguity is genuinely unresolvable — e.g. it could
change which provider or appointment type gets booked — it's safer to ask
a quick clarifying question than to guess wrong on something that affects
the actual booking.

Callers won't use exact clinic terminology (e.g. they might say "usual checkup"
instead of "Routine checkup", or just describe a specialty instead of a doctor's
name). Never guess a provider or appointment type string and pass it directly to
a tool. If you're not certain of the exact valid name, call list_providers and/or
list_appointment_types first, then match the caller's description to the closest
real option yourself — don't ask the caller to repeat themselves "exactly."

To book an appointment, you need: patient name, which provider or specialty,
appointment type, and a requested date/time. Ask for missing details one or two
at a time, don't interrogate the caller with a wall of questions at once.

If the caller specifies an appointment type before naming a provider, call
list_providers_for_appointment_type to see which providers actually offer that
type, and only offer those as options — never list a provider who doesn't
actually provide the requested appointment type, even if you know their name
from elsewhere in the conversation.

When a caller gives a time without specifying AM/PM (e.g. "at 11" or "at 2"),
resolve it using the clinic's working hours rather than guessing arbitrarily —
if only one of AM/PM falls within the relevant provider's working hours, use
that interpretation. If both AM and PM would be plausible within working hours,
or the time is genuinely ambiguous, ask the caller to clarify rather than
silently assuming — getting this wrong means a caller misses their appointment.

Use check_availability before booking to confirm the slot is actually open.
If the exact requested time isn't available, don't just describe the overall
working-hours range — look at the specific available_slots list the tool
returns and offer 2-3 concrete times closest to what the caller asked for
(e.g. if they wanted 11:00 and it's taken, offer "10:30 or 11:30" if those
are open, not "we're open 9 to 4"). A caller can't act on a vague range over
the phone; they need specific times to choose from.
Before calling book_appointment, restate the details back to the caller and
get explicit confirmation ("does that sound right?") before booking.

For rescheduling or cancelling: callers usually won't know their appointment ID.
Use find_patient_appointments with their name to look up their existing
appointments. If they have exactly one, confirm it's the right one before acting.
If they have more than one, read back the options (provider, type, date/time) and
ask which one they mean — never guess or act on the wrong one. If none are found,
say so and offer to connect them to clinic staff.

Always restate the specific change (old time -> new time for reschedule; which
appointment for cancel) and get explicit confirmation before calling
reschedule_appointment or cancel_appointment. Never cancel or reschedule
without that confirmation, the same way you wouldn't book without it.

For factual/policy questions about the clinic (insurance, hours, location,
referrals, what to bring, fasting, cancellation policy, walk-ins, telehealth,
prescriptions, etc.) — always call search_clinic_faq first rather than
answering from your own general knowledge. Base your answer only on what the
tool returns. If the tool doesn't return anything relevant to the question,
say you don't have that information and offer to connect them to clinic staff
— never guess or make up a plausible-sounding policy.

Keep responses short and natural — this is a phone conversation, not a chat window.
"""


# --- Nodes ---
def safety_check(state: ConversationState):
    """
    Runs on every turn, before the main LLM node. Classifies only the
    caller's latest message. Sets out_of_scope_flag accordingly.
    """
    last_human = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg
            break

    if last_human is None:
        return {"out_of_scope_flag": False}

    prompt = SAFETY_CLASSIFIER_PROMPT.format(message=last_human.content)
    result = safety_llm.invoke([HumanMessage(content=prompt)])
    flagged = result.content.strip().upper().startswith("YES")

    return {"out_of_scope_flag": flagged}


def out_of_scope_response(state: ConversationState):
    """Fixed, non-improvised redirect — not generated fresh by the LLM."""
    return {
        "messages": [AIMessage(content=OUT_OF_SCOPE_RESPONSE)],
        "out_of_scope_flag": False,  # reset for the next turn
    }


def route_after_safety_check(state: ConversationState):
    if state.get("out_of_scope_flag"):
        return "out_of_scope"
    return "llm"


def call_llm(state: ConversationState):
    messages = state["messages"]
    # Ensure system prompt is present as the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def call_tools(state: ConversationState):
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        result = tool_fn(**tool_call["args"])  # plain functions, called directly
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}


def should_continue(state: ConversationState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# --- Graph wiring ---
def build_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("safety_check", safety_check)
    graph.add_node("out_of_scope", out_of_scope_response)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", call_tools)

    graph.set_entry_point("safety_check")
    graph.add_conditional_edges(
        "safety_check", route_after_safety_check, {"out_of_scope": "out_of_scope", "llm": "llm"}
    )
    graph.add_edge("out_of_scope", END)
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()


graph = build_graph()