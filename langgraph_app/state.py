"""
Conversation state schema — the object that flows through every LangGraph node.
"""
from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages


class ConversationState(TypedDict):
    # Conversation history — LangGraph convention, uses add_messages reducer
    # so new messages append rather than overwrite on each node run.
    messages: Annotated[list, add_messages]

    # What the caller is trying to do. Starts as None until intent
    # classification runs; 'unclear' is a valid value if the agent
    # genuinely can't tell yet and needs to ask.
    intent: Optional[str]  # 'book' | 'reschedule' | 'cancel' | 'faq' | 'out_of_scope' | 'unclear'

    # --- slot-filling fields (booking/reschedule/cancel share most of these) ---
    patient_name: Optional[str]
    provider: Optional[str]            # provider name as given/matched, e.g. "Dr. Ahmed Raza"
    appointment_type: Optional[str]    # e.g. "Follow-up"
    requested_datetime: Optional[str]  # ISO-ish string, e.g. "2026-08-16 11:00"
    contact_info: Optional[str]

    # For reschedule/cancel — which existing appointment are we acting on
    target_appointment_id: Optional[int]

    # Which required slots are still missing — used to decide what to ask next
    missing_slots: List[str]

    # Gate before any DB write — nothing gets booked/cancelled/rescheduled
    # until this is True and the caller has confirmed the restated details
    awaiting_confirmation: bool

    # Set True when the safety/scope guardrail fires (medical content) —
    # once True, routing goes straight to the fixed redirect response
    out_of_scope_flag: bool
