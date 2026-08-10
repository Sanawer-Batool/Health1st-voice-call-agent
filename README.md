# Clinic Voice Agent

AI phone agent for a healthcare clinic — booking, rescheduling, cancellation via voice.
Stack: Twilio + Pipecat (voice) + LangGraph (reasoning) + SQLite (booking DB) + RAG (clinic FAQ).

Build order: text-first LangGraph → Twilio/Pipecat plumbing (isolated) → integration → deploy.

See `docs/` for schema design, repo structure, and full architecture notes.
