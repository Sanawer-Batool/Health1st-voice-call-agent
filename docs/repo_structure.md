# Repo structure (rough — will evolve as we build)

```
clinic-voice-agent/
├── db/
│   ├── clinic.db              # the actual SQLite file
│   ├── schema.sql              # CREATE TABLE statements (source of truth)
│   └── seed.py                  # populates fake providers/types/appointments for testing
│
├── langgraph_app/
│   ├── state.py                 # conversation state schema (TypedDict/Pydantic)
│   ├── tools.py                  # check_availability, book_appointment, reschedule, cancel
│   ├── graph.py                   # the LangGraph graph definition (nodes, edges)
│   └── run_text.py                 # terminal entry point — text-first testing (Step 1 focus)
│
├── rag/
│   ├── faq_data.md                  # raw clinic FAQ content, written by us
│   ├── build_index.py                 # chunk + embed faq_data.md into a vector store
│   └── vectorstore/                    # persisted Chroma index (added later)
│
├── pipecat_app/                          # not touched yet — Phase 3+
│   ├── pipeline.py
│   └── twilio_transport_config.py
│
├── tests/
│   └── test_conversations.md              # realistic test scripts (happy path, mid-flow change, etc.)
│
├── docs/
│   ├── schema.md
│   ├── repo_structure.md
│   └── architecture.md                      # the full architecture writeup we made earlier
│
├── requirements.txt
└── README.md
```

## Notes
- `pipecat_app/` exists as a placeholder folder now so the structure is visible in git history, but stays empty until Step 2 of implementation (Twilio/Pipecat plumbing) — we are not touching it yet.
- `db/schema.sql` is the single source of truth for table definitions; `seed.py` reads/executes it and then inserts fake data. Never hand-edit `clinic.db`'s schema directly — always go through `schema.sql` so it's versioned in git.
- This structure will very likely gain a `config.py` or `.env` once API keys (LLM provider, etc.) enter the picture — not needed yet for pure text-first SQLite work.
