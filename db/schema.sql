CREATE TABLE providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL,
    working_hours_start TEXT NOT NULL,   -- "09:00"
    working_hours_end TEXT NOT NULL      -- "17:00"
);

CREATE TABLE appointment_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL
);

CREATE TABLE provider_appointment_types (
    provider_id INTEGER NOT NULL,
    appointment_type_id INTEGER NOT NULL,
    PRIMARY KEY (provider_id, appointment_type_id),
    FOREIGN KEY (provider_id) REFERENCES providers(id),
    FOREIGN KEY (appointment_type_id) REFERENCES appointment_types(id)
);

CREATE TABLE appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL,
    appointment_type_id INTEGER NOT NULL,
    patient_name TEXT NOT NULL,
    patient_contact TEXT,
    appointment_datetime TEXT NOTpip install langgraph langchain-openai  # or whichever LLM provider you're using NULL,   -- "2026-08-15 10:00"
    status TEXT NOT NULL DEFAULT 'booked', -- booked / cancelled / rescheduled
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (provider_id) REFERENCES providers(id),
    FOREIGN KEY (appointment_type_id) REFERENCES appointment_types(id)
);
