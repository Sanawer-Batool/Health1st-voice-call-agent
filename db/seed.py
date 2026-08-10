"""
Builds db/clinic.db from schema.sql and inserts fake providers/appointment
types/appointments so there's real data to test tool functions against.

Run from the repo root:
    python db/seed.py
"""
import sqlite3
import os

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "clinic.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")


def build_db():
    # Start fresh every time this is run, so seeding is repeatable during dev
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    with open(SCHEMA_PATH, "r") as f:
        cur.executescript(f.read())

    # --- providers ---
    providers = [
        ("Dr. Ahmed Raza", "General Physician", "09:00", "17:00"),
        ("Dr. Sana Malik", "Pediatrician", "10:00", "16:00"),
        ("Dr. Bilal Khan", "Cardiologist", "09:00", "14:00"),
    ]
    cur.executemany(
        "INSERT INTO providers (name, specialty, working_hours_start, working_hours_end) VALUES (?, ?, ?, ?)",
        providers,
    )

    # --- appointment types ---
    appt_types = [
        ("New patient consult", 45),
        ("Follow-up", 15),
        ("Routine checkup", 30),
    ]
    cur.executemany(
        "INSERT INTO appointment_types (name, duration_minutes) VALUES (?, ?)",
        appt_types,
    )

    # --- provider_appointment_types (which provider offers which type) ---
    # provider ids 1,2,3 ; appointment_type ids 1,2,3 (insertion order above)
    provider_type_links = [
        (1, 1), (1, 2), (1, 3),   # Dr. Ahmed Raza offers all three
        (2, 1), (2, 2),           # Dr. Sana Malik: new patient + follow-up only
        (3, 2), (3, 3),           # Dr. Bilal Khan: follow-up + routine checkup only
    ]
    cur.executemany(
        "INSERT INTO provider_appointment_types (provider_id, appointment_type_id) VALUES (?, ?)",
        provider_type_links,
    )

    # --- a couple of existing appointments, so availability logic has something to work around ---
    existing_appointments = [
        (1, 2, "Fatima Noor", "0300-1234567", "2026-08-15 10:00", "booked"),
        (2, 1, "Zainab Ali", "0301-7654321", "2026-08-16 11:00", "booked"),
    ]
    cur.executemany(
        """INSERT INTO appointments
           (provider_id, appointment_type_id, patient_name, patient_contact, appointment_datetime, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        existing_appointments,
    )

    conn.commit()
    conn.close()
    print(f"Built {DB_PATH} with seed data.")


if __name__ == "__main__":
    build_db()
