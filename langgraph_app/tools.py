"""
Tool functions the LangGraph LLM node will call. Each is plain Python
querying db/clinic.db directly — no ORM, kept simple on purpose.

These are tested standalone (see __main__ block / test run below) BEFORE
ever being wired into the graph or bound to an LLM — isolate the bug
surface, per the build plan.
"""
import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "clinic.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_providers() -> dict:
    """Returns all providers with their specialty, so the caller's request can be matched correctly."""
    conn = _connect()
    rows = conn.execute("SELECT name, specialty FROM providers").fetchall()
    conn.close()
    return {"providers": [dict(r) for r in rows]}


def list_appointment_types(provider_name: str = None) -> dict:
    """
    Returns valid appointment types. If provider_name is given, only returns
    types that provider actually offers (uses case-insensitive match on provider_name).
    """
    conn = _connect()
    if provider_name:
        rows = conn.execute(
            """SELECT at.name, at.duration_minutes FROM appointment_types at
               JOIN provider_appointment_types pat ON at.id = pat.appointment_type_id
               JOIN providers p ON pat.provider_id = p.id
               WHERE LOWER(p.name) = LOWER(?)""",
            (provider_name,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT name, duration_minutes FROM appointment_types").fetchall()
    conn.close()
    return {"appointment_types": [dict(r) for r in rows]}


def check_availability(provider_name: str, appointment_type: str, date: str) -> dict:
    """
    Returns available time slots for a given provider + appointment type + date.
    date format: "2026-08-16"
    Matching on provider_name and appointment_type is case-insensitive.
    """
    conn = _connect()
    cur = conn.cursor()

    provider = cur.execute(
        "SELECT * FROM providers WHERE LOWER(name) = LOWER(?)", (provider_name,)
    ).fetchone()
    if not provider:
        conn.close()
        return {"error": f"No provider found named '{provider_name}'. Call list_providers to see valid names."}

    appt_type = cur.execute(
        "SELECT * FROM appointment_types WHERE LOWER(name) = LOWER(?)", (appointment_type,)
    ).fetchone()
    if not appt_type:
        conn.close()
        return {"error": f"No appointment type found named '{appointment_type}'. Call list_appointment_types to see valid options."}

    # confirm this provider actually offers this appointment type
    offers = cur.execute(
        "SELECT 1 FROM provider_appointment_types WHERE provider_id = ? AND appointment_type_id = ?",
        (provider["id"], appt_type["id"]),
    ).fetchone()
    if not offers:
        conn.close()
        return {"error": f"{provider_name} does not offer '{appointment_type}' appointments."}

    duration = appt_type["duration_minutes"]

    # existing booked appointments for this provider on this date
    booked = cur.execute(
        """SELECT appointment_datetime, appointment_type_id FROM appointments
           WHERE provider_id = ? AND status = 'booked'
           AND appointment_datetime LIKE ?""",
        (provider["id"], f"{date}%"),
    ).fetchall()
    conn.close()

    booked_ranges = []
    for row in booked:
        start = datetime.strptime(row["appointment_datetime"], "%Y-%m-%d %H:%M")
        # look up that appointment's own duration to know its end time
        conn2 = _connect()
        t = conn2.execute(
            "SELECT duration_minutes FROM appointment_types WHERE id = ?",
            (row["appointment_type_id"],),
        ).fetchone()
        conn2.close()
        end = start + timedelta(minutes=t["duration_minutes"])
        booked_ranges.append((start, end))

    # generate candidate slots in 15-min increments across working hours
    day_start = datetime.strptime(f"{date} {provider['working_hours_start']}", "%Y-%m-%d %H:%M")
    day_end = datetime.strptime(f"{date} {provider['working_hours_end']}", "%Y-%m-%d %H:%M")

    available = []
    slot = day_start
    while slot + timedelta(minutes=duration) <= day_end:
        slot_end = slot + timedelta(minutes=duration)
        overlaps = any(slot < b_end and slot_end > b_start for b_start, b_end in booked_ranges)
        if not overlaps:
            available.append(slot.strftime("%H:%M"))
        slot += timedelta(minutes=15)

    return {"provider": provider_name, "date": date, "available_slots": available}


def book_appointment(provider_name: str, appointment_type: str, patient_name: str,
                      patient_contact: str, requested_datetime: str) -> dict:
    """
    requested_datetime format: "2026-08-16 11:00"
    """
    conn = _connect()
    cur = conn.cursor()

    provider = cur.execute("SELECT * FROM providers WHERE LOWER(name) = LOWER(?)", (provider_name,)).fetchone()
    appt_type = cur.execute("SELECT * FROM appointment_types WHERE LOWER(name) = LOWER(?)", (appointment_type,)).fetchone()

    if not provider or not appt_type:
        conn.close()
        return {"error": "Provider or appointment type not found."}

    cur.execute(
        """INSERT INTO appointments
           (provider_id, appointment_type_id, patient_name, patient_contact, appointment_datetime, status)
           VALUES (?, ?, ?, ?, ?, 'booked')""",
        (provider["id"], appt_type["id"], patient_name, patient_contact, requested_datetime),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return {
        "success": True,
        "appointment_id": new_id,
        "provider": provider_name,
        "appointment_type": appointment_type,
        "datetime": requested_datetime,
    }


def reschedule_appointment(appointment_id: int, new_datetime: str) -> dict:
    conn = _connect()
    cur = conn.cursor()

    existing = cur.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,)).fetchone()
    if not existing:
        conn.close()
        return {"error": f"No appointment found with id {appointment_id}."}
    if existing["status"] == "cancelled":
        conn.close()
        return {"error": "This appointment is already cancelled and can't be rescheduled."}

    cur.execute(
        "UPDATE appointments SET appointment_datetime = ? WHERE id = ?",
        (new_datetime, appointment_id),
    )
    conn.commit()
    conn.close()

    return {"success": True, "appointment_id": appointment_id, "new_datetime": new_datetime}


def cancel_appointment(appointment_id: int) -> dict:
    conn = _connect()
    cur = conn.cursor()

    existing = cur.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,)).fetchone()
    if not existing:
        conn.close()
        return {"error": f"No appointment found with id {appointment_id}."}

    cur.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,))
    conn.commit()
    conn.close()

    return {"success": True, "appointment_id": appointment_id, "status": "cancelled"}


def find_patient_appointments(patient_name: str) -> dict:
    """Helper for reschedule/cancel flows — find a caller's existing appointments by name (case-insensitive)."""
    conn = _connect()
    rows = conn.execute(
        """SELECT a.id, p.name as provider, at.name as appointment_type,
                  a.appointment_datetime, a.status
           FROM appointments a
           JOIN providers p ON a.provider_id = p.id
           JOIN appointment_types at ON a.appointment_type_id = at.id
           WHERE LOWER(a.patient_name) = LOWER(?) AND a.status != 'cancelled'""",
        (patient_name,),
    ).fetchall()
    conn.close()
    return {"appointments": [dict(r) for r in rows]}