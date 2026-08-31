"""
A small, self-contained dashboard endpoint for demo purposes: shows every
appointment in the DB as an HTML table, auto-refreshing every 5 seconds
via a meta-refresh tag (deliberately no JS needed — simplest thing that
reliably works live during a demo, no extra moving parts to fail).

Import and mount this into your FastAPI app:
    from dashboard import router as dashboard_router
    app.include_router(dashboard_router)

Then open http://localhost:8000/dashboard in a browser tab during your
demo — leave it open next to the call. New bookings/reschedules/
cancellations made during a live call appear within 5 seconds.
"""
import os
import sqlite3
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "clinic.db")


def _fetch_appointments():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT a.id, a.patient_name, a.patient_contact, p.name as provider,
                  at.name as appointment_type, a.appointment_datetime, a.status,
                  a.created_at
           FROM appointments a
           JOIN providers p ON a.provider_id = p.id
           JOIN appointment_types at ON a.appointment_type_id = at.id
           ORDER BY a.created_at DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    appointments = _fetch_appointments()

    booked = sum(1 for a in appointments if a["status"] == "booked")
    cancelled = sum(1 for a in appointments if a["status"] == "cancelled")
    total = len(appointments)

    status_colors = {"booked": "#2e7d32", "cancelled": "#c62828", "rescheduled": "#ef6c00"}

    rows_html = ""
    for a in appointments:
        color = status_colors.get(a["status"], "#555")
        rows_html += f"""<tr>
            <td>{a['id']}</td>
            <td>{a['patient_name']}</td>
            <td>{a['patient_contact'] or '—'}</td>
            <td>{a['provider']}</td>
            <td>{a['appointment_type']}</td>
            <td>{a['appointment_datetime']}</td>
            <td style="color:{color}; font-weight:600">{a['status']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="5">
    <title>Health1st Clinic — Live Appointments</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f7f7f8; }}
        h1 {{ margin-bottom: 4px; }}
        .summary {{ margin-bottom: 24px; color: #444; }}
        .summary span {{ font-weight: 700; margin-right: 24px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #eee; }}
        th {{ background: #fafafa; font-size: 13px; text-transform: uppercase; color: #666; }}
        tr:hover {{ background: #fafafa; }}
    </style>
</head>
<body>
    <h1>Health1st Clinic — Live Appointments</h1>
    <div class="summary">
        <span>Total: {total}</span>
        <span style="color:#2e7d32">Booked: {booked}</span>
        <span style="color:#c62828">Cancelled: {cancelled}</span>
    </div>
    <table>
        <tr><th>ID</th><th>Patient</th><th>Contact</th><th>Provider</th><th>Type</th><th>Date/Time</th><th>Status</th></tr>
        {rows_html if rows_html else '<tr><td colspan="7" style="text-align:center;color:#999;padding:24px">No appointments yet</td></tr>'}
    </table>
</body>
</html>"""
    return HTMLResponse(content=html)
