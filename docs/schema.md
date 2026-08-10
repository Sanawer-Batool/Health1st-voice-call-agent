# Clinic Booking Database — Schema Design

Simplifying assumption for now: every provider has the **same working hours every day** (no per-day variation, no day-off tracking yet). This can be split into a separate `provider_availability` table later if needed — noted as a future extension, not built now.

## Tables

### `providers`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | e.g. "Dr. Ahmed Raza" |
| specialty | TEXT | e.g. "General Physician", "Pediatrician" |
| working_hours_start | TEXT | "09:00" (24hr, stored as text for simplicity) |
| working_hours_end | TEXT | "17:00" |

### `appointment_types`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | e.g. "New patient consult", "Follow-up" |
| duration_minutes | INTEGER | e.g. 45, 15 |

### `provider_appointment_types`
Join table — not every provider offers every appointment type.

| Column | Type | Notes |
|---|---|---|
| provider_id | INTEGER FK → providers.id | |
| appointment_type_id | INTEGER FK → appointment_types.id | |

Composite primary key: (provider_id, appointment_type_id)

### `appointments`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| provider_id | INTEGER FK → providers.id | |
| appointment_type_id | INTEGER FK → appointment_types.id | |
| patient_name | TEXT | |
| patient_contact | TEXT | phone number |
| appointment_datetime | TEXT | ISO format "2026-08-15 10:00" |
| status | TEXT | 'booked' / 'cancelled' / 'rescheduled' |
| created_at | TEXT | timestamp, defaults to now |

## Design notes

- **Availability is computed, not stored.** There's no separate `slots` table listing every open slot — availability for a given provider/date is calculated on the fly: (provider's working hours) minus (existing booked appointments + their durations). This avoids having to pre-generate and maintain a slots table, and is simpler to keep in sync. If performance becomes an issue at scale, this can change later — not a concern at this project's scale.
- **Soft delete via `status`, not row deletion.** Cancelling an appointment sets `status = 'cancelled'` rather than deleting the row — keeps history for debugging and demo purposes (e.g. showing "3 cancellations, 12 successful bookings" as part of your task-success-rate metric later).
- **Rescheduling** = update `appointment_datetime` on the existing row and set `status = 'rescheduled'` (or keep `status = 'booked'` and just log the change — pick one convention and stay consistent; recommended: keep status `booked`, since it's still an active appointment, just moved).
