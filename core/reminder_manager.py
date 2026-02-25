"""
Reminder Manager — SQLite-Backed Event Reminders
===================================================
Checks stored events against the current time and generates
reminders. All state persists to SQLite via Repository.

Features:
  - Get upcoming events (within N minutes)
  - Check for due events right now
  - Auto-schedule reminders for new events
  - Background reminder loop (runs in a thread)
  - Persist: scheduled, triggered, snoozed, dismissed
  - Restart-safe — reloads pending reminders from DB

No heavy dependencies — uses threading for background loop.
"""

import threading
import time as time_module
from datetime import datetime, timedelta

from core.query_engine import MemoryStore
from core.date_parser import parse_date, parse_time, combine_datetime


class ReminderManager:
    """
    SQLite-backed reminder manager.

    Usage:
        reminder = ReminderManager(memory, repo)
        reminder.load_pending()               # Reload from DB on startup
        reminder.auto_schedule()              # Create reminders for new events
        reminder.start_reminder_loop(60)      # Check every 60 sec
        # ... later ...
        reminder.stop()
    """

    def __init__(self, memory: MemoryStore, repo=None):
        """
        Initialize the reminder manager.

        Args:
            memory: Object with get_all_events() and search_events() methods.
            repo:   Repository instance for SQLite persistence (optional).
                    If None, falls back to in-memory only (legacy mode).
        """
        self.memory = memory
        self.repo = repo
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # In-memory cache of fired alert keys (prevents duplicate alerts per session)
        self._alerted: set[str] = set()

        # Load previously fired reminders into alerted set on startup
        if self.repo:
            self._load_fired_keys()

    # ── Persistence ──────────────────────────────────────────

    def _load_fired_keys(self) -> None:
        """Load fired/dismissed reminder keys into _alerted set (restart safety)."""
        if not self.repo:
            return

        try:
            fired = self.repo.get_reminders_by_status("fired")
            dismissed = self.repo.get_reminders_by_status("dismissed")

            for r in fired + dismissed:
                desc = r.get("description", "")
                dt = r.get("trigger_time", "")
                self._alerted.add(f"{desc}_{dt}")
        except Exception:
            pass  # DB may not have reminders table yet

    def load_pending(self) -> list[dict]:
        """
        Load all pending reminders from SQLite.
        Call this on startup to restore state.

        Returns:
            List of pending reminder dicts.
        """
        if not self.repo:
            return []

        return self.repo.get_pending_reminders()

    def auto_schedule(self, lead_minutes: int = 15) -> int:
        """
        Auto-create reminders for events that don't have one yet.
        Trigger time = event time minus `lead_minutes`.

        Returns:
            Count of new reminders created.
        """
        if not self.repo:
            return 0

        return self.repo.auto_schedule_reminders(lead_minutes=lead_minutes)

    # ── Query methods ────────────────────────────────────────

    def get_upcoming_events(self, minutes: int = 60) -> list[dict]:
        """
        Get events due within the next `minutes` minutes.

        Returns:
            List of events with computed event_datetime and minutes_until.
        """
        now = datetime.now()
        window_end = now + timedelta(minutes=minutes)
        upcoming = []

        for event in self.memory.get_all_events():
            event_dt = self._get_event_datetime(event)
            if event_dt is None:
                continue

            if now <= event_dt <= window_end:
                mins_until = int((event_dt - now).total_seconds() / 60)
                upcoming.append({
                    **event,
                    "event_datetime": event_dt.isoformat(),
                    "minutes_until": mins_until,
                })

        # Sort by nearest first
        upcoming.sort(key=lambda e: e["minutes_until"])
        return upcoming

    def check_due_events(self, window_minutes: int = 5) -> list[str]:
        """
        Check for events due within the next `window_minutes` minutes.
        Fires reminders and persists status to SQLite.

        Returns:
            List of formatted alert strings.
        """
        upcoming = self.get_upcoming_events(minutes=window_minutes)
        alerts = []

        for event in upcoming:
            desc = event.get("description", "Unknown event")
            event_dt = event.get("event_datetime", "")
            alert_key = f"{desc}_{event_dt}"

            with self._lock:
                # Skip if already alerted this session
                if alert_key in self._alerted:
                    continue
                self._alerted.add(alert_key)

            mins = event["minutes_until"]
            time_str = event.get("parsed_time") or event.get("time") or ""

            if mins <= 0:
                alert = f"⏰ REMINDER: {desc}"
                if time_str:
                    alert += f" at {time_str}"
                alert += " (NOW!)"
            elif mins == 1:
                alert = f"⏰ REMINDER: {desc}"
                if time_str:
                    alert += f" at {time_str}"
                alert += " (in 1 minute)"
            else:
                alert = f"⏰ REMINDER: {desc}"
                if time_str:
                    alert += f" at {time_str}"
                alert += f" (in {mins} minutes)"

            alerts.append(alert)

            # Persist fired status to SQLite
            if self.repo:
                self._mark_event_reminder_fired(event)

        return alerts

    def _mark_event_reminder_fired(self, event: dict) -> None:
        """Find and mark the pending reminder for this event as fired."""
        try:
            event_id = event.get("id")
            if not event_id:
                return

            pending = self.repo.get_pending_reminders()
            for r in pending:
                if r.get("event_id") == event_id:
                    self.repo.mark_reminder_fired(r["id"])
                    break
        except Exception:
            pass  # Don't crash the alert system

    def dismiss(self, reminder_id: str) -> bool:
        """
        Dismiss a reminder (user explicitly silenced it).

        Returns True if successful.
        """
        if not self.repo:
            return False

        try:
            self.repo.dismiss_reminder(reminder_id)
            return True
        except Exception:
            return False

    def snooze(self, reminder_id: str, snooze_minutes: int = 10) -> bool:
        """
        Snooze a reminder by N minutes.

        Returns True if successful.
        """
        if not self.repo:
            return False

        try:
            new_time = (datetime.now() + timedelta(minutes=snooze_minutes)).isoformat()
            self.repo.snooze_reminder(reminder_id, new_time)

            # Remove from alerted set so it can fire again
            with self._lock:
                keys_to_remove = [k for k in self._alerted if reminder_id in k]
                for k in keys_to_remove:
                    self._alerted.discard(k)

            return True
        except Exception:
            return False

    def get_todays_schedule(self) -> list[dict]:
        """
        Get all events scheduled for today.

        Returns:
            List of events with parsed datetimes for today.
        """
        today = datetime.now().date()
        schedule = []

        for event in self.memory.get_all_events():
            event_dt = self._get_event_datetime(event)
            if event_dt and event_dt.date() == today:
                schedule.append({
                    **event,
                    "event_datetime": event_dt.isoformat(),
                })

        schedule.sort(key=lambda e: e["event_datetime"])
        return schedule

    def format_schedule(self, events: list[dict]) -> str:
        """Format a list of events into a readable schedule."""
        if not events:
            return "No events scheduled."

        lines = []
        for e in events:
            time_str = e.get("parsed_time") or e.get("time") or "TBD"
            desc = e.get("description", "Unknown")
            event_type = e.get("type", "event").upper()
            lines.append(f"  {time_str:>8s}  [{event_type}] {desc}")

        return "\n".join(lines)

    def get_status(self) -> dict:
        """Get reminder system status."""
        result = {
            "running": self._running,
            "alerted_this_session": len(self._alerted),
        }

        if self.repo:
            try:
                result["pending"] = len(self.repo.get_pending_reminders())
                result["fired"] = len(self.repo.get_reminders_by_status("fired"))
                result["dismissed"] = len(self.repo.get_reminders_by_status("dismissed"))
                result["snoozed"] = len(self.repo.get_reminders_by_status("snoozed"))
                result["persisted"] = True
            except Exception:
                result["persisted"] = False
        else:
            result["persisted"] = False

        return result

    # ── Reminder loop ────────────────────────────────────────

    def start_reminder_loop(self, interval: int = 60):
        """
        Start a background thread that checks for due events.

        Args:
            interval: How often to check, in seconds.
        """
        if self._running:
            return

        self._running = True

        # Auto-schedule reminders on start
        if self.repo:
            scheduled = self.auto_schedule()
            if scheduled:
                print(f"[Reminders] Auto-scheduled {scheduled} new reminder(s)")

        def _loop():
            while self._running:
                try:
                    alerts = self.check_due_events(window_minutes=15)
                    for alert in alerts:
                        print(f"\n  {alert}")

                    # Periodically auto-schedule new events
                    if self.repo:
                        self.auto_schedule()
                except Exception as e:
                    print(f"[Reminders] Error in loop: {e}")

                time_module.sleep(interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the reminder loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    # ── Internal helpers ─────────────────────────────────────

    def _get_event_datetime(self, event: dict) -> datetime | None:
        """
        Try to build a datetime from event's parsed_date / parsed_time,
        falling back to raw date/time fields.
        """
        # Try parsed fields first
        date_str = event.get("parsed_date")
        time_str = event.get("parsed_time")

        # Fall back to raw fields and parse them
        if not date_str:
            raw_date = event.get("raw_date") or event.get("date")
            if raw_date:
                date_str = parse_date(raw_date)

        if not time_str:
            raw_time = event.get("time")
            if raw_time:
                time_str = parse_time(raw_time)

        return combine_datetime(date_str, time_str)


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    from core.date_parser import parse_date as pd

    class _TestStore:
        def __init__(self, events):
            self._events = events
        def get_all_events(self):
            return self._events
        def search_events(self, keyword):
            kw = keyword.lower()
            return [e for e in self._events
                    if kw in (e.get('description','') + e.get('type','')).lower()]

    tomorrow = pd("tomorrow")

    store = _TestStore([
        {"type": "meeting", "raw_date": "tomorrow", "parsed_date": tomorrow,
         "time": "10 AM", "parsed_time": "10:00",
         "description": "Doctor appointment"},
        {"type": "task", "raw_date": "tomorrow", "parsed_date": tomorrow,
         "time": None, "parsed_time": None,
         "description": "Buy groceries"},
        {"type": "medication", "raw_date": None, "parsed_date": None,
         "time": None, "parsed_time": None,
         "description": "Take medicine after breakfast"},
    ])

    reminder = ReminderManager(store)  # No repo = legacy mode

    print("--- Today's Schedule ---")
    schedule = reminder.get_todays_schedule()
    print(reminder.format_schedule(schedule) or "  Nothing today.")

    print("\n--- Upcoming (next 24h) ---")
    upcoming = reminder.get_upcoming_events(minutes=24 * 60)
    for e in upcoming:
        print(f"  {e['description']} (in {e['minutes_until']} min)")

    print("\n--- Status ---")
    print(f"  {reminder.get_status()}")
