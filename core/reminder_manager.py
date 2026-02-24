"""
Reminder Manager Module — Event Reminders & Alerts
====================================================
Checks stored events against the current time and generates
reminders for upcoming events.

Features:
  - Get upcoming events (within N minutes)
  - Check for due events right now
  - Background reminder loop (runs in a thread)
  - Console alerts with ⏰ emoji

No heavy dependencies — uses threading.Timer for scheduling.
"""

import threading
import time as time_module
from datetime import datetime, timedelta

from core.memory_manager import MemoryManager
from core.date_parser import parse_date, parse_time, combine_datetime


class ReminderManager:
    """
    Manages reminders for stored events.

    Usage:
        reminder = ReminderManager(memory)
        reminder.start_reminder_loop(interval=60)  # check every 60 sec
        # ... later ...
        reminder.stop()
    """

    def __init__(self, memory: MemoryManager):
        """
        Initialize with a MemoryManager instance.

        Args:
            memory: A MemoryManager with events already loaded.
        """
        self.memory = memory
        self._running = False
        self._thread = None
        self._alerted = set()  # Track already-alerted event descriptions

    # -- Query methods -----------------------------------------------------

    def get_upcoming_events(self, minutes: int = 60) -> list[dict]:
        """
        Get events that are due within the next `minutes` minutes.

        Args:
            minutes: Look-ahead window in minutes.

        Returns:
            List of events with their computed datetime and minutes_until.
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

        Returns:
            List of formatted alert strings.
        """
        upcoming = self.get_upcoming_events(minutes=window_minutes)
        alerts = []

        for event in upcoming:
            desc = event.get("description", "Unknown event")
            alert_key = f"{desc}_{event.get('event_datetime', '')}"

            # Skip if already alerted
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

        return alerts

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

    # -- Reminder loop -----------------------------------------------------

    def start_reminder_loop(self, interval: int = 60):
        """
        Start a background thread that checks for due events.

        Args:
            interval: How often to check, in seconds.
        """
        if self._running:
            return

        self._running = True

        def _loop():
            while self._running:
                alerts = self.check_due_events(window_minutes=15)
                for alert in alerts:
                    print(f"\n  {alert}")

                time_module.sleep(interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the reminder loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    # -- Internal helpers --------------------------------------------------

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


# -------------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from core.date_parser import parse_date as pd

    memory = MemoryManager()
    tomorrow = pd("tomorrow")

    memory.add_events([
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

    reminder = ReminderManager(memory)

    print("--- Today's Schedule ---")
    schedule = reminder.get_todays_schedule()
    print(reminder.format_schedule(schedule) or "  Nothing today.")

    print("\n--- Upcoming (next 24h) ---")
    upcoming = reminder.get_upcoming_events(minutes=24 * 60)
    for e in upcoming:
        print(f"  {e['description']} (in {e['minutes_until']} min)")

    print("\n--- Due Alerts ---")
    alerts = reminder.check_due_events(window_minutes=24 * 60)
    for a in alerts:
        print(f"  {a}")
