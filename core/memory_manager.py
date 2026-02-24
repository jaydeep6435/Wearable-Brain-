"""
Memory Manager Module — Event Storage & Retrieval
===================================================
Stores extracted events in-memory and optionally persists to a JSON file.

Provides:
  - add_event(event)        : Store a single structured event
  - add_events(events)      : Store a list of events
  - get_all_events()        : Retrieve all stored events
  - get_today_events()      : Filter events relevant to today/tomorrow
  - search_events(keyword)  : Search across all fields
  - save_to_file(path)      : Persist to JSON
  - load_from_file(path)    : Load from JSON
"""

import json
import os
from datetime import datetime


class MemoryManager:
    """
    Simple in-memory event store with optional JSON file persistence.

    Usage:
        memory = MemoryManager()
        memory.add_event({"type": "meeting", "date": "tomorrow", ...})
        results = memory.search_events("doctor")
        memory.save_to_file("memory.json")
    """

    def __init__(self):
        """Initialize with an empty event list."""
        self._events: list[dict] = []

    # -- Add events --------------------------------------------------------

    def add_event(self, event: dict) -> None:
        """
        Add a single event to memory.

        Args:
            event: dict with keys like type, date, time, person, description
        """
        # Add a timestamp for when this event was recorded
        event_copy = dict(event)
        event_copy["recorded_at"] = datetime.now().isoformat()
        self._events.append(event_copy)

    def add_events(self, events: list[dict]) -> None:
        """Add multiple events at once."""
        for event in events:
            self.add_event(event)

    # -- Retrieve events ---------------------------------------------------

    def get_all_events(self) -> list[dict]:
        """Return all stored events."""
        return list(self._events)

    def get_today_events(self) -> list[dict]:
        """
        Return events that reference today or tomorrow.
        Matches common date words: today, tonight, tomorrow, this morning, etc.
        """
        today_keywords = {"today", "tonight", "tomorrow", "this morning",
                          "this afternoon", "this evening"}

        results = []
        for event in self._events:
            date_val = (event.get("date") or "").lower()
            desc_val = (event.get("description") or "").lower()

            # Check if any today-keyword appears in the date or description
            for keyword in today_keywords:
                if keyword in date_val or keyword in desc_val:
                    results.append(event)
                    break

        return results

    def search_events(self, keyword: str) -> list[dict]:
        """
        Search events by keyword across all fields.

        Args:
            keyword: search term (case-insensitive)

        Returns:
            List of events where the keyword appears in any field value.
        """
        keyword_lower = keyword.lower()
        results = []

        for event in self._events:
            for value in event.values():
                if value and keyword_lower in str(value).lower():
                    results.append(event)
                    break  # Don't add same event twice

        return results

    # -- Persistence -------------------------------------------------------

    def save_to_file(self, path: str) -> None:
        """
        Save all events to a JSON file.

        Args:
            path: file path for the JSON output
        """
        # Create parent directories if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2, ensure_ascii=False)

        print(f"[Memory] Saved {len(self._events)} events to: {path}")

    def load_from_file(self, path: str) -> None:
        """
        Load events from a JSON file (appends to existing events).

        Args:
            path: file path to load from
        """
        if not os.path.isfile(path):
            print(f"[Memory] File not found: {path} — starting fresh.")
            return

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        self._events.extend(loaded)
        print(f"[Memory] Loaded {len(loaded)} events from: {path}")

    # -- Utility -----------------------------------------------------------

    def count(self) -> int:
        """Return the total number of stored events."""
        return len(self._events)

    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()

    def __repr__(self) -> str:
        return f"MemoryManager({len(self._events)} events)"


# -------------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    memory = MemoryManager()

    # Add some sample events
    memory.add_event({
        "type": "meeting",
        "date": "tomorrow",
        "time": "10 AM",
        "person": "Dr. Smith",
        "description": "Doctor appointment tomorrow at 10 AM",
    })
    memory.add_event({
        "type": "task",
        "date": "today",
        "time": None,
        "person": None,
        "description": "Call the pharmacy to refill the prescription",
    })
    memory.add_event({
        "type": "medication",
        "date": None,
        "time": None,
        "person": None,
        "description": "Take your medicine after breakfast",
    })

    print(f"Total events: {memory.count()}")
    print(f"\nToday's events: {memory.get_today_events()}")
    print(f"\nSearch 'doctor': {memory.search_events('doctor')}")

    # Save and reload
    memory.save_to_file("test_memory.json")
