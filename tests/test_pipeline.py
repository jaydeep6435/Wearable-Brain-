"""
Tests for Day 4 MVP -- Pipeline with Reminders & Date Parsing
===============================================================
Tests cover:
  - Summarizer (original + highlighted)
  - Structured event extraction (JSON with parsed dates)
  - Memory Manager (add, get, search, persist)
  - Query Engine (intent detection, date extraction, responses)
  - Date Parser (date/time parsing, combining)
  - Reminder Manager (upcoming events, alerts, scheduling)

Run with:  python -m pytest tests/test_pipeline.py -v
"""

import json
import os
import pytest
from datetime import datetime, timedelta

from core.summarizer import summarize, _split_sentences, summarize_with_highlights
from core.event_extractor import extract_events, extract_structured_events
from core.memory_manager import MemoryManager
from core.query_engine import QueryEngine
from core.date_parser import parse_date, parse_time, combine_datetime
from core.reminder_manager import ReminderManager


# =========================================================================
# Shared test data
# =========================================================================

SAMPLE_TEXT = (
    "Good morning! I have a doctor appointment tomorrow at 10 AM. "
    "Don't forget to take your medicine after breakfast. "
    "We need to call the pharmacy to refill the prescription. "
    "Your son David is visiting this weekend. "
    "Remember to do your morning exercises before lunch. "
    "The meeting with Dr. Smith is on March 15 at 11 AM."
)


def _create_test_memory() -> MemoryManager:
    """Create a MemoryManager loaded with sample events (new schema)."""
    tomorrow = parse_date("tomorrow")
    memory = MemoryManager()
    memory.add_events([
        {"type": "meeting", "raw_date": "tomorrow", "parsed_date": tomorrow,
         "time": "10 AM", "parsed_time": "10:00",
         "person": "Dr. Smith", "description": "Doctor appointment"},
        {"type": "meeting", "raw_date": None, "parsed_date": None,
         "time": None, "parsed_time": None,
         "person": "David", "description": "Visiting this weekend"},
        {"type": "task", "raw_date": "tomorrow", "parsed_date": tomorrow,
         "time": None, "parsed_time": None,
         "person": None, "description": "Buy groceries tomorrow"},
        {"type": "task", "raw_date": None, "parsed_date": None,
         "time": None, "parsed_time": None,
         "person": None, "description": "Call the pharmacy to refill prescription"},
        {"type": "task", "raw_date": None, "parsed_date": None,
         "time": None, "parsed_time": None,
         "person": None, "description": "Remember to do morning exercises"},
        {"type": "medication", "raw_date": None, "parsed_date": None,
         "time": None, "parsed_time": None,
         "person": None, "description": "Take your medicine after breakfast"},
        {"type": "medication", "raw_date": None, "parsed_date": None,
         "time": None, "parsed_time": None,
         "person": None, "description": "Refill the prescription"},
    ])
    return memory


# =========================================================================
# Summarizer Tests
# =========================================================================

class TestSummarizer:
    def test_split_sentences_returns_list(self):
        assert len(_split_sentences(SAMPLE_TEXT)) > 0

    def test_summarize_returns_string(self):
        result = summarize(SAMPLE_TEXT, num_sentences=2)
        assert isinstance(result, str) and len(result) > 0

    def test_summarize_respects_num_sentences(self):
        for n in [1, 2, 3]:
            assert len(_split_sentences(summarize(SAMPLE_TEXT, num_sentences=n))) <= n

    def test_summarize_empty_text(self):
        assert isinstance(summarize("", num_sentences=3), str)

    def test_summarize_single_sentence(self):
        assert "only sentence" in summarize("This is the only sentence here.", num_sentences=3)


# =========================================================================
# Highlighted Summary Tests
# =========================================================================

class TestHighlightedSummary:
    def test_returns_list_of_dicts(self):
        result = summarize_with_highlights(SAMPLE_TEXT)
        assert all(isinstance(item, dict) for item in result)

    def test_each_item_has_required_keys(self):
        for item in summarize_with_highlights(SAMPLE_TEXT):
            assert {"sentence", "important", "tags"}.issubset(item.keys())

    def test_detects_meeting_tag(self):
        assert any("meeting" in i["tags"] for i in summarize_with_highlights(SAMPLE_TEXT))

    def test_detects_medication_tag(self):
        assert any("medication" in i["tags"] for i in summarize_with_highlights(SAMPLE_TEXT))

    def test_empty_text_returns_empty_list(self):
        assert summarize_with_highlights("") == []


# =========================================================================
# Structured Event Extractor Tests (Day 4 — with parsed dates)
# =========================================================================

class TestStructuredExtractor:
    def test_returns_list(self):
        assert isinstance(extract_structured_events(SAMPLE_TEXT), list)

    def test_events_have_required_fields(self):
        required = {"type", "raw_date", "parsed_date", "time", "parsed_time", "person", "description"}
        for event in extract_structured_events(SAMPLE_TEXT):
            assert required.issubset(event.keys()), f"Missing: {required - event.keys()}"

    def test_type_is_valid(self):
        for event in extract_structured_events(SAMPLE_TEXT):
            assert event["type"] in {"meeting", "task", "medication"}

    def test_detects_meetings(self):
        assert any(e["type"] == "meeting" for e in extract_structured_events(SAMPLE_TEXT))

    def test_detects_tasks(self):
        assert any(e["type"] == "task" for e in extract_structured_events(SAMPLE_TEXT))

    def test_detects_medication(self):
        assert any(e["type"] == "medication" for e in extract_structured_events(SAMPLE_TEXT))

    def test_detects_person(self):
        assert any(e["person"] for e in extract_structured_events(SAMPLE_TEXT))

    def test_raw_date_present(self):
        events = extract_structured_events(SAMPLE_TEXT)
        with_raw = [e for e in events if e["raw_date"]]
        assert len(with_raw) > 0

    def test_parsed_date_is_iso_format(self):
        """parsed_date should be YYYY-MM-DD format."""
        events = extract_structured_events(SAMPLE_TEXT)
        for e in events:
            if e["parsed_date"]:
                datetime.strptime(e["parsed_date"], "%Y-%m-%d")  # Should not raise

    def test_parsed_time_is_normalized(self):
        """parsed_time should be HH:MM format."""
        events = extract_structured_events(SAMPLE_TEXT)
        for e in events:
            if e["parsed_time"]:
                assert len(e["parsed_time"]) == 5, f"Bad time: {e['parsed_time']}"
                assert ":" in e["parsed_time"]

    def test_json_serializable(self):
        try:
            json.dumps(extract_structured_events(SAMPLE_TEXT))
        except (TypeError, ValueError) as e:
            pytest.fail(f"Not JSON-serializable: {e}")

    def test_empty_text(self):
        assert extract_structured_events("") == []

    def test_legacy_api_still_works(self):
        events = extract_events(SAMPLE_TEXT)
        if events:
            assert "type" in events[0] and "value" in events[0]


# =========================================================================
# Memory Manager Tests
# =========================================================================

class TestMemoryManager:
    def test_add_and_count(self):
        m = MemoryManager()
        m.add_event({"type": "task", "description": "Test"})
        assert m.count() == 1

    def test_get_all_events(self):
        m = MemoryManager()
        m.add_event({"type": "task", "description": "Buy milk"})
        assert m.get_all_events()[0]["description"] == "Buy milk"

    def test_recorded_at_timestamp(self):
        m = MemoryManager()
        m.add_event({"type": "task", "description": "Test"})
        assert "recorded_at" in m.get_all_events()[0]

    def test_search_case_insensitive(self):
        m = MemoryManager()
        m.add_event({"type": "task", "description": "Call Pharmacy"})
        assert len(m.search_events("pharmacy")) == 1
        assert len(m.search_events("PHARMACY")) == 1

    def test_clear(self):
        m = MemoryManager()
        m.add_event({"type": "task", "description": "Test"})
        m.clear()
        assert m.count() == 0

    def test_save_and_load(self, tmp_path):
        m = MemoryManager()
        m.add_event({"type": "task", "description": "Test save"})
        filepath = str(tmp_path / "test.json")
        m.save_to_file(filepath)
        m2 = MemoryManager()
        m2.load_from_file(filepath)
        assert m2.count() == 1

    def test_load_nonexistent_file(self):
        m = MemoryManager()
        m.load_from_file("nonexistent.json")
        assert m.count() == 0


# =========================================================================
# Query Engine Tests
# =========================================================================

class TestQueryEngine:
    @pytest.fixture
    def engine(self):
        return QueryEngine(_create_test_memory())

    def test_detects_meeting_intent(self, engine):
        assert "meeting" in engine.query("What meetings do I have?").lower()

    def test_detects_task_intent(self, engine):
        assert "task" in engine.query("What tasks do I have?").lower()

    def test_detects_medication_intent(self, engine):
        assert "medication" in engine.query("Did I take medicine?").lower()

    def test_detects_summary_intent(self, engine):
        assert "event" in engine.query("Give me a summary").lower()

    def test_filters_by_tomorrow(self, engine):
        result = engine.query("What meetings do I have tomorrow?")
        assert "tomorrow" in result.lower() and "doctor" in result.lower()

    def test_returns_string(self, engine):
        assert isinstance(engine.query("What meetings do I have?"), str)

    def test_keyword_search(self, engine):
        result = engine.query("Tell me about pharmacy")
        assert "pharmacy" in result.lower() or "prescription" in result.lower()

    def test_empty_memory(self):
        result = QueryEngine(MemoryManager()).query("What meetings do I have?")
        assert "no meeting" in result.lower()

    def test_empty_question(self, engine):
        assert isinstance(engine.query(""), str)


# =========================================================================
# Date Parser Tests (Day 4)
# =========================================================================

class TestDateParser:
    """Tests for core.date_parser module."""

    def test_parse_today(self):
        result = parse_date("today")
        expected = datetime.now().strftime("%Y-%m-%d")
        assert result == expected

    def test_parse_tomorrow(self):
        result = parse_date("tomorrow")
        expected = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == expected

    def test_parse_yesterday(self):
        result = parse_date("yesterday")
        expected = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == expected

    def test_parse_month_day(self):
        result = parse_date("March 15")
        assert result is not None
        assert "-03-15" in result

    def test_parse_none_returns_none(self):
        assert parse_date(None) is None

    def test_parse_empty_returns_none(self):
        assert parse_date("") is None

    def test_parse_gibberish_returns_none(self):
        assert parse_date("xyzzy foobar") is None

    def test_parse_time_am(self):
        assert parse_time("10 AM") == "10:00"

    def test_parse_time_pm(self):
        assert parse_time("3 PM") == "15:00"

    def test_parse_time_with_minutes(self):
        assert parse_time("10:30 AM") == "10:30"

    def test_parse_time_noon(self):
        assert parse_time("noon") == "12:00"

    def test_parse_time_morning(self):
        assert parse_time("morning") == "09:00"

    def test_parse_time_none(self):
        assert parse_time(None) is None

    def test_parse_time_empty(self):
        assert parse_time("") is None

    def test_combine_datetime(self):
        result = combine_datetime("2026-02-25", "10:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 25
        assert result.hour == 10

    def test_combine_datetime_no_time_defaults_9am(self):
        result = combine_datetime("2026-02-25", None)
        assert result is not None
        assert result.hour == 9
        assert result.minute == 0

    def test_combine_datetime_no_date_returns_none(self):
        assert combine_datetime(None, "10:00") is None


# =========================================================================
# Reminder Manager Tests (Day 4)
# =========================================================================

class TestReminderManager:
    """Tests for core.reminder_manager module."""

    def _create_memory_with_upcoming(self) -> MemoryManager:
        """Create memory with an event happening in 10 minutes."""
        now = datetime.now()
        future = now + timedelta(minutes=10)
        memory = MemoryManager()
        memory.add_event({
            "type": "meeting",
            "raw_date": "today",
            "parsed_date": now.strftime("%Y-%m-%d"),
            "time": future.strftime("%I:%M %p"),
            "parsed_time": future.strftime("%H:%M"),
            "person": "Dr. Smith",
            "description": "Doctor appointment",
        })
        return memory

    def test_get_upcoming_events(self):
        memory = self._create_memory_with_upcoming()
        reminder = ReminderManager(memory)
        upcoming = reminder.get_upcoming_events(minutes=60)
        assert len(upcoming) == 1
        assert upcoming[0]["description"] == "Doctor appointment"

    def test_upcoming_has_minutes_until(self):
        memory = self._create_memory_with_upcoming()
        reminder = ReminderManager(memory)
        upcoming = reminder.get_upcoming_events(minutes=60)
        assert "minutes_until" in upcoming[0]
        assert 5 <= upcoming[0]["minutes_until"] <= 15  # ~10 min window

    def test_check_due_events_returns_alerts(self):
        memory = self._create_memory_with_upcoming()
        reminder = ReminderManager(memory)
        alerts = reminder.check_due_events(window_minutes=60)
        assert len(alerts) == 1
        assert "REMINDER" in alerts[0]
        assert "Doctor appointment" in alerts[0]

    def test_check_due_events_deduplicates(self):
        memory = self._create_memory_with_upcoming()
        reminder = ReminderManager(memory)
        alerts1 = reminder.check_due_events(window_minutes=60)
        alerts2 = reminder.check_due_events(window_minutes=60)
        assert len(alerts1) == 1
        assert len(alerts2) == 0  # Already alerted

    def test_no_upcoming_events(self):
        memory = MemoryManager()
        memory.add_event({
            "type": "meeting",
            "raw_date": "yesterday",
            "parsed_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "time": "10 AM", "parsed_time": "10:00",
            "description": "Past event",
        })
        reminder = ReminderManager(memory)
        assert len(reminder.get_upcoming_events(minutes=60)) == 0

    def test_get_todays_schedule(self):
        memory = self._create_memory_with_upcoming()
        reminder = ReminderManager(memory)
        schedule = reminder.get_todays_schedule()
        assert len(schedule) == 1

    def test_format_schedule(self):
        memory = self._create_memory_with_upcoming()
        reminder = ReminderManager(memory)
        schedule = reminder.get_todays_schedule()
        formatted = reminder.format_schedule(schedule)
        assert "MEETING" in formatted
        assert "Doctor" in formatted

    def test_format_schedule_empty(self):
        reminder = ReminderManager(MemoryManager())
        assert "No events" in reminder.format_schedule([])

    def test_start_and_stop_loop(self):
        reminder = ReminderManager(MemoryManager())
        reminder.start_reminder_loop(interval=1)
        assert reminder._running is True
        reminder.stop()
        assert reminder._running is False
