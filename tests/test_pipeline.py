"""
Tests for Day 2 MVP -- Pipeline Upgrade
=========================================
Tests cover:
  - Summarizer (original + highlighted)
  - Structured event extraction (JSON format)
  - Memory Manager (add, get, search, persist)

Run with:  python -m pytest tests/test_pipeline.py -v
"""

import json
import os
import pytest

from core.summarizer import summarize, _split_sentences, summarize_with_highlights
from core.event_extractor import extract_events, extract_structured_events
from core.memory_manager import MemoryManager


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


# =========================================================================
# Summarizer Tests (Day 1 + Day 2)
# =========================================================================

class TestSummarizer:
    """Tests for core.summarizer module."""

    def test_split_sentences_returns_list(self):
        sentences = _split_sentences(SAMPLE_TEXT)
        assert isinstance(sentences, list)
        assert len(sentences) > 0

    def test_summarize_returns_string(self):
        result = summarize(SAMPLE_TEXT, num_sentences=2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_respects_num_sentences(self):
        for n in [1, 2, 3]:
            result = summarize(SAMPLE_TEXT, num_sentences=n)
            sentence_count = len(_split_sentences(result))
            assert sentence_count <= n

    def test_summarize_empty_text(self):
        result = summarize("", num_sentences=3)
        assert isinstance(result, str)

    def test_summarize_single_sentence(self):
        single = "This is the only sentence here."
        result = summarize(single, num_sentences=3)
        assert "only sentence" in result


# =========================================================================
# Highlighted Summary Tests (Day 2)
# =========================================================================

class TestHighlightedSummary:
    """Tests for the new summarize_with_highlights function."""

    def test_returns_list_of_dicts(self):
        result = summarize_with_highlights(SAMPLE_TEXT)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_each_item_has_required_keys(self):
        result = summarize_with_highlights(SAMPLE_TEXT)
        for item in result:
            assert "sentence" in item
            assert "important" in item
            assert "tags" in item

    def test_important_flag_is_boolean(self):
        result = summarize_with_highlights(SAMPLE_TEXT, num_sentences=2)
        for item in result:
            assert isinstance(item["important"], bool)

    def test_tags_are_list(self):
        result = summarize_with_highlights(SAMPLE_TEXT)
        for item in result:
            assert isinstance(item["tags"], list)

    def test_detects_meeting_tag(self):
        """Sentences with 'doctor' or 'meeting' should have meeting tag."""
        result = summarize_with_highlights(SAMPLE_TEXT)
        meeting_tagged = [
            item for item in result
            if "meeting" in item["tags"]
        ]
        assert len(meeting_tagged) > 0

    def test_detects_medication_tag(self):
        """Sentences with 'medicine' should have medication tag."""
        result = summarize_with_highlights(SAMPLE_TEXT)
        med_tagged = [
            item for item in result
            if "medication" in item["tags"]
        ]
        assert len(med_tagged) > 0

    def test_empty_text_returns_empty_list(self):
        result = summarize_with_highlights("")
        assert result == []


# =========================================================================
# Structured Event Extractor Tests (Day 2)
# =========================================================================

class TestStructuredExtractor:
    """Tests for the new extract_structured_events function."""

    def test_returns_list(self):
        result = extract_structured_events(SAMPLE_TEXT)
        assert isinstance(result, list)

    def test_events_have_required_fields(self):
        """Each event should have type, date, time, person, description."""
        events = extract_structured_events(SAMPLE_TEXT)
        required_keys = {"type", "date", "time", "person", "description"}
        for event in events:
            assert required_keys.issubset(event.keys()), (
                f"Event missing keys: {required_keys - event.keys()}"
            )

    def test_type_is_valid(self):
        """Event type should be one of meeting, task, medication."""
        valid_types = {"meeting", "task", "medication"}
        events = extract_structured_events(SAMPLE_TEXT)
        for event in events:
            assert event["type"] in valid_types, (
                f"Invalid type: {event['type']}"
            )

    def test_detects_meetings(self):
        events = extract_structured_events(SAMPLE_TEXT)
        meetings = [e for e in events if e["type"] == "meeting"]
        assert len(meetings) > 0

    def test_detects_tasks(self):
        events = extract_structured_events(SAMPLE_TEXT)
        tasks = [e for e in events if e["type"] == "task"]
        assert len(tasks) > 0

    def test_detects_medication(self):
        events = extract_structured_events(SAMPLE_TEXT)
        meds = [e for e in events if e["type"] == "medication"]
        assert len(meds) > 0

    def test_detects_person(self):
        """Should detect 'Dr. Smith' or 'David' as a person."""
        events = extract_structured_events(SAMPLE_TEXT)
        persons = [e["person"] for e in events if e["person"]]
        assert len(persons) > 0, (
            f"Expected at least one person, got none. Events: {events}"
        )

    def test_detects_date_in_event(self):
        """At least one event should have a date field populated."""
        events = extract_structured_events(SAMPLE_TEXT)
        events_with_date = [e for e in events if e["date"]]
        assert len(events_with_date) > 0

    def test_detects_time_in_event(self):
        """At least one event should have a time field populated."""
        events = extract_structured_events(SAMPLE_TEXT)
        events_with_time = [e for e in events if e["time"]]
        assert len(events_with_time) > 0

    def test_json_serializable(self):
        """All events should be JSON-serializable."""
        events = extract_structured_events(SAMPLE_TEXT)
        try:
            json.dumps(events)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Events not JSON-serializable: {e}")

    def test_empty_text(self):
        result = extract_structured_events("")
        assert result == []

    def test_legacy_api_still_works(self):
        """The old extract_events API should still return results."""
        events = extract_events(SAMPLE_TEXT)
        assert isinstance(events, list)
        if events:
            assert "type" in events[0]
            assert "value" in events[0]
            assert "context" in events[0]


# =========================================================================
# Memory Manager Tests (Day 2)
# =========================================================================

class TestMemoryManager:
    """Tests for core.memory_manager module."""

    def test_add_and_count(self):
        memory = MemoryManager()
        assert memory.count() == 0
        memory.add_event({"type": "task", "description": "Buy groceries"})
        assert memory.count() == 1

    def test_add_events_bulk(self):
        memory = MemoryManager()
        memory.add_events([
            {"type": "task", "description": "Task 1"},
            {"type": "meeting", "description": "Meeting 1"},
        ])
        assert memory.count() == 2

    def test_get_all_events(self):
        memory = MemoryManager()
        memory.add_event({"type": "task", "description": "Buy milk"})
        events = memory.get_all_events()
        assert len(events) == 1
        assert events[0]["description"] == "Buy milk"

    def test_recorded_at_timestamp(self):
        """Each event should get a recorded_at timestamp."""
        memory = MemoryManager()
        memory.add_event({"type": "task", "description": "Test"})
        events = memory.get_all_events()
        assert "recorded_at" in events[0]

    def test_get_today_events(self):
        memory = MemoryManager()
        memory.add_event({"type": "task", "date": "tomorrow", "description": "Buy milk"})
        memory.add_event({"type": "task", "date": "next week", "description": "Visit park"})
        today = memory.get_today_events()
        assert len(today) == 1
        assert today[0]["description"] == "Buy milk"

    def test_search_events(self):
        memory = MemoryManager()
        memory.add_event({"type": "meeting", "person": "Dr. Smith", "description": "Doctor visit"})
        memory.add_event({"type": "task", "description": "Buy groceries"})

        results = memory.search_events("doctor")
        assert len(results) == 1
        assert results[0]["person"] == "Dr. Smith"

    def test_search_case_insensitive(self):
        memory = MemoryManager()
        memory.add_event({"type": "task", "description": "Call Pharmacy"})
        assert len(memory.search_events("pharmacy")) == 1
        assert len(memory.search_events("PHARMACY")) == 1

    def test_clear(self):
        memory = MemoryManager()
        memory.add_event({"type": "task", "description": "Test"})
        memory.clear()
        assert memory.count() == 0

    def test_save_and_load(self, tmp_path):
        """Save to file and reload should preserve events."""
        memory = MemoryManager()
        memory.add_event({"type": "task", "description": "Test save"})
        memory.add_event({"type": "meeting", "description": "Test meeting"})

        filepath = str(tmp_path / "test_memory.json")
        memory.save_to_file(filepath)
        assert os.path.isfile(filepath)

        # Load into a fresh manager
        memory2 = MemoryManager()
        memory2.load_from_file(filepath)
        assert memory2.count() == 2
        assert memory2.get_all_events()[0]["description"] == "Test save"

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file should not crash."""
        memory = MemoryManager()
        memory.load_from_file("nonexistent_file.json")
        assert memory.count() == 0
