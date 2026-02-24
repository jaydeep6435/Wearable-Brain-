"""
Tests for Day 1 MVP — Core AI Pipeline
========================================
These tests verify the summarizer and event extractor modules
using known input text. No audio files or Whisper model needed.

Run with:  python -m pytest tests/test_pipeline.py -v
"""

import pytest
from core.summarizer import summarize, _split_sentences
from core.event_extractor import extract_events


# ═══════════════════════════════════════════════════════════════════════════
# Shared test data
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_TEXT = (
    "Good morning! I have a doctor appointment tomorrow at 10 AM. "
    "Don't forget to take your medicine after breakfast. "
    "We need to call the pharmacy to refill the prescription. "
    "Your son David is visiting this weekend. "
    "Remember to do your morning exercises before lunch. "
    "The meeting with Dr. Smith is on March 15."
)


# ═══════════════════════════════════════════════════════════════════════════
# Summarizer Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSummarizer:
    """Tests for core.summarizer module."""

    def test_split_sentences_returns_list(self):
        """Sentence splitter should return a non-empty list."""
        sentences = _split_sentences(SAMPLE_TEXT)
        assert isinstance(sentences, list)
        assert len(sentences) > 0

    def test_summarize_returns_string(self):
        """Summarize should return a non-empty string."""
        result = summarize(SAMPLE_TEXT, num_sentences=2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_respects_num_sentences(self):
        """Summary should contain at most num_sentences sentences."""
        for n in [1, 2, 3]:
            result = summarize(SAMPLE_TEXT, num_sentences=n)
            # Count sentences by splitting on sentence-ending punctuation
            sentence_count = len(_split_sentences(result))
            assert sentence_count <= n, (
                f"Expected at most {n} sentences, got {sentence_count}"
            )

    def test_summarize_empty_text(self):
        """Summarizing empty/short text should not crash."""
        result = summarize("", num_sentences=3)
        assert isinstance(result, str)

    def test_summarize_single_sentence(self):
        """Summarizing a single sentence returns that sentence."""
        single = "This is the only sentence here."
        result = summarize(single, num_sentences=3)
        assert "only sentence" in result


# ═══════════════════════════════════════════════════════════════════════════
# Event Extractor Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEventExtractor:
    """Tests for core.event_extractor module."""

    def test_returns_list(self):
        """extract_events should return a list."""
        result = extract_events(SAMPLE_TEXT)
        assert isinstance(result, list)

    def test_detects_dates(self):
        """Should detect date-related words like 'tomorrow'."""
        events = extract_events(SAMPLE_TEXT)
        date_events = [e for e in events if e["type"] == "date"]
        date_values = [e["value"].lower() for e in date_events]
        assert "tomorrow" in date_values, (
            f"Expected 'tomorrow' in dates, got: {date_values}"
        )

    def test_detects_times(self):
        """Should detect time references like '10 AM'."""
        events = extract_events(SAMPLE_TEXT)
        time_events = [e for e in events if e["type"] == "time"]
        time_values = [e["value"].lower() for e in time_events]
        assert any("10" in v and "am" in v for v in time_values), (
            f"Expected '10 AM' in times, got: {time_values}"
        )

    def test_detects_meetings(self):
        """Should detect meetings/appointments."""
        events = extract_events(SAMPLE_TEXT)
        meeting_events = [e for e in events if e["type"] == "meeting"]
        assert len(meeting_events) > 0, "Expected at least one meeting event"

    def test_detects_tasks(self):
        """Should detect tasks/to-dos."""
        events = extract_events(SAMPLE_TEXT)
        task_events = [e for e in events if e["type"] == "task"]
        assert len(task_events) > 0, "Expected at least one task event"

    def test_detects_medication(self):
        """Should detect medication-related events."""
        events = extract_events(SAMPLE_TEXT)
        med_events = [e for e in events if e["type"] == "medication"]
        assert len(med_events) > 0, "Expected at least one medication event"

    def test_event_structure(self):
        """Each event should have type, value, and context keys."""
        events = extract_events(SAMPLE_TEXT)
        for event in events:
            assert "type" in event, "Event missing 'type' key"
            assert "value" in event, "Event missing 'value' key"
            assert "context" in event, "Event missing 'context' key"

    def test_empty_text(self):
        """Extracting from empty text should return empty list."""
        result = extract_events("")
        assert result == []
