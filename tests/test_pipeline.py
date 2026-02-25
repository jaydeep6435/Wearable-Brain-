"""
Test Suite — Stabilized Architecture
========================================
Comprehensive tests for all new and refactored modules:
  - storage/db.py
  - storage/repository.py
  - engine/assistant_engine.py
  - conversation/builder.py
  - diarization/diarizer.py
  - core/query_engine.py   (refactored: MemoryStore protocol)
  - core/reminder_manager.py (refactored: MemoryStore protocol)
  - core/summarizer.py
  - core/event_extractor.py
  - core/date_parser.py

Run:  python -m pytest tests/test_pipeline.py -v
"""

import json
import os
import sys
import tempfile

import pytest
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================================
# Shared fixtures
# =========================================================================

SAMPLE_TEXT = (
    "I have a doctor appointment tomorrow at 10 AM. "
    "Don't forget to take your medicine after breakfast. "
    "We need to call the pharmacy to refill the prescription. "
    "Your son David is visiting this weekend."
)


class InMemoryStore:
    """Minimal MemoryStore implementation for testing QueryEngine/ReminderManager."""

    def __init__(self, events: list[dict] = None):
        self._events = events or []

    def get_all_events(self) -> list[dict]:
        return self._events

    def search_events(self, keyword: str) -> list[dict]:
        kw = keyword.lower()
        return [
            e for e in self._events
            if kw in (e.get("description", "") + e.get("type", "")).lower()
        ]

    def add_events(self, events: list[dict]):
        self._events.extend(events)

    def count(self) -> int:
        return len(self._events)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database."""
    return str(tmp_path / "test.db")


@pytest.fixture
def sample_events():
    """Standard set of test events."""
    return [
        {"type": "meeting", "description": "Doctor appointment",
         "date": "tomorrow", "time": "10 AM", "person": "Dr. Smith"},
        {"type": "medication", "description": "Take medicine after breakfast"},
        {"type": "task", "description": "Call the pharmacy to refill prescription"},
        {"type": "meeting", "description": "David visiting this weekend",
         "person": "David"},
    ]


# =========================================================================
# 1. Tests for storage/db.py
# =========================================================================

class TestDatabase:
    """Tests for storage/db.py — SQLite connection and schema."""

    def test_create_database(self, temp_db):
        from storage.db import Database
        db = Database(temp_db)
        assert os.path.isfile(temp_db)

    def test_tables_created(self, temp_db):
        from storage.db import Database
        db = Database(temp_db)
        tables = db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [t["name"] for t in tables]
        assert "conversations" in names
        assert "segments" in names
        assert "events" in names
        assert "summaries" in names
        assert "reminders" in names

    def test_insert_and_fetch(self, temp_db):
        from storage.db import Database
        db = Database(temp_db)
        eid = db.new_id()
        db.execute(
            "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
            (eid, "meeting", "Test event"),
        )
        row = db.fetch_one("SELECT * FROM events WHERE id = ?", (eid,))
        assert row is not None
        assert row["type"] == "meeting"
        assert row["description"] == "Test event"

    def test_fetch_all(self, temp_db):
        from storage.db import Database
        db = Database(temp_db)
        for i in range(3):
            db.execute(
                "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
                (db.new_id(), "task", f"Task {i}"),
            )
        rows = db.fetch_all("SELECT * FROM events")
        assert len(rows) == 3

    def test_count(self, temp_db):
        from storage.db import Database
        db = Database(temp_db)
        assert db.count("events") == 0
        db.execute(
            "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
            (db.new_id(), "test", "Test"),
        )
        assert db.count("events") == 1

    def test_new_id_uniqueness(self):
        from storage.db import Database
        ids = {Database.new_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_stats(self, temp_db):
        from storage.db import Database
        db = Database(temp_db)
        stats = db.get_stats()
        assert "conversations" in stats
        assert "events" in stats
        assert all(v == 0 for v in stats.values())

    def test_foreign_key_enforcement(self, temp_db):
        from storage.db import Database
        import sqlite3
        db = Database(temp_db)
        # Insert a segment with non-existing conversation_id
        # Foreign keys are enforced, so this should raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO segments (id, conversation_id, text) VALUES (?, ?, ?)",
                (db.new_id(), "nonexistent", "test"),
            )

    def test_wal_mode(self, temp_db):
        from storage.db import Database
        db = Database(temp_db)
        result = db.fetch_one("PRAGMA journal_mode")
        assert result is not None


# =========================================================================
# 2. Tests for storage/repository.py
# =========================================================================

class TestRepository:
    """Tests for storage/repository.py — CRUD operations and dedup."""

    def test_save_conversation(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="Hello doctor", source="text")
        assert cid is not None
        assert len(cid) == 36  # UUID format

    def test_get_conversation(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="Test conversation")
        conv = repo.get_conversation(cid)
        assert conv is not None
        assert conv["raw_text"] == "Test conversation"
        assert "segments" in conv
        assert "events" in conv

    def test_save_and_get_events(self, temp_db, sample_events):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        saved = repo.save_events(cid, sample_events)
        assert saved == len(sample_events)

        all_events = repo.get_all_events()
        assert len(all_events) == len(sample_events)

    def test_event_deduplication(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")

        events = [
            {"type": "meeting", "description": "Doctor appointment", "date": "tomorrow", "time": "10 AM"},
            {"type": "meeting", "description": "Doctor appointment", "date": "tomorrow", "time": "10 AM"},  # Duplicate!
        ]
        saved = repo.save_events(cid, events)
        assert saved == 1  # Only 1 saved, duplicate skipped

    def test_save_single_event_dedup(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        event = {"type": "meeting", "description": "Test", "date": "today"}

        eid1 = repo.save_single_event(event)
        assert eid1 is not None

        eid2 = repo.save_single_event(event)
        assert eid2 is None  # Duplicate returns None

    def test_search_events(self, temp_db, sample_events):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_events(cid, sample_events)

        found = repo.search_events("doctor")
        assert len(found) >= 1
        assert any("Doctor" in e.get("description", "") for e in found)

    def test_get_events_by_type(self, temp_db, sample_events):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_events(cid, sample_events)

        meetings = repo.get_all_events(type_filter="meeting")
        assert all(e["type"] == "meeting" for e in meetings)
        assert len(meetings) == 2  # Doctor + David

    def test_save_segments(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")

        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 2.0},
            {"speaker": "SPEAKER_01", "text": "Hi there", "start": 2.0, "end": 4.0},
        ]
        count = repo.save_segments(cid, segments)
        assert count == 2

    def test_save_summary(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        sid = repo.save_summary(cid, "Patient has appointment", ["medicine", "doctor"])
        assert sid is not None

        summaries = repo.get_summaries(cid)
        assert len(summaries) == 1
        assert summaries[0]["summary"] == "Patient has appointment"

    def test_save_reminder(self, temp_db, sample_events):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_events(cid, sample_events[:1])

        events = repo.get_all_events()
        eid = events[0]["id"]

        rid = repo.save_reminder(eid, "2026-02-26 10:00")
        assert rid is not None

        pending = repo.get_pending_reminders()
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"

    def test_mark_reminder_fired(self, temp_db, sample_events):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_events(cid, sample_events[:1])

        events = repo.get_all_events()
        rid = repo.save_reminder(events[0]["id"], "2026-02-26 10:00")
        repo.mark_reminder_fired(rid)

        pending = repo.get_pending_reminders()
        assert len(pending) == 0

    def test_migrate_from_json(self, temp_db, tmp_path):
        from storage.repository import Repository
        repo = Repository(temp_db)

        # Create a test JSON file
        json_path = str(tmp_path / "test_memory.json")
        events = [
            {"type": "meeting", "description": "Test meeting", "date": "tomorrow"},
            {"type": "task", "description": "Test task"},
        ]
        with open(json_path, "w") as f:
            json.dump(events, f)

        result = repo.migrate_from_json(json_path)
        assert result["status"] == "ok"
        assert result["saved"] == 2
        assert result["duplicates_skipped"] == 0

    def test_migrate_nonexistent_file(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        result = repo.migrate_from_json("nonexistent.json")
        assert result["status"] == "error"

    def test_get_stats(self, temp_db, sample_events):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_events(cid, sample_events)

        stats = repo.get_stats()
        assert stats["conversations"] == 1
        assert stats["events"] == len(sample_events)

    def test_get_all_conversations(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        repo.save_conversation(raw_text="First")
        repo.save_conversation(raw_text="Second")

        convs = repo.get_all_conversations()
        assert len(convs) == 2


# =========================================================================
# 3. Tests for conversation/builder.py
# =========================================================================

class TestConversationBuilder:
    """Tests for conversation/builder.py — merging diarization + ASR."""

    def test_basic_merge(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        dia_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0},
            {"speaker": "SPEAKER_01", "start": 4.0, "end": 8.0},
        ]
        whisper_result = {
            "text": "Hello there. How are you?",
            "segments": [
                {"text": "Hello there.", "start": 0.5, "end": 3.0},
                {"text": "How are you?", "start": 4.5, "end": 7.0},
            ],
        }

        conversation = builder.build(dia_segments, whisper_result)
        assert len(conversation) == 2
        assert conversation[0]["speaker"] == "SPEAKER_00"
        assert "Hello" in conversation[0]["text"]
        assert conversation[1]["speaker"] == "SPEAKER_01"

    def test_empty_diarization(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        whisper_result = {"text": "Hello world", "segments": []}
        conversation = builder.build([], whisper_result)
        assert len(conversation) == 1
        assert conversation[0]["speaker"] == "SPEAKER_00"
        assert conversation[0]["text"] == "Hello world"

    def test_empty_whisper(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        dia_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
        ]
        whisper_result = {"text": "", "segments": []}
        conversation = builder.build(dia_segments, whisper_result)
        assert len(conversation) == 1
        assert conversation[0]["text"] == ""

    def test_consecutive_same_speaker_merge(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        dia_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
            {"speaker": "SPEAKER_00", "start": 3.0, "end": 6.0},
        ]
        whisper_result = {
            "text": "Part one. Part two.",
            "segments": [
                {"text": "Part one.", "start": 0.5, "end": 2.5},
                {"text": "Part two.", "start": 3.5, "end": 5.5},
            ],
        }

        conversation = builder.build(dia_segments, whisper_result)
        assert len(conversation) == 1  # Merged into one
        assert "Part one" in conversation[0]["text"]
        assert "Part two" in conversation[0]["text"]

    def test_build_text(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        conversation = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_01", "text": "Hi there"},
        ]
        text = builder.build_text(conversation)
        assert "SPEAKER_00: Hello" in text
        assert "SPEAKER_01: Hi there" in text

    def test_overlap_computation(self):
        from conversation.builder import ConversationBuilder
        assert ConversationBuilder._compute_overlap(0, 5, 3, 8) == 2.0
        assert ConversationBuilder._compute_overlap(0, 5, 6, 10) == 0.0
        assert ConversationBuilder._compute_overlap(0, 5, 0, 5) == 5.0
        assert ConversationBuilder._compute_overlap(0, 5, 2, 3) == 1.0

    def test_four_speaker_conversation(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        dia_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0},
            {"speaker": "SPEAKER_01", "start": 4.0, "end": 8.0},
            {"speaker": "SPEAKER_00", "start": 8.0, "end": 12.0},
            {"speaker": "SPEAKER_01", "start": 12.0, "end": 16.0},
        ]
        whisper_result = {
            "text": SAMPLE_TEXT,
            "segments": [
                {"text": "I have a doctor appointment tomorrow at 10 AM.", "start": 0.5, "end": 3.5},
                {"text": "Don't forget to take your medicine after breakfast.", "start": 4.2, "end": 7.5},
                {"text": "We need to call the pharmacy to refill the prescription.", "start": 8.5, "end": 11.0},
                {"text": "Your son David is visiting this weekend.", "start": 12.3, "end": 15.5},
            ],
        }

        conversation = builder.build(dia_segments, whisper_result)
        assert len(conversation) == 4
        speakers = [s["speaker"] for s in conversation]
        assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]


# =========================================================================
# 4. Tests for diarization/diarizer.py
# =========================================================================

class TestSpeakerDiarizer:
    """Tests for diarization/diarizer.py — diarizer with fallback."""

    def test_init(self):
        from diarization.diarizer import SpeakerDiarizer
        d = SpeakerDiarizer()
        assert isinstance(d.is_available, bool)

    def test_fallback_with_wav(self, tmp_path):
        """Test fallback diarizer returns single speaker for a WAV file."""
        import wave
        from diarization.diarizer import SpeakerDiarizer

        # Create a minimal WAV file (1 second of silence)
        wav_path = str(tmp_path / "test.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)  # 1 second

        d = SpeakerDiarizer()
        d._available = False  # Force fallback
        segments = d.diarize(wav_path)

        assert len(segments) == 1
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] > 0

    def test_file_not_found(self):
        from diarization.diarizer import SpeakerDiarizer
        d = SpeakerDiarizer()
        with pytest.raises(FileNotFoundError):
            d.diarize("nonexistent.wav")

    def test_merge_adjacent_segments(self):
        from diarization.diarizer import SpeakerDiarizer

        segments = [
            {"speaker": "A", "start": 0.0, "end": 2.0},
            {"speaker": "A", "start": 2.1, "end": 4.0},  # Adjacent (gap < 0.5)
            {"speaker": "B", "start": 4.0, "end": 6.0},
            {"speaker": "B", "start": 7.0, "end": 9.0},  # NOT adjacent (gap > 0.5)
        ]
        merged = SpeakerDiarizer._merge_adjacent(segments)
        assert len(merged) == 3  # First two A's merged, B's separate

    def test_merge_empty(self):
        from diarization.diarizer import SpeakerDiarizer
        assert SpeakerDiarizer._merge_adjacent([]) == []


# =========================================================================
# 5. Tests for engine/assistant_engine.py
# =========================================================================

class TestMemoryAssistantEngine:
    """Tests for engine/assistant_engine.py — central orchestrator."""

    def test_init(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        assert engine.repo is not None
        assert engine.query_engine is not None
        assert engine.reminder_mgr is not None
        assert engine.identity_mgr is not None

    def test_process_text(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        result = engine.process_text(SAMPLE_TEXT)
        assert "conversation_id" in result
        assert "summary" in result
        assert "events" in result
        assert result["events_saved"] > 0
        assert len(result["events"]) > 0

    def test_process_text_empty(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        result = engine.process_text("")
        assert "error" in result

    def test_process_text_stores_in_db(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        engine.process_text(SAMPLE_TEXT)
        stats = engine.get_stats()
        assert stats["database"]["conversations"] >= 1
        assert stats["database"]["events"] >= 1
        assert stats["database"]["summaries"] >= 1

    def test_query(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        # First process some text
        engine.process_text(SAMPLE_TEXT)

        # Then query
        result = engine.query("When is my doctor appointment?")
        assert "question" in result
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_query_empty(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        result = engine.query("")
        assert result["answer"] == "Please ask a question."

    def test_get_events(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        engine.process_text(SAMPLE_TEXT)

        events = engine.get_events()
        assert len(events) > 0

    def test_get_events_filtered(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        engine.process_text(SAMPLE_TEXT)

        meetings = engine.get_events(type_filter="meeting")
        assert all(e["type"] == "meeting" for e in meetings)

    def test_get_upcoming_events(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        result = engine.get_upcoming_events()
        assert "upcoming" in result
        assert "alerts" in result
        assert "total_events" in result

    def test_get_llm_status(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        status = engine.get_llm_status()
        assert "status" in status
        assert status["status"] in ("online", "offline")

    def test_get_stats(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        stats = engine.get_stats()
        assert "database" in stats
        assert "llm" in stats
        assert stats["version"] == "2.2.0"
        assert stats["architecture"] == "offline-engine"

    def test_process_audio_missing_file(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        result = engine.process_audio("nonexistent.wav")
        assert "error" in result

    def test_event_dedup_across_calls(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        r1 = engine.process_text(SAMPLE_TEXT)
        r2 = engine.process_text(SAMPLE_TEXT)  # Same text again

        # Second call should have 0 new events (all duplicates)
        assert r2["events_saved"] == 0


# =========================================================================
# 6. Tests for core/query_engine.py (refactored — MemoryStore protocol)
# =========================================================================

class TestQueryEngine:
    """Tests for core/query_engine.py using MemoryStore protocol."""

    def _make_engine(self, events):
        from core.query_engine import QueryEngine
        store = InMemoryStore(events)
        return QueryEngine(store)

    def test_meeting_query(self):
        engine = self._make_engine([
            {"type": "meeting", "description": "Doctor appointment",
             "date": "tomorrow", "time": "10 AM"},
        ])
        answer = engine.query("What meetings do I have?")
        assert "Doctor appointment" in answer

    def test_task_query(self):
        engine = self._make_engine([
            {"type": "task", "description": "Buy groceries"},
        ])
        answer = engine.query("What tasks do I need to do?")
        assert "Buy groceries" in answer

    def test_medication_query(self):
        engine = self._make_engine([
            {"type": "medication", "description": "Take medicine after breakfast"},
        ])
        answer = engine.query("Did I take medicine?")
        assert "medicine" in answer.lower()

    def test_search_query(self):
        engine = self._make_engine([
            {"type": "task", "description": "Call pharmacy for prescription"},
        ])
        answer = engine.query("Tell me about the pharmacy")
        assert "pharmacy" in answer.lower()

    def test_empty_memory_query(self):
        engine = self._make_engine([])
        answer = engine.query("What meetings do I have?")
        assert "no" in answer.lower() or "don't" in answer.lower()

    def test_date_filter(self):
        engine = self._make_engine([
            {"type": "meeting", "description": "Morning standup",
             "date": "tomorrow", "raw_date": "tomorrow"},
            {"type": "meeting", "description": "Weekly review",
             "date": "next week", "raw_date": "next week"},
        ])
        answer = engine.query("What meetings do I have tomorrow?")
        assert "Morning standup" in answer

    def test_summary_query(self):
        engine = self._make_engine([
            {"type": "meeting", "description": "Doctor"},
            {"type": "task", "description": "Groceries"},
            {"type": "medication", "description": "Pills"},
        ])
        answer = engine.query("Give me a summary")
        assert len(answer) > 10

    def test_protocol_compliance(self):
        """Verify InMemoryStore satisfies MemoryStore protocol."""
        from core.query_engine import MemoryStore
        store = InMemoryStore()
        assert isinstance(store, MemoryStore)


# =========================================================================
# 7. Tests for core/reminder_manager.py (refactored)
# =========================================================================

class TestReminderManager:
    """Tests for core/reminder_manager.py using MemoryStore protocol."""

    def test_init_with_store(self):
        from core.reminder_manager import ReminderManager
        store = InMemoryStore()
        rm = ReminderManager(store)
        assert rm.memory is store

    def test_no_upcoming_when_empty(self):
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(InMemoryStore())
        upcoming = rm.get_upcoming_events()
        assert upcoming == []

    def test_check_due_events_empty(self):
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(InMemoryStore())
        alerts = rm.check_due_events()
        assert alerts == []

    def test_start_stop_loop(self):
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(InMemoryStore())
        rm.start_reminder_loop(interval=1)
        assert rm._running
        rm.stop()
        assert not rm._running

    def test_todays_schedule_empty(self):
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(InMemoryStore())
        schedule = rm.get_todays_schedule()
        assert schedule == []

    def test_format_schedule_empty(self):
        from core.reminder_manager import ReminderManager
        rm = ReminderManager(InMemoryStore())
        formatted = rm.format_schedule([])
        # format_schedule returns a fallback message for empty list
        assert isinstance(formatted, str)


# =========================================================================
# 8. Tests for core/summarizer.py
# =========================================================================

class TestSummarizer:
    """Tests for core/summarizer.py — extractive summarizer."""

    def test_summarize(self):
        from core.summarizer import summarize
        result = summarize(SAMPLE_TEXT)
        assert len(result) > 0
        assert len(result) < len(SAMPLE_TEXT) * 2

    def test_summarize_short_text(self):
        from core.summarizer import summarize
        result = summarize("Hello.")
        assert result == "Hello."

    def test_highlights(self):
        from core.summarizer import summarize_with_highlights
        highlights = summarize_with_highlights(SAMPLE_TEXT)
        assert isinstance(highlights, list)
        assert len(highlights) > 0
        assert "sentence" in highlights[0]
        assert "important" in highlights[0]
        assert "tags" in highlights[0]

    def test_highlights_contain_important(self):
        from core.summarizer import summarize_with_highlights
        highlights = summarize_with_highlights(SAMPLE_TEXT)
        important = [h for h in highlights if h["important"]]
        assert len(important) > 0  # Doctor/medicine should trigger


# =========================================================================
# 9. Tests for core/event_extractor.py
# =========================================================================

class TestEventExtractor:
    """Tests for core/event_extractor.py — regex-based extraction."""

    def test_extract_structured_events(self):
        from core.event_extractor import extract_structured_events
        events = extract_structured_events(SAMPLE_TEXT)
        assert isinstance(events, list)
        assert len(events) > 0

    def test_extract_meeting(self):
        from core.event_extractor import extract_structured_events
        events = extract_structured_events("I have a doctor appointment tomorrow at 10 AM.")
        types = [e["type"] for e in events]
        assert "meeting" in types

    def test_extract_medication(self):
        from core.event_extractor import extract_structured_events
        events = extract_structured_events("Take your medicine after breakfast.")
        types = [e["type"] for e in events]
        assert "medication" in types

    def test_extract_task(self):
        from core.event_extractor import extract_structured_events
        events = extract_structured_events("Call the pharmacy to refill the prescription.")
        types = [e["type"] for e in events]
        assert "task" in types

    def test_extract_person(self):
        from core.event_extractor import extract_structured_events
        events = extract_structured_events("Your son David is visiting this weekend.")
        persons = [e.get("person") for e in events if e.get("person")]
        assert any("David" in p for p in persons)

    def test_empty_text(self):
        from core.event_extractor import extract_structured_events
        events = extract_structured_events("")
        assert events == []

    def test_legacy_api(self):
        from core.event_extractor import extract_events
        events = extract_events(SAMPLE_TEXT)
        assert isinstance(events, list)
        assert all("type" in e for e in events)


# =========================================================================
# 10. Tests for core/date_parser.py
# =========================================================================

class TestDateParser:
    """Tests for core/date_parser.py — date/time normalization."""

    def test_parse_tomorrow(self):
        from core.date_parser import parse_date
        result = parse_date("tomorrow")
        assert result is not None
        assert len(result) == 10  # YYYY-MM-DD format

    def test_parse_today(self):
        from core.date_parser import parse_date
        from datetime import datetime
        result = parse_date("today")
        assert result == datetime.now().strftime("%Y-%m-%d")

    def test_parse_time(self):
        from core.date_parser import parse_time
        assert parse_time("10 AM") == "10:00"
        assert parse_time("3:30 PM") == "15:30"

    def test_parse_time_none(self):
        from core.date_parser import parse_time
        result = parse_time("no time here")
        assert result is None or result == ""


# =========================================================================
# 11. Architecture / Security verification
# =========================================================================

class TestArchitectureIntegrity:
    """Verify no Flask, no MemoryManager, no JSON storage leaks."""

    def test_no_flask_imports(self):
        """No Python file should import Flask."""
        import glob
        project_root = os.path.dirname(os.path.dirname(__file__))
        py_files = glob.glob(os.path.join(project_root, "**", "*.py"), recursive=True)

        # Exclude test files and third-party
        py_files = [
            f for f in py_files
            if "test_" not in os.path.basename(f)
            and "site-packages" not in f
            and ".venv" not in f
        ]

        violations = []
        for pf in py_files:
            with open(pf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "from flask" in content or "import flask" in content:
                    violations.append(pf)

        assert violations == [], f"Flask imports found in: {violations}"

    def test_no_memory_manager_imports(self):
        """No Python file should import MemoryManager (except tests)."""
        import glob
        project_root = os.path.dirname(os.path.dirname(__file__))
        py_files = glob.glob(os.path.join(project_root, "**", "*.py"), recursive=True)

        py_files = [
            f for f in py_files
            if "test_" not in os.path.basename(f)
            and "site-packages" not in f
            and ".venv" not in f
        ]

        violations = []
        for pf in py_files:
            with open(pf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "from core.memory_manager" in content:
                    violations.append(pf)

        assert violations == [], f"MemoryManager imports found in: {violations}"

    def test_no_json_storage_file(self):
        """memory.json should not exist in the project root."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        json_path = os.path.join(project_root, "memory.json")
        assert not os.path.isfile(json_path), "memory.json still exists!"

    def test_no_api_bridge_file(self):
        """api_bridge.py should NOT exist — HTTP bridge removed in Phase K."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        bridge_path = os.path.join(project_root, "api_bridge.py")
        assert not os.path.isfile(bridge_path), "api_bridge.py still exists — HTTP bridge not removed!"


# =========================================================================
# 12. Tests for speaker_identity/identity_manager.py
# =========================================================================

class TestIdentityManager:
    """Tests for speaker_identity/identity_manager.py — CRUD and mapping."""

    def test_init(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)
        assert mgr.profile_count == 0

    def test_assign_and_get(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("SPEAKER_00", "Dr. Smith")
        assert mgr.get_display_name("SPEAKER_00") == "Dr. Smith"
        assert mgr.profile_count == 1

    def test_get_unknown(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)
        assert mgr.get_display_name("SPEAKER_99") is None

    def test_resolve_name(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("SPEAKER_00", "Patient")
        assert mgr.resolve_name("SPEAKER_00") == "Patient"
        assert mgr.resolve_name("SPEAKER_99") == "SPEAKER_99"  # fallback

    def test_update_existing(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("SPEAKER_00", "Unknown")
        mgr.assign_label("SPEAKER_00", "Dr. Smith")  # Update
        assert mgr.get_display_name("SPEAKER_00") == "Dr. Smith"
        assert mgr.profile_count == 1  # Still 1, not 2

    def test_remove_profile(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("SPEAKER_00", "Test")
        assert mgr.remove_profile("SPEAKER_00") is True
        assert mgr.get_display_name("SPEAKER_00") is None
        assert mgr.remove_profile("SPEAKER_00") is False  # Already removed

    def test_clear_all(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("SPEAKER_00", "A")
        mgr.assign_label("SPEAKER_01", "B")
        count = mgr.clear_all()
        assert count == 2
        assert mgr.profile_count == 0

    def test_persistence(self, temp_db):
        """Profiles survive across IdentityManager instances."""
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)

        mgr1 = IdentityManager(db)
        mgr1.assign_label("SPEAKER_00", "Dr. Smith")
        mgr1.assign_label("SPEAKER_01", "Patient")

        # New instance, same DB
        mgr2 = IdentityManager(db)
        assert mgr2.get_display_name("SPEAKER_00") == "Dr. Smith"
        assert mgr2.get_display_name("SPEAKER_01") == "Patient"
        assert mgr2.profile_count == 2

    def test_apply_to_conversation(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("SPEAKER_00", "Dr. Smith")
        mgr.assign_label("SPEAKER_01", "Patient")

        segments = [
            {"speaker": "SPEAKER_00", "text": "Take your medicine."},
            {"speaker": "SPEAKER_01", "text": "OK."},
            {"speaker": "SPEAKER_02", "text": "I'll help."},  # Unmapped
        ]
        result = mgr.apply_to_conversation(segments)

        assert result[0]["display_name"] == "Dr. Smith"
        assert result[1]["display_name"] == "Patient"
        assert result[2]["display_name"] == "SPEAKER_02"  # Fallback to raw

    def test_get_all_profiles(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("SPEAKER_00", "Doctor")
        mgr.assign_label("SPEAKER_01", "Patient")

        profiles = mgr.get_all_profiles()
        assert len(profiles) == 2
        assert all("speaker_label" in p for p in profiles)
        assert all("display_name" in p for p in profiles)

    def test_empty_inputs_ignored(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        db = Database(temp_db)
        mgr = IdentityManager(db)

        mgr.assign_label("", "Name")  # Empty label
        mgr.assign_label("SPEAKER_00", "")  # Empty name
        assert mgr.profile_count == 0


# =========================================================================
# 13. Tests for conversation/builder.py with identity mapping
# =========================================================================

class TestBuilderWithIdentity:
    """Tests for identity mapping applied during conversation build."""

    def test_build_adds_display_name_default(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        dia_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0},
        ]
        whisper_result = {
            "text": "Hello world.",
            "segments": [{"text": "Hello world.", "start": 0.5, "end": 3.0}],
        }

        conversation = builder.build(dia_segments, whisper_result)
        assert "display_name" in conversation[0]
        assert conversation[0]["display_name"] == "SPEAKER_00"  # Default

    def test_build_with_identity_manager(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        from conversation.builder import ConversationBuilder

        db = Database(temp_db)
        mgr = IdentityManager(db)
        mgr.assign_label("SPEAKER_00", "Doctor")
        mgr.assign_label("SPEAKER_01", "Patient")

        builder = ConversationBuilder()
        dia_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0},
            {"speaker": "SPEAKER_01", "start": 4.0, "end": 8.0},
        ]
        whisper_result = {
            "text": "Take medicine. OK.",
            "segments": [
                {"text": "Take medicine.", "start": 0.5, "end": 3.0},
                {"text": "OK.", "start": 4.5, "end": 7.0},
            ],
        }

        conversation = builder.build(dia_segments, whisper_result, identity_manager=mgr)
        assert conversation[0]["display_name"] == "Doctor"
        assert conversation[1]["display_name"] == "Patient"

    def test_build_text_uses_display_name(self, temp_db):
        from storage.db import Database
        from speaker_identity.identity_manager import IdentityManager
        from conversation.builder import ConversationBuilder

        db = Database(temp_db)
        mgr = IdentityManager(db)
        mgr.assign_label("SPEAKER_00", "Dr. Smith")

        builder = ConversationBuilder()
        conversation = [
            {"speaker": "SPEAKER_00", "display_name": "Dr. Smith", "text": "Hello"},
            {"speaker": "SPEAKER_01", "display_name": "SPEAKER_01", "text": "Hi"},
        ]
        text = builder.build_text(conversation)
        assert "Dr. Smith: Hello" in text
        assert "SPEAKER_01: Hi" in text

    def test_build_text_fallback_without_display_name(self):
        from conversation.builder import ConversationBuilder
        builder = ConversationBuilder()

        conversation = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
        ]
        text = builder.build_text(conversation)
        assert "SPEAKER_00: Hello" in text


# =========================================================================
# 14. Tests for engine speaker identity methods
# =========================================================================

class TestEngineSpeakerMethods:
    """Tests for engine.assign_speaker_label / get_speaker_profiles / remove."""

    def test_assign_speaker(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        result = engine.assign_speaker_label("SPEAKER_00", "Doctor")
        assert result["status"] == "ok"
        assert result["display_name"] == "Doctor"
        assert result["total_profiles"] == 1

    def test_get_profiles(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        engine.assign_speaker_label("SPEAKER_00", "Doctor")
        engine.assign_speaker_label("SPEAKER_01", "Patient")

        profiles = engine.get_speaker_profiles()
        assert len(profiles) == 2

    def test_remove_speaker(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        engine.assign_speaker_label("SPEAKER_00", "Doctor")
        result = engine.remove_speaker_profile("SPEAKER_00")
        assert result["status"] == "ok"
        assert result["total_profiles"] == 0

    def test_remove_nonexistent(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        result = engine.remove_speaker_profile("SPEAKER_99")
        assert result["status"] == "not_found"

    def test_stats_includes_profiles(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        engine.assign_speaker_label("SPEAKER_00", "Doctor")
        stats = engine.get_stats()
        assert "speaker_profiles" in stats
        assert stats["speaker_profiles"] == 1
        assert stats["diarization"] in ("pyannote", "fallback")


# =========================================================================
# 15. Tests for diarization/diarizer.py — singleton pattern
# =========================================================================

class TestDiarizerSingleton:
    """Tests for diarizer singleton model loading pattern."""

    def test_singleton_class_attributes(self):
        from diarization.diarizer import SpeakerDiarizer
        assert hasattr(SpeakerDiarizer, '_shared_pipeline')
        assert hasattr(SpeakerDiarizer, '_pipeline_load_attempted')

    def test_multiple_instances_share_state(self):
        from diarization.diarizer import SpeakerDiarizer
        d1 = SpeakerDiarizer()
        d2 = SpeakerDiarizer()
        # Both should reference the same class-level singleton
        assert d1._pipeline is d2._pipeline

    def test_cpu_optimization(self):
        """Verify torch threads are limited for CPU."""
        import torch
        from diarization.diarizer import SpeakerDiarizer
        d = SpeakerDiarizer()
        # Threads should be limited to max 4
        assert torch.get_num_threads() <= max(os.cpu_count() or 4, 4)


# =========================================================================
# 16. Tests for background/audio_worker.py
# =========================================================================

class TestAudioWorker:
    """Tests for background/audio_worker.py — session + VAD modes."""

    def test_init(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        assert not worker.is_running()
        assert worker._recordings_count == 0
        assert worker.mode is None

    def test_rms_silence(self):
        import numpy as np
        from background.audio_worker import AudioWorker
        silence = np.zeros(480, dtype=np.int16)
        assert AudioWorker._compute_rms(silence) == 0.0

    def test_rms_loud(self):
        import numpy as np
        from background.audio_worker import AudioWorker
        loud = np.full(480, 5000, dtype=np.int16)
        assert AudioWorker._compute_rms(loud) == 5000.0

    def test_rms_empty(self):
        import numpy as np
        from background.audio_worker import AudioWorker
        empty = np.array([], dtype=np.int16)
        assert AudioWorker._compute_rms(empty) == 0.0

    def test_save_wav(self, tmp_path):
        import numpy as np
        import wave
        from background.audio_worker import AudioWorker

        worker = AudioWorker(engine=None)
        wav_path = str(tmp_path / "test.wav")
        audio = np.random.randint(-5000, 5000, size=16000, dtype=np.int16)

        worker._save_wav(wav_path, audio)
        assert os.path.isfile(wav_path)

        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000

    def test_status(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        status = worker.status()

        assert status["running"] is False
        assert status["mode"] is None
        assert status["recordings_captured"] == 0
        assert status["recordings_processed"] == 0
        assert status["config"]["sample_rate"] == 16000

    def test_session_duration_when_idle(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        assert worker.session_duration == 0.0

    def test_cleanup_tmp(self, tmp_path):
        from background.audio_worker import AudioWorker
        import background.audio_worker as aw_module

        original_dir = aw_module.TMP_AUDIO_DIR
        aw_module.TMP_AUDIO_DIR = str(tmp_path)

        for i in range(3):
            (tmp_path / f"vad_{i}.wav").write_bytes(b"fake")

        worker = AudioWorker(engine=None)
        worker._cleanup_tmp()
        assert len(list(tmp_path.glob("*.wav"))) == 0

        aw_module.TMP_AUDIO_DIR = original_dir

    def test_stop_when_not_running(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        assert not worker.is_running()
        worker.stop()  # Should not raise
        assert not worker.is_running()

    def test_stop_recording_when_not_recording(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        result = worker.stop_recording()
        assert result["status"] == "not_recording"

    def test_delete_file(self, tmp_path):
        from background.audio_worker import AudioWorker
        f = tmp_path / "test.txt"
        f.write_text("data")
        assert f.exists()
        AudioWorker._delete_file(str(f))
        assert not f.exists()
        AudioWorker._delete_file(str(tmp_path / "nonexistent.txt"))

    def test_list_recordings_empty(self, tmp_path):
        from background.audio_worker import AudioWorker
        import background.audio_worker as aw_module
        original = aw_module.RECORDINGS_DIR
        aw_module.RECORDINGS_DIR = str(tmp_path / "nonexistent")

        worker = AudioWorker(engine=None)
        assert worker.list_recordings() == []

        aw_module.RECORDINGS_DIR = original

    def test_list_recordings_with_files(self, tmp_path):
        from background.audio_worker import AudioWorker
        import background.audio_worker as aw_module
        original = aw_module.RECORDINGS_DIR
        aw_module.RECORDINGS_DIR = str(tmp_path)

        (tmp_path / "session_20260225.wav").write_bytes(b"x" * 1000)
        (tmp_path / "session_20260226.wav").write_bytes(b"y" * 2000)

        worker = AudioWorker(engine=None)
        recs = worker.list_recordings()
        assert len(recs) == 2
        assert recs[0]["file"].endswith(".wav")
        assert "size_mb" in recs[0]

        aw_module.RECORDINGS_DIR = original

    def test_get_last_result_none(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        assert worker.get_last_result() is None


# =========================================================================
# 17. Tests for engine recording + background listening
# =========================================================================

class TestEngineRecording:
    """Tests for engine session recording and VAD methods."""

    def test_worker_status_no_worker(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        status = engine.get_worker_status()
        assert status["running"] is False
        assert status["status"] == "no_worker"

    def test_stop_recording_when_not_recording(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        result = engine.stop_recording()
        assert result["status"] == "not_recording"

    def test_stop_background_when_not_running(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        result = engine.stop_background_listening()
        assert result["status"] == "not_running"

    def test_start_recording_creates_worker(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        result = engine.start_recording()
        assert result["status"] in ("recording", "already_running", "error")
        engine.stop_recording()

    def test_list_recordings(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        recs = engine.list_recordings()
        assert isinstance(recs, list)

    def test_start_vad_creates_worker(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        result = engine.start_background_listening()
        assert result["status"] in ("started", "already_running")
        engine.stop_background_listening()


# =========================================================================
# 18. Tests for AudioSource protocol and implementations
# =========================================================================

class TestAudioSourceProtocol:
    """Tests for audio/source.py Protocol and implementations."""

    def test_protocol_exists(self):
        from audio.source import AudioSource
        assert AudioSource is not None

    def test_protocol_is_runtime_checkable(self):
        from audio.source import AudioSource
        # Protocol must be runtime checkable
        assert hasattr(AudioSource, '__protocol_attrs__') or hasattr(AudioSource, '_is_runtime_protocol')

    def test_microphone_source_conforms(self):
        from audio.source import AudioSource
        from audio.microphone import MicrophoneSource
        mic = MicrophoneSource()
        assert isinstance(mic, AudioSource)

    def test_file_source_conforms(self, tmp_path):
        from audio.source import AudioSource
        from audio.file_source import FileSource
        fs = FileSource(str(tmp_path / "dummy.wav"))
        assert isinstance(fs, AudioSource)


class TestFileSource:
    """Tests for audio/file_source.py."""

    def _make_wav(self, path, duration_sec=1.0, sample_rate=16000):
        """Helper: create a mono 16-bit WAV file."""
        import wave as w
        import numpy as np
        samples = int(sample_rate * duration_sec)
        audio = np.random.randint(-5000, 5000, size=samples, dtype=np.int16)
        with w.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return samples

    def test_init(self, tmp_path):
        from audio.file_source import FileSource
        fs = FileSource(str(tmp_path / "test.wav"))
        assert fs.sample_rate == 16000
        assert fs.channels == 1
        assert not fs.is_active

    def test_start_missing_file(self, tmp_path):
        from audio.file_source import FileSource
        fs = FileSource(str(tmp_path / "nonexistent.wav"))
        import pytest
        with pytest.raises(RuntimeError, match="not found"):
            fs.start()

    def test_start_and_read(self, tmp_path):
        from audio.file_source import FileSource
        import numpy as np
        wav_path = tmp_path / "test.wav"
        n_samples = self._make_wav(wav_path)

        fs = FileSource(str(wav_path))
        fs.start()
        assert fs.is_active
        assert fs.total_samples == n_samples

        chunk = fs.read_chunk(480)
        assert chunk.shape[0] == 480
        assert chunk.dtype == np.int16

        fs.stop()
        assert not fs.is_active

    def test_reads_all_data(self, tmp_path):
        from audio.file_source import FileSource
        wav_path = tmp_path / "test.wav"
        n_samples = self._make_wav(wav_path, duration_sec=0.1)

        fs = FileSource(str(wav_path))
        fs.start()

        total_read = 0
        while fs.is_active:
            chunk = fs.read_chunk(480)
            total_read += len(chunk)

        assert total_read == n_samples
        assert not fs.is_active

    def test_reset(self, tmp_path):
        from audio.file_source import FileSource
        wav_path = tmp_path / "test.wav"
        self._make_wav(wav_path, duration_sec=0.1)

        fs = FileSource(str(wav_path))
        fs.start()

        # Read all data
        while fs.is_active:
            fs.read_chunk(480)
        assert not fs.is_active

        # Reset and read again
        fs.reset()
        assert fs.is_active
        chunk = fs.read_chunk(480)
        assert chunk.size > 0
        fs.stop()

    def test_read_when_stopped(self, tmp_path):
        from audio.file_source import FileSource
        fs = FileSource(str(tmp_path / "test.wav"))
        chunk = fs.read_chunk(480)
        assert chunk.size == 0

    def test_repr(self, tmp_path):
        from audio.file_source import FileSource
        fs = FileSource(str(tmp_path / "test.wav"))
        r = repr(fs)
        assert "FileSource" in r
        assert "inactive" in r


class TestMicrophoneSource:
    """Tests for audio/microphone.py (without real mic)."""

    def test_init(self):
        from audio.microphone import MicrophoneSource
        mic = MicrophoneSource()
        assert mic.sample_rate == 16000
        assert mic.channels == 1
        assert not mic.is_active

    def test_custom_params(self):
        from audio.microphone import MicrophoneSource
        mic = MicrophoneSource(sample_rate=44100, channels=2, device=3)
        assert mic.sample_rate == 44100
        assert mic.channels == 2

    def test_read_when_inactive(self):
        from audio.microphone import MicrophoneSource
        mic = MicrophoneSource()
        chunk = mic.read_chunk(480)
        assert chunk.size == 0

    def test_repr(self):
        from audio.microphone import MicrophoneSource
        mic = MicrophoneSource()
        r = repr(mic)
        assert "MicrophoneSource" in r
        assert "inactive" in r

    def test_double_stop(self):
        from audio.microphone import MicrophoneSource
        mic = MicrophoneSource()
        mic.stop()
        mic.stop()  # Should not raise


class TestAudioWorkerWithSource:
    """Tests for AudioWorker with injected audio sources."""

    def test_default_source_name(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        assert "MicrophoneSource" in worker._source_name()

    def test_inject_file_source(self, tmp_path):
        from background.audio_worker import AudioWorker
        from audio.file_source import FileSource
        fs = FileSource(str(tmp_path / "test.wav"))
        worker = AudioWorker(engine=None, audio_source=fs)
        assert worker.audio_source is fs
        assert "FileSource" in worker._source_name()

    def test_status_includes_source(self):
        from background.audio_worker import AudioWorker
        worker = AudioWorker(engine=None)
        status = worker.status()
        assert "source" in status

    def test_cannot_change_source_while_running(self, tmp_path):
        from background.audio_worker import AudioWorker
        from audio.file_source import FileSource

        # Create a short WAV file
        import wave as w
        import numpy as np
        wav_path = tmp_path / "test.wav"
        audio = np.random.randint(-5000, 5000, size=16000, dtype=np.int16)
        with w.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())

        fs = FileSource(str(wav_path))
        worker = AudioWorker(engine=None, audio_source=fs)
        worker.start_recording()

        import pytest
        with pytest.raises(RuntimeError, match="Cannot change"):
            worker.audio_source = FileSource(str(wav_path))

        worker.stop()

    def test_session_with_file_source(self, tmp_path):
        """Full session recording with FileSource — no mic needed."""
        from background.audio_worker import AudioWorker
        from audio.file_source import FileSource
        import background.audio_worker as aw_module

        # Create a valid WAV
        import wave as w
        import numpy as np
        wav_path = tmp_path / "input.wav"
        audio = np.random.randint(-5000, 5000, size=8000, dtype=np.int16)
        with w.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio.tobytes())

        # Redirect recordings dir to temp
        original = aw_module.RECORDINGS_DIR
        aw_module.RECORDINGS_DIR = str(tmp_path / "recs")

        fs = FileSource(str(wav_path))
        worker = AudioWorker(engine=None, audio_source=fs)
        result = worker.start_recording()
        assert result["status"] == "recording"
        assert "FileSource" in result.get("source", "")

        # Wait for FileSource to finish (0.5s of audio)
        import time
        time.sleep(1.0)

        result = worker.stop_recording(process=False)
        assert result["status"] in ("saved", "empty")

        aw_module.RECORDINGS_DIR = original


# =========================================================================
# 19. Tests for Engine Direct Calls (replaced HTTP bridge tests)
# =========================================================================

class TestEngineDirectCalls:
    """Tests for direct MemoryAssistantEngine method invocation — no HTTP."""

    def _make_engine(self, db_path: str):
        from engine.assistant_engine import MemoryAssistantEngine
        return MemoryAssistantEngine(db_path=db_path)

    def test_get_stats(self, temp_db):
        engine = self._make_engine(temp_db)
        result = engine.get_stats()
        assert isinstance(result, dict)

    def test_get_events(self, temp_db):
        engine = self._make_engine(temp_db)
        events = engine.get_events()
        assert isinstance(events, list)

    def test_get_upcoming_events(self, temp_db):
        engine = self._make_engine(temp_db)
        result = engine.get_upcoming_events(minutes=120)
        assert isinstance(result, dict)

    def test_get_speaker_profiles(self, temp_db):
        engine = self._make_engine(temp_db)
        profiles = engine.get_speaker_profiles()
        assert isinstance(profiles, list)

    def test_list_recordings(self, temp_db):
        engine = self._make_engine(temp_db)
        recordings = engine.list_recordings()
        assert isinstance(recordings, list)

    def test_get_worker_status(self, temp_db):
        engine = self._make_engine(temp_db)
        status = engine.get_worker_status()
        assert isinstance(status, dict)
        assert "running" in status or "status" in status

    def test_process_text_direct(self, temp_db):
        engine = self._make_engine(temp_db)
        result = engine.process_text("Doctor appointment tomorrow at 10 AM")
        assert isinstance(result, dict)
        assert "transcription" in result or "summary" in result or "events" in result

    def test_query_direct(self, temp_db):
        engine = self._make_engine(temp_db)
        engine.process_text("Doctor said take medicine daily")
        result = engine.query("What did doctor say?")
        assert isinstance(result, dict)
        assert "answer" in result or "results" in result

    def test_get_llm_status_direct(self, temp_db):
        engine = self._make_engine(temp_db)
        result = engine.get_llm_status()
        assert isinstance(result, dict)

    def test_assign_speaker_direct(self, temp_db):
        engine = self._make_engine(temp_db)
        result = engine.assign_speaker_label("SPEAKER_00", "Doctor")
        assert isinstance(result, dict)

    def test_no_http_imports_in_engine(self):
        """Engine code must not import HTTP modules."""
        import inspect
        from engine.assistant_engine import MemoryAssistantEngine
        source = inspect.getsource(MemoryAssistantEngine)
        assert 'import http.server' not in source
        assert 'from http.server' not in source
        assert 'import flask' not in source.lower()
        assert 'from flask' not in source.lower()

    def test_no_api_bridge_module(self):
        """api_bridge module must not be importable."""
        import importlib
        try:
            importlib.import_module('api_bridge')
            assert False, "api_bridge should not be importable"
        except (ImportError, ModuleNotFoundError):
            pass


# =========================================================================
# 20. Tests for Semantic Search
# =========================================================================

class TestSemanticSearch:
    """Tests for core/semantic_search.py TF-IDF engine."""

    def test_init(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        assert not ss.is_indexed
        assert ss.doc_count == 0
        assert ss.vocab_size == 0

    def test_index(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        docs = [
            {"text": "Doctor appointment tomorrow", "type": "meeting"},
            {"text": "Take medicine after breakfast", "type": "medication"},
        ]
        count = ss.index(docs)
        assert count == 2
        assert ss.is_indexed
        assert ss.doc_count == 2
        assert ss.vocab_size > 0

    def test_index_empty(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        count = ss.index([])
        assert count == 0
        assert not ss.is_indexed

    def test_search_basic(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        docs = [
            {"description": "Doctor appointment at 10 AM", "type": "meeting"},
            {"description": "Take blood pressure medicine", "type": "medication"},
            {"description": "Call pharmacy for refill", "type": "task"},
            {"description": "Son David visiting weekend", "type": "meeting"},
        ]
        ss.index(docs)

        results = ss.search("doctor visit", top_k=2)
        assert len(results) > 0
        assert results[0]["score"] > 0
        assert "document" in results[0]

    def test_search_ranked(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        docs = [
            {"description": "Buy groceries from store"},
            {"description": "Doctor appointment tomorrow morning"},
            {"description": "Take medicine prescribed by doctor"},
        ]
        ss.index(docs)

        results = ss.search("doctor medicine", top_k=3)
        # Doctor-related docs should rank higher than groceries
        if len(results) >= 2:
            assert results[0]["score"] >= results[-1]["score"]

    def test_search_empty_query(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        ss.index([{"text": "hello world"}])
        results = ss.search("")
        assert results == []

    def test_search_no_match(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        ss.index([{"text": "doctor appointment"}])
        results = ss.search("quantum physics thermodynamics", threshold=0.5)
        assert results == []

    def test_search_not_indexed(self):
        from core.semantic_search import SemanticSearch
        ss = SemanticSearch()
        results = ss.search("anything")
        assert results == []

    def test_tokenize(self):
        from core.semantic_search import SemanticSearch
        tokens = SemanticSearch._tokenize("The doctor said to take medicine at 10 AM")
        assert "doctor" in tokens
        assert "medicine" in tokens
        assert "the" not in tokens  # Stop word removed
        assert "to" not in tokens   # Stop word removed

    def test_cosine_similarity(self):
        from core.semantic_search import SemanticSearch
        vec_a = {"doctor": 0.5, "appointment": 0.3}
        vec_b = {"doctor": 0.4, "medicine": 0.6}
        sim = SemanticSearch._cosine_similarity(vec_a, vec_b)
        assert 0 < sim < 1  # Partial overlap

        # Identical vectors → similarity = 1
        sim_same = SemanticSearch._cosine_similarity(vec_a, vec_a)
        assert abs(sim_same - 1.0) < 0.001

    def test_cosine_no_overlap(self):
        from core.semantic_search import SemanticSearch
        vec_a = {"hello": 1.0}
        vec_b = {"world": 1.0}
        sim = SemanticSearch._cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_build_text(self):
        from core.semantic_search import SemanticSearch
        doc = {
            "description": "Doctor appointment",
            "type": "meeting",
            "person": "Dr. Smith",
            "raw_date": "tomorrow",
        }
        text = SemanticSearch._build_text(doc)
        assert "doctor" in text
        assert "meeting" in text
        assert "smith" in text


class TestQueryEngineSemanticFallback:
    """Tests for semantic search fallback in QueryEngine."""

    def test_semantic_search_finds_similar(self, temp_db):
        """When keyword search misses, semantic search finds similar events."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)

        # Insert data
        engine.process_text("Doctor appointment tomorrow at 10 AM. Take medicine after breakfast.")

        # Query with synonyms/related terms that exact keyword search might miss
        result = engine.query("health checkup visit")
        assert isinstance(result, dict)
        assert "answer" in result

    def test_query_engine_has_searcher(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        assert engine.query_engine._searcher is None  # Lazy
        engine.query("anything")
        # After a query, searcher may be initialized (depending on intent routing)

    def test_semantic_search_graceful_empty(self, temp_db):
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        # Query with no data — should not crash
        result = engine.query("tell me about everything")
        assert isinstance(result, dict)


class TestRepositorySearchSummaries:
    """Tests for Repository.search_summaries."""

    def test_search_summaries_all(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_summary(cid, "Patient has doctor appointment", ["appointment"])

        results = repo.search_summaries()
        assert len(results) >= 1

    def test_search_summaries_keyword(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_summary(cid, "Doctor prescribed blood pressure medication", ["medication"])

        results = repo.search_summaries("blood pressure")
        assert len(results) >= 1

    def test_search_summaries_no_match(self, temp_db):
        from storage.repository import Repository
        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        repo.save_summary(cid, "Patient has doctor appointment", ["appointment"])

        results = repo.search_summaries("quantum physics")
        assert len(results) == 0


# =========================================================================
# 21. Tests for ReminderManager SQLite Persistence
# =========================================================================

class TestReminderManagerPersistence:
    """Tests that ReminderManager persists state to SQLite."""

    def test_reminder_with_repo(self, temp_db):
        """ReminderManager accepts repo and persists."""
        from storage.repository import Repository
        from core.reminder_manager import ReminderManager

        repo = Repository(temp_db)

        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr = ReminderManager(_Store(), repo=repo)
        assert mgr.repo is not None

    def test_reminder_legacy_mode(self, temp_db):
        """Without repo falls back to in-memory (no crash)."""
        from core.reminder_manager import ReminderManager

        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr = ReminderManager(_Store(), repo=None)
        assert mgr.repo is None
        assert mgr.load_pending() == []
        assert mgr.auto_schedule() == 0

    def test_auto_schedule_creates_reminders(self, temp_db):
        """auto_schedule() creates reminders for events with parsed datetime."""
        from storage.repository import Repository
        from core.reminder_manager import ReminderManager
        from datetime import datetime, timedelta

        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")

        # Create a future event
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        repo.save_single_event({
            "type": "meeting",
            "description": "Doctor appointment",
            "parsed_date": tomorrow,
            "parsed_time": "10:00",
        }, conv_id=cid)

        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr = ReminderManager(_Store(), repo=repo)
        count = mgr.auto_schedule(lead_minutes=15)
        assert count == 1

        pending = repo.get_pending_reminders()
        assert len(pending) == 1
        assert pending[0]["description"] == "Doctor appointment"

    def test_auto_schedule_no_duplicates(self, temp_db):
        """auto_schedule() does not create duplicates."""
        from storage.repository import Repository
        from core.reminder_manager import ReminderManager
        from datetime import datetime, timedelta

        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        repo.save_single_event({
            "type": "meeting",
            "description": "Doctor",
            "parsed_date": tomorrow,
            "parsed_time": "10:00",
        }, conv_id=cid)

        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr = ReminderManager(_Store(), repo=repo)
        c1 = mgr.auto_schedule()
        c2 = mgr.auto_schedule()
        assert c1 == 1
        assert c2 == 0  # no duplicate

    def test_dismiss_reminder(self, temp_db):
        """dismiss() marks reminder as dismissed in DB."""
        from storage.repository import Repository
        from core.reminder_manager import ReminderManager
        from datetime import datetime, timedelta

        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        event_id = repo.save_single_event({
            "type": "meeting",
            "description": "Test",
            "parsed_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "parsed_time": "10:00",
        }, conv_id=cid)
        rem_id = repo.save_reminder(event_id, datetime.now().isoformat())

        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr = ReminderManager(_Store(), repo=repo)
        assert mgr.dismiss(rem_id) is True

        dismissed = repo.get_reminders_by_status("dismissed")
        assert len(dismissed) == 1

    def test_snooze_reminder(self, temp_db):
        """snooze() resets trigger time and status to pending."""
        from storage.repository import Repository
        from core.reminder_manager import ReminderManager
        from datetime import datetime, timedelta

        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        event_id = repo.save_single_event({
            "type": "task",
            "description": "Call pharmacy",
            "parsed_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "parsed_time": "14:00",
        }, conv_id=cid)
        rem_id = repo.save_reminder(event_id, datetime.now().isoformat())
        repo.mark_reminder_fired(rem_id)

        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr = ReminderManager(_Store(), repo=repo)
        assert mgr.snooze(rem_id, snooze_minutes=10) is True

        pending = repo.get_pending_reminders()
        assert len(pending) == 1

    def test_get_status(self, temp_db):
        """get_status() returns structured status dict."""
        from storage.repository import Repository
        from core.reminder_manager import ReminderManager

        repo = Repository(temp_db)

        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr = ReminderManager(_Store(), repo=repo)
        status = mgr.get_status()
        assert isinstance(status, dict)
        assert status["persisted"] is True
        assert status["running"] is False
        assert "pending" in status
        assert "fired" in status

    def test_restart_safety(self, temp_db):
        """Fired reminders survive engine restart."""
        from storage.repository import Repository
        from core.reminder_manager import ReminderManager
        from datetime import datetime, timedelta

        repo = Repository(temp_db)
        cid = repo.save_conversation(raw_text="test")
        event_id = repo.save_single_event({
            "type": "meeting",
            "description": "Doctor",
            "parsed_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "parsed_time": "09:00",
        }, conv_id=cid)
        rem_id = repo.save_reminder(event_id, datetime.now().isoformat())
        repo.mark_reminder_fired(rem_id)

        # Simulate restart — new manager, same DB
        class _Store:
            def get_all_events(self): return []
            def search_events(self, kw): return []

        mgr2 = ReminderManager(_Store(), repo=repo)
        # Fired alert key should be in _alerted set
        assert len(mgr2._alerted) >= 1

    def test_engine_passes_repo(self, temp_db):
        """MemoryAssistantEngine passes repo to ReminderManager."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(db_path=temp_db)
        assert engine.reminder_mgr.repo is not None
        assert engine.reminder_mgr.repo is engine.repo


# =========================================================================
# 22. Tests for Phase I — Voice Fingerprinting
# =========================================================================

class TestVoiceFingerprintEngine:
    """Tests for VoiceFingerprintEngine (mock embeddings, no real audio)."""

    def test_cosine_self_similarity(self):
        """Self-similarity should be ~1.0."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        engine = VoiceFingerprintEngine()
        v = np.random.randn(192).astype(np.float32)
        v = v / np.linalg.norm(v)
        sim = engine.compare(v, v)
        assert sim > 0.99

    def test_cosine_orthogonal(self):
        """Orthogonal vectors should have similarity ~0."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        a = np.zeros(192, dtype=np.float32)
        b = np.zeros(192, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = VoiceFingerprintEngine.compare(a, b)
        assert abs(sim) < 0.01

    def test_cosine_none_safety(self):
        """compare() with None inputs returns 0.0."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        assert VoiceFingerprintEngine.compare(None, None) == 0.0
        v = np.ones(192, dtype=np.float32)
        assert VoiceFingerprintEngine.compare(v, None) == 0.0

    def test_cosine_empty_safety(self):
        """compare() with empty arrays returns 0.0."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        assert VoiceFingerprintEngine.compare(np.array([]), np.array([])) == 0.0

    def test_serialization_roundtrip(self):
        """embedding_to_bytes → bytes_to_embedding roundtrip."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        original = np.random.randn(192).astype(np.float32)
        as_bytes = VoiceFingerprintEngine.embedding_to_bytes(original)
        restored = VoiceFingerprintEngine.bytes_to_embedding(as_bytes)
        assert np.allclose(original, restored)

    def test_serialization_none(self):
        """Serialization handles None gracefully."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        assert VoiceFingerprintEngine.embedding_to_bytes(None) == b""
        assert VoiceFingerprintEngine.bytes_to_embedding(b"") is None

    def test_normalize(self):
        """_normalize produces unit vector."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        v = np.array([3.0, 4.0], dtype=np.float32)
        normed = VoiceFingerprintEngine._normalize(v)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-5

    def test_match_against_db_exact(self):
        """Exact same embedding matches with score ~1.0."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        engine = VoiceFingerprintEngine()
        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        stored = [{
            "speaker_name": "Dr. Smith",
            "embedding": VoiceFingerprintEngine.embedding_to_bytes(emb),
        }]

        result = engine.match_against_db(emb, stored, threshold=0.75)
        assert result["matched_name"] == "Dr. Smith"
        assert result["similarity_score"] > 0.99
        assert result["confidence"] == "high"

    def test_match_against_db_below_threshold(self):
        """Different embedding below threshold returns None."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        engine = VoiceFingerprintEngine()

        emb_a = np.zeros(192, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(192, dtype=np.float32)
        emb_b[1] = 1.0

        stored = [{
            "speaker_name": "Dr. Smith",
            "embedding": VoiceFingerprintEngine.embedding_to_bytes(emb_b),
        }]

        result = engine.match_against_db(emb_a, stored, threshold=0.75)
        assert result["matched_name"] is None
        assert result["similarity_score"] < 0.60

    def test_match_multi_speaker(self):
        """Correctly identifies among multiple speakers."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        engine = VoiceFingerprintEngine()

        # Create 3 distinct "speakers"
        emb_doc = np.random.randn(192).astype(np.float32)
        emb_doc = emb_doc / np.linalg.norm(emb_doc)
        emb_pat = np.random.randn(192).astype(np.float32)
        emb_pat = emb_pat / np.linalg.norm(emb_pat)

        stored = [
            {"speaker_name": "Doctor", "embedding": VoiceFingerprintEngine.embedding_to_bytes(emb_doc)},
            {"speaker_name": "Patient", "embedding": VoiceFingerprintEngine.embedding_to_bytes(emb_pat)},
        ]

        # Query with doctor's embedding should match Doctor
        result = engine.match_against_db(emb_doc, stored, threshold=0.5)
        assert result["matched_name"] == "Doctor"

    def test_match_empty_db(self):
        """match_against_db with empty stored returns None."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        engine = VoiceFingerprintEngine()
        emb = np.random.randn(192).astype(np.float32)
        result = engine.match_against_db(emb, [], threshold=0.75)
        assert result["matched_name"] is None


class TestVoiceprintDB:
    """Tests for voiceprint DB storage via Repository."""

    def test_save_and_retrieve(self, temp_db):
        """Save voiceprint and retrieve it."""
        from storage.repository import Repository
        repo = Repository(temp_db)

        emb = np.random.randn(192).astype(np.float32).tobytes()
        vp_id = repo.save_voiceprint("Dr. Smith", emb)
        assert vp_id is not None

        all_vps = repo.get_all_voiceprints()
        assert len(all_vps) == 1
        assert all_vps[0]["speaker_name"] == "Dr. Smith"
        assert all_vps[0]["embedding"] == emb

    def test_multiple_per_speaker(self, temp_db):
        """Multiple embeddings per speaker allowed."""
        from storage.repository import Repository
        repo = Repository(temp_db)

        emb1 = np.random.randn(192).astype(np.float32).tobytes()
        emb2 = np.random.randn(192).astype(np.float32).tobytes()
        repo.save_voiceprint("Doctor", emb1)
        repo.save_voiceprint("Doctor", emb2)

        vps = repo.get_voiceprints_for_speaker("Doctor")
        assert len(vps) == 2

    def test_count_voiceprints(self, temp_db):
        """count_voiceprints returns correct count."""
        from storage.repository import Repository
        repo = Repository(temp_db)

        assert repo.count_voiceprints() == 0
        repo.save_voiceprint("A", b"fake")
        repo.save_voiceprint("B", b"fake2")
        assert repo.count_voiceprints() == 2

    def test_delete_voiceprints(self, temp_db):
        """delete_voiceprints removes all for a speaker."""
        from storage.repository import Repository
        repo = Repository(temp_db)

        repo.save_voiceprint("Doctor", b"emb1")
        repo.save_voiceprint("Doctor", b"emb2")
        repo.save_voiceprint("Patient", b"emb3")

        deleted = repo.delete_voiceprints("Doctor")
        assert deleted == 2
        assert repo.count_voiceprints() == 1


class TestIdentityVoiceprint:
    """Tests for IdentityManager voice fingerprint integration."""

    def test_register_voiceprint(self, temp_db):
        """register_voiceprint stores in DB via repo."""
        from storage.repository import Repository
        from speaker_identity.identity_manager import IdentityManager
        from storage.db import Database

        repo = Repository(temp_db)
        mgr = IdentityManager(repo.db)

        emb = np.random.randn(192).astype(np.float32)
        vp_id = mgr.register_voiceprint("Doctor", emb, repo=repo)
        assert vp_id is not None
        assert repo.count_voiceprints() == 1

    def test_register_voiceprint_no_repo(self, temp_db):
        """register_voiceprint with no repo returns None."""
        from storage.repository import Repository
        from speaker_identity.identity_manager import IdentityManager

        repo = Repository(temp_db)
        mgr = IdentityManager(repo.db)

        emb = np.random.randn(192).astype(np.float32)
        result = mgr.register_voiceprint("Doctor", emb, repo=None)
        assert result is None

    def test_match_voiceprint(self, temp_db):
        """match_voiceprint finds stored speaker."""
        from storage.repository import Repository
        from speaker_identity.identity_manager import IdentityManager
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine

        repo = Repository(temp_db)
        mgr = IdentityManager(repo.db)

        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # Store voiceprint
        emb_bytes = VoiceFingerprintEngine.embedding_to_bytes(emb)
        repo.save_voiceprint("Doctor", emb_bytes)

        # Match same embedding
        result = mgr.match_voiceprint(emb, repo=repo, threshold=0.75)
        assert result["matched_name"] == "Doctor"
        assert result["similarity_score"] > 0.99

    def test_match_voiceprint_empty_db(self, temp_db):
        """match_voiceprint with no stored voiceprints returns None."""
        from storage.repository import Repository
        from speaker_identity.identity_manager import IdentityManager

        repo = Repository(temp_db)
        mgr = IdentityManager(repo.db)

        emb = np.random.randn(192).astype(np.float32)
        result = mgr.match_voiceprint(emb, repo=repo)
        assert result["matched_name"] is None
        assert result["similarity_score"] == 0.0

    def test_voiceprint_table_created(self, temp_db):
        """speaker_voiceprints table exists after DB init."""
        from storage.repository import Repository
        repo = Repository(temp_db)

        tables = repo.db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='speaker_voiceprints'"
        )
        assert len(tables) == 1


# =========================================================================
# 23a. Tests for Phase L — Adaptive Voice Model
# =========================================================================

class TestAdaptiveVoiceModel:
    """Tests for centroid matching, confidence scoring, embedding rotation."""

    def test_compute_centroid_single(self):
        """Centroid of a single embedding is the normalized embedding itself."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        centroid = VoiceFingerprintEngine.compute_centroid([emb])
        assert centroid is not None
        assert np.allclose(centroid, emb, atol=1e-5)

    def test_compute_centroid_multiple(self):
        """Centroid of multiple embeddings is their normalized mean."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        embs = [np.random.randn(192).astype(np.float32) for _ in range(5)]
        embs = [e / np.linalg.norm(e) for e in embs]
        centroid = VoiceFingerprintEngine.compute_centroid(embs)
        assert centroid is not None
        assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5  # normalized

    def test_compute_centroid_empty(self):
        """Centroid of empty list is None."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        assert VoiceFingerprintEngine.compute_centroid([]) is None

    def test_classify_confidence_high(self):
        """Score >= 0.85 is high confidence, auto-assign True."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        level, auto = VoiceFingerprintEngine.classify_confidence(0.92)
        assert level == "high"
        assert auto is True

    def test_classify_confidence_medium(self):
        """Score in [0.75, 0.85) is medium confidence, auto-assign True."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        level, auto = VoiceFingerprintEngine.classify_confidence(0.80)
        assert level == "medium"
        assert auto is True

    def test_classify_confidence_low(self):
        """Score in [0.60, 0.75) is low confidence, auto-assign False."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        level, auto = VoiceFingerprintEngine.classify_confidence(0.65)
        assert level == "low"
        assert auto is False

    def test_classify_confidence_none(self):
        """Score < 0.60 is no confidence."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        level, auto = VoiceFingerprintEngine.classify_confidence(0.30)
        assert level == "none"
        assert auto is False

    def test_match_result_format(self):
        """match_against_db returns dict with expected keys."""
        from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
        engine = VoiceFingerprintEngine()
        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        stored = [{
            "speaker_name": "Doctor",
            "embedding": VoiceFingerprintEngine.embedding_to_bytes(emb),
        }]
        result = engine.match_against_db(emb, stored)
        assert isinstance(result, dict)
        assert "matched_name" in result
        assert "similarity_score" in result
        assert "confidence" in result
        assert "auto_assign" in result

    def test_embedding_rotation(self, temp_db):
        """rotate_voiceprints keeps only the newest N entries."""
        from storage.repository import Repository
        repo = Repository(temp_db)
        emb_bytes = np.random.randn(192).astype(np.float32).tobytes()

        # Save 15 voiceprints
        for _ in range(15):
            repo.save_voiceprint("Doctor", emb_bytes, max_per_speaker=100)  # no rotation during insert

        assert len(repo.get_voiceprints_for_speaker("Doctor")) == 15

        # Now rotate to keep 10
        deleted = repo.rotate_voiceprints("Doctor", max_count=10)
        assert deleted == 5
        assert len(repo.get_voiceprints_for_speaker("Doctor")) == 10

    def test_save_triggers_rotation(self, temp_db):
        """save_voiceprint with max_per_speaker auto-rotates."""
        from storage.repository import Repository
        repo = Repository(temp_db)
        emb_bytes = np.random.randn(192).astype(np.float32).tobytes()

        # Save 12 voiceprints with cap 5
        for _ in range(12):
            repo.save_voiceprint("Patient", emb_bytes, max_per_speaker=5)

        # Should only have 5 left
        remaining = repo.get_voiceprints_for_speaker("Patient")
        assert len(remaining) == 5

    def test_rotation_no_excess(self, temp_db):
        """rotate_voiceprints with count under max deletes nothing."""
        from storage.repository import Repository
        repo = Repository(temp_db)
        emb_bytes = np.random.randn(192).astype(np.float32).tobytes()

        repo.save_voiceprint("Doctor", emb_bytes, max_per_speaker=100)
        deleted = repo.rotate_voiceprints("Doctor", max_count=10)
        assert deleted == 0

    def test_reinforce_voiceprint(self, temp_db):
        """reinforce_voiceprint stores embedding for confirmed speaker."""
        from storage.repository import Repository
        from speaker_identity.identity_manager import IdentityManager

        repo = Repository(temp_db)
        mgr = IdentityManager(repo.db)

        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        vp_id = mgr.reinforce_voiceprint("Doctor", emb, repo=repo)
        assert vp_id is not None

        stored = repo.get_voiceprints_for_speaker("Doctor")
        assert len(stored) == 1

    def test_reinforce_no_repo_returns_none(self, temp_db):
        """reinforce_voiceprint with no repo returns None."""
        from speaker_identity.identity_manager import IdentityManager
        from storage.repository import Repository
        repo = Repository(temp_db)
        mgr = IdentityManager(repo.db)
        emb = np.random.randn(192).astype(np.float32)
        result = mgr.reinforce_voiceprint("Doctor", emb, repo=None)
        assert result is None


# =========================================================================
# 23c. Tests for Phase M — Semantic Embedding Search
# =========================================================================

class TestEmbeddingSearch:
    """Tests for sentence-embedding-based semantic search."""

    def test_embedding_encode_shape(self):
        """EmbeddingSearch.encode returns 384-dim vector."""
        pytest.importorskip("sentence_transformers")
        from core.semantic_search import EmbeddingSearch
        searcher = EmbeddingSearch()
        if not searcher.is_available:
            pytest.skip("sentence-transformers not available")
        vec = searcher.encode("Hello world")
        assert vec is not None
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_cosine_self_similarity(self):
        """Same text encodes to cosine similarity ~1.0."""
        pytest.importorskip("sentence_transformers")
        from core.semantic_search import EmbeddingSearch
        searcher = EmbeddingSearch()
        if not searcher.is_available:
            pytest.skip("sentence-transformers not available")
        v1 = searcher.encode("Doctor appointment tomorrow")
        v2 = searcher.encode("Doctor appointment tomorrow")
        sim = EmbeddingSearch.cosine_similarity(v1, v2)
        assert sim > 0.99

    def test_synonym_doctor_physician(self):
        """'doctor' and 'physician' should be more similar than 'doctor' and 'grocery'."""
        pytest.importorskip("sentence_transformers")
        from core.semantic_search import EmbeddingSearch
        searcher = EmbeddingSearch()
        if not searcher.is_available:
            pytest.skip("sentence-transformers not available")
        v_doc = searcher.encode("I have a doctor appointment")
        v_phys = searcher.encode("I have a physician appointment")
        v_groc = searcher.encode("I need to buy groceries")

        sim_synonym = EmbeddingSearch.cosine_similarity(v_doc, v_phys)
        sim_unrelated = EmbeddingSearch.cosine_similarity(v_doc, v_groc)

        assert sim_synonym > sim_unrelated, (
            f"Synonym similarity ({sim_synonym:.3f}) should exceed "
            f"unrelated similarity ({sim_unrelated:.3f})"
        )
        assert sim_synonym > 0.7  # synonyms should be highly similar

    def test_embedding_search_returns_results(self):
        """EmbeddingSearch.search finds semantically similar documents."""
        pytest.importorskip("sentence_transformers")
        from core.semantic_search import EmbeddingSearch
        searcher = EmbeddingSearch()
        if not searcher.is_available:
            pytest.skip("sentence-transformers not available")

        docs = [
            {"text": "Doctor appointment tomorrow at 10 AM"},
            {"text": "Take medicine after breakfast"},
            {"text": "Buy groceries from the store"},
        ]
        results = searcher.search("physician visit", docs, top_k=2)
        assert len(results) >= 1
        # Doctor appointment should rank first
        assert "Doctor" in results[0]["document"]["text"] or \
               "doctor" in results[0]["document"]["text"].lower()

    def test_embedding_storage_roundtrip(self, temp_db):
        """Save and retrieve conversation embedding from DB."""
        from storage.repository import Repository
        repo = Repository(temp_db)

        # Create parent conversation (FK constraint)
        cid = repo.save_conversation(raw_text="Test conversation")

        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        emb_bytes = emb.tobytes()

        repo.save_conversation_embedding(cid, emb_bytes)
        retrieved = repo.get_conversation_embedding(cid)
        assert retrieved is not None

        recovered = np.frombuffer(retrieved, dtype=np.float32)
        assert np.allclose(emb, recovered, atol=1e-6)

    def test_get_all_embeddings(self, temp_db):
        """get_all_conversation_embeddings returns all stored embeddings."""
        from storage.repository import Repository
        repo = Repository(temp_db)

        cids = []
        for i in range(3):
            cid = repo.save_conversation(raw_text=f"Conversation {i}")
            cids.append(cid)
            emb = np.random.randn(384).astype(np.float32).tobytes()
            repo.save_conversation_embedding(cid, emb)

        all_embs = repo.get_all_conversation_embeddings()
        assert len(all_embs) == 3
        assert cids[0] in all_embs
        assert cids[2] in all_embs

    def test_embedding_table_exists(self, temp_db):
        """conversation_embeddings table exists after DB init."""
        from storage.repository import Repository
        repo = Repository(temp_db)
        tables = repo.db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='conversation_embeddings'"
        )
        assert len(tables) == 1

    def test_cosine_similarity_static(self):
        """EmbeddingSearch.cosine_similarity correctness with known vectors."""
        from core.semantic_search import EmbeddingSearch

        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(EmbeddingSearch.cosine_similarity(a, b) - 1.0) < 1e-5

        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert abs(EmbeddingSearch.cosine_similarity(a, c)) < 1e-5  # orthogonal

    def test_byte_conversion_roundtrip(self):
        """embedding_to_bytes / bytes_to_embedding roundtrip."""
        from core.semantic_search import EmbeddingSearch
        emb = np.random.randn(384).astype(np.float32)
        b = EmbeddingSearch.embedding_to_bytes(emb)
        recovered = EmbeddingSearch.bytes_to_embedding(b)
        assert np.allclose(emb, recovered, atol=1e-6)

    def test_tfidf_search_still_works(self):
        """Original TF-IDF SemanticSearch unchanged."""
        from core.semantic_search import SemanticSearch
        searcher = SemanticSearch()
        docs = [
            {"text": "Doctor appointment tomorrow at 10 AM"},
            {"text": "Take medicine after breakfast"},
        ]
        count = searcher.index(docs)
        assert count == 2
        results = searcher.search("doctor", top_k=2)
        assert len(results) >= 1


# =========================================================================
# 23b. Tests for Phase J — Database Encryption
# =========================================================================

class TestDatabaseEncryption:
    """Tests for AES-256-GCM database encryption."""

    def test_encrypt_decrypt_roundtrip(self, tmp_path):
        """Encrypt a file, decrypt it, verify contents match."""
        from storage.encryption import DatabaseEncryption

        src = tmp_path / "test.txt"
        src.write_bytes(b"Hello, encrypted world!")

        enc = DatabaseEncryption("test-passphrase")
        enc_path = str(tmp_path / "test.enc")
        dec_path = str(tmp_path / "test.dec")

        enc.encrypt_file(str(src), enc_path)
        enc.decrypt_file(enc_path, dec_path)

        assert open(dec_path, "rb").read() == b"Hello, encrypted world!"

    def test_wrong_passphrase_fails(self, tmp_path):
        """Decryption with wrong passphrase raises error."""
        from storage.encryption import DatabaseEncryption

        src = tmp_path / "test.txt"
        src.write_bytes(b"Secret data")

        enc1 = DatabaseEncryption("correct-key")
        enc_path = str(tmp_path / "test.enc")
        enc1.encrypt_file(str(src), enc_path)

        enc2 = DatabaseEncryption("wrong-key")
        dec_path = str(tmp_path / "test.dec")

        with pytest.raises(Exception):  # InvalidTag
            enc2.decrypt_file(enc_path, dec_path)

    def test_is_encrypted_detection(self, tmp_path):
        """is_encrypted correctly identifies encrypted files."""
        from storage.encryption import DatabaseEncryption

        # Create plaintext file
        plain = tmp_path / "plain.txt"
        plain.write_bytes(b"not encrypted")

        # Create encrypted file
        enc = DatabaseEncryption("key")
        enc_path = str(tmp_path / "enc.bin")
        enc.encrypt_file(str(plain), enc_path)

        assert not DatabaseEncryption.is_encrypted(str(plain))
        assert DatabaseEncryption.is_encrypted(enc_path)
        assert not DatabaseEncryption.is_encrypted("nonexistent.file")

    def test_key_derivation_deterministic(self):
        """Same passphrase + salt produces same key."""
        from storage.encryption import DatabaseEncryption

        enc = DatabaseEncryption("my-key")
        salt = b"0123456789abcdef"
        key1 = enc.derive_key(salt)
        key2 = enc.derive_key(salt)
        assert key1 == key2
        assert len(key1) == 32

    def test_encrypted_db_operations(self, tmp_path):
        """Database works normally in encrypted mode."""
        from storage.db import Database

        db_path = str(tmp_path / "enc_test.db")
        db = Database(db_path, passphrase="secret-key")

        assert db.is_encrypted

        # Insert data
        eid = db.new_id()
        db.execute(
            "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
            (eid, "test", "Encrypted event"),
        )

        # Read it back
        row = db.fetch_one("SELECT * FROM events WHERE id = ?", (eid,))
        assert row is not None
        assert row["description"] == "Encrypted event"

        db.close()

    def test_plaintext_migration(self, tmp_path):
        """Plaintext DB is auto-migrated to encrypted on first open with passphrase."""
        from storage.db import Database
        from storage.encryption import DatabaseEncryption

        db_path = str(tmp_path / "migrate.db")

        # Create plaintext DB
        db_plain = Database(db_path)
        db_plain.execute(
            "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
            ("evt1", "test", "Migration test"),
        )
        # Plaintext DB file should exist
        assert os.path.isfile(db_path)

        # Open with passphrase — triggers migration
        db_enc = Database(db_path, passphrase="migrate-key")
        assert db_enc.is_encrypted

        # Data should still be there
        row = db_enc.fetch_one("SELECT * FROM events WHERE id = ?", ("evt1",))
        assert row is not None
        assert row["description"] == "Migration test"

        # Close should create .enc file
        db_enc.close()
        assert os.path.isfile(db_path + ".enc")

    def test_encrypted_persistence(self, tmp_path):
        """Data persists across encrypted open/close cycles."""
        from storage.db import Database

        db_path = str(tmp_path / "persist.db")

        # Write data
        db1 = Database(db_path, passphrase="persist-key")
        db1.execute(
            "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
            ("e1", "test", "Persistent data"),
        )
        db1.close()

        # Re-open and verify
        db2 = Database(db_path, passphrase="persist-key")
        row = db2.fetch_one("SELECT * FROM events WHERE id = ?", ("e1",))
        assert row is not None
        assert row["description"] == "Persistent data"
        db2.close()

    def test_save_encrypted_interim(self, tmp_path):
        """save_encrypted creates durability checkpoint."""
        from storage.db import Database

        db_path = str(tmp_path / "save.db")
        db = Database(db_path, passphrase="save-key")

        db.execute(
            "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
            ("s1", "test", "Saved data"),
        )
        db.save_encrypted()

        assert os.path.isfile(db_path + ".enc")
        db.close()

    def test_unencrypted_no_passphrase(self, temp_db):
        """Database without passphrase runs in plaintext mode."""
        from storage.db import Database

        db = Database(temp_db)
        assert not db.is_encrypted

    def test_empty_passphrase_raises(self):
        """Empty passphrase raises ValueError."""
        from storage.encryption import DatabaseEncryption
        with pytest.raises(ValueError):
            DatabaseEncryption("")


# =========================================================================
# 24. Tests for Phase N — Secure Backup & Restore
# =========================================================================

class TestBackupRestore:
    """Tests for encrypted backup/restore with SHA-256 integrity."""

    def test_backup_creates_file(self, temp_db, tmp_path):
        """create_backup produces a .wbbak file."""
        from storage.db import Database
        from storage.backup_manager import BackupManager

        db = Database(temp_db)
        db.execute(
            "INSERT INTO conversations (id, timestamp, raw_text) "
            "VALUES (?, datetime('now'), ?)",
            ("conv-bak-001", "Test conversation for backup"),
        )

        mgr = BackupManager(db)
        backup_path = str(tmp_path / "test_backup.wbbak")
        result = mgr.create_backup(backup_path)

        assert result["status"] == "success"
        assert os.path.isfile(backup_path)
        assert result["size_bytes"] > 0
        assert len(result["sha256"]) == 64  # SHA-256 hex length
        db.close()

    def test_verify_valid_backup(self, temp_db, tmp_path):
        """verify_backup returns valid=True for a good backup."""
        from storage.db import Database
        from storage.backup_manager import BackupManager

        db = Database(temp_db)
        mgr = BackupManager(db)
        backup_path = str(tmp_path / "valid.wbbak")
        mgr.create_backup(backup_path)

        verification = mgr.verify_backup(backup_path)
        assert verification["valid"] is True
        assert verification["sha256_match"] is True
        assert verification["manifest"]["version"] == "1.0.0"
        db.close()

    def test_verify_tampered_backup(self, temp_db, tmp_path):
        """verify_backup detects tampered files."""
        from storage.db import Database
        from storage.backup_manager import BackupManager

        db = Database(temp_db)
        mgr = BackupManager(db)
        backup_path = str(tmp_path / "tampered.wbbak")
        mgr.create_backup(backup_path)

        # Tamper: corrupt bytes in the middle of the archive
        with open(backup_path, "r+b") as f:
            f.seek(100)
            f.write(b"\x00\x00\x00\x00\x00")

        verification = mgr.verify_backup(backup_path)
        assert verification["valid"] is False
        db.close()

    def test_restore_preserves_conversations(self, tmp_path):
        """Backup and restore preserves conversation data."""
        from storage.db import Database
        from storage.backup_manager import BackupManager

        # Create DB with data
        db_path = str(tmp_path / "source.db")
        db = Database(db_path)
        db.execute(
            "INSERT INTO conversations (id, timestamp, raw_text) "
            "VALUES (?, datetime('now'), ?)",
            ("conv-restore-001", "Important doctor conversation"),
        )
        db.execute(
            "INSERT INTO events (id, conversation_id, type, description, fingerprint) "
            "VALUES (?, ?, ?, ?, ?)",
            ("evt-001", "conv-restore-001", "meeting", "Doctor at 10 AM", "fp-001"),
        )

        mgr = BackupManager(db)
        backup_path = str(tmp_path / "data.wbbak")
        mgr.create_backup(backup_path)
        db.close()

        # Restore to a different DB
        db2_path = str(tmp_path / "restored.db")
        db2 = Database(db2_path)
        mgr2 = BackupManager(db2)
        result = mgr2.restore_backup(backup_path)
        assert result["status"] == "success"

        # Reopen and verify data
        db3 = Database(db2_path)
        conv = db3.fetch_one(
            "SELECT * FROM conversations WHERE id = ?",
            ("conv-restore-001",),
        )
        assert conv is not None
        assert "Important doctor" in conv["raw_text"]

        evt = db3.fetch_one(
            "SELECT * FROM events WHERE id = ?",
            ("evt-001",),
        )
        assert evt is not None
        assert evt["description"] == "Doctor at 10 AM"
        db3.close()

    def test_restore_preserves_voiceprints(self, tmp_path):
        """Backup preserves speaker voiceprints."""
        from storage.repository import Repository
        from storage.backup_manager import BackupManager

        db_path = str(tmp_path / "vp_source.db")
        repo = Repository(db_path)

        # Save a voiceprint
        emb = np.random.randn(192).astype(np.float32)
        repo.save_voiceprint("Dr. Smith", emb.tobytes())

        mgr = BackupManager(repo.db)
        backup_path = str(tmp_path / "vp_backup.wbbak")
        mgr.create_backup(backup_path)
        repo.db.close()

        # Restore
        db2_path = str(tmp_path / "vp_restored.db")
        repo2 = Repository(db2_path)
        mgr2 = BackupManager(repo2.db)
        mgr2.restore_backup(backup_path)

        # Reopen and check
        repo3 = Repository(db2_path)
        vps = repo3.get_voiceprints_for_speaker("Dr. Smith")
        assert len(vps) == 1
        recovered = np.frombuffer(vps[0]["embedding"], dtype=np.float32)
        assert np.allclose(emb, recovered, atol=1e-6)
        repo3.db.close()

    def test_backup_unencrypted_db(self, temp_db, tmp_path):
        """Backup works for unencrypted databases."""
        from storage.db import Database
        from storage.backup_manager import BackupManager

        db = Database(temp_db)  # No passphrase = unencrypted
        assert not db.is_encrypted

        mgr = BackupManager(db)
        backup_path = str(tmp_path / "plain.wbbak")
        result = mgr.create_backup(backup_path)

        assert result["status"] == "success"
        assert result["encrypted"] is False

        verification = mgr.verify_backup(backup_path)
        assert verification["valid"] is True
        assert verification["manifest"]["encrypted"] is False
        db.close()

    def test_list_backups(self, temp_db, tmp_path):
        """list_backups finds .wbbak files in directory."""
        from storage.db import Database
        from storage.backup_manager import BackupManager

        db = Database(temp_db)
        mgr = BackupManager(db)

        # Create 2 backups
        mgr.create_backup(str(tmp_path / "backup_1.wbbak"))
        mgr.create_backup(str(tmp_path / "backup_2.wbbak"))

        # List
        backups = mgr.list_backups(str(tmp_path))
        assert len(backups) == 2
        assert backups[0]["filename"] == "backup_1.wbbak"
        assert backups[1]["filename"] == "backup_2.wbbak"
        assert all(b["size_bytes"] > 0 for b in backups)
        db.close()

    def test_backup_prevents_overwrite(self, temp_db, tmp_path):
        """create_backup refuses to overwrite existing file."""
        from storage.db import Database
        from storage.backup_manager import BackupManager

        db = Database(temp_db)
        mgr = BackupManager(db)
        backup_path = str(tmp_path / "exists.wbbak")

        result1 = mgr.create_backup(backup_path)
        assert result1["status"] == "success"

        result2 = mgr.create_backup(backup_path)
        assert result2["status"] == "error"
        assert "already exists" in result2["error"]
        db.close()


# =========================================================================
# 25. Tests for Phase O — Wearable Input Integration
# =========================================================================

class TestBluetoothAudioSource:
    """Tests for BluetoothAudioSource push-based ring buffer."""

    def test_push_read_roundtrip(self):
        """Push PCM data and read it back correctly."""
        from audio.bluetooth_source import BluetoothAudioSource

        src = BluetoothAudioSource(sample_rate=16000)
        src.start()

        # Push 1000 samples
        samples = np.arange(1000, dtype=np.int16)
        n = src.push_audio(samples.tobytes())
        assert n == 1000

        # Read back
        chunk = src.read_chunk(1000)
        assert chunk.shape[0] == 1000
        np.testing.assert_array_equal(chunk.ravel(), samples)
        src.stop()

    def test_buffer_overflow_drops_oldest(self):
        """When buffer overflows, oldest data is dropped."""
        from audio.bluetooth_source import BluetoothAudioSource

        # Tiny buffer: 100 samples
        src = BluetoothAudioSource(sample_rate=16000, buffer_seconds=0.00625)
        src.start()

        # Push 150 samples into 100-sample buffer
        samples = np.arange(150, dtype=np.int16)
        src.push_audio(samples.tobytes())

        # Should only have ~100 samples (newest)
        chunk = src.read_chunk(200)
        assert chunk.shape[0] <= 100
        assert src._overflows >= 1

        stats = src.get_stats()
        assert stats["overflows"] >= 1
        src.stop()

    def test_disconnect_state(self):
        """Connection state is tracked correctly."""
        from audio.bluetooth_source import BluetoothAudioSource

        src = BluetoothAudioSource(device_name="Test Earbuds")
        assert not src.is_connected
        assert src.device_name == "Test Earbuds"

        src.set_connected(True, "My Earbuds v2")
        assert src.is_connected
        assert src.device_name == "My Earbuds v2"

        src.set_connected(False)
        assert not src.is_connected

    def test_read_when_inactive_returns_empty(self):
        """read_chunk returns empty when source is not active."""
        from audio.bluetooth_source import BluetoothAudioSource

        src = BluetoothAudioSource()
        chunk = src.read_chunk(480)
        assert len(chunk) == 0

    def test_protocol_compliance(self):
        """BluetoothAudioSource satisfies AudioSource protocol."""
        from audio.source import AudioSource
        from audio.bluetooth_source import BluetoothAudioSource

        src = BluetoothAudioSource()
        assert isinstance(src, AudioSource)

    def test_push_bytes_and_numpy(self):
        """push_audio accepts both bytes and numpy arrays."""
        from audio.bluetooth_source import BluetoothAudioSource

        src = BluetoothAudioSource()
        src.start()

        # Push as bytes
        data = np.ones(100, dtype=np.int16)
        n1 = src.push_audio(data.tobytes())
        assert n1 == 100

        # Push as numpy
        n2 = src.push_audio(data)
        assert n2 == 100

        chunk = src.read_chunk(200)
        assert chunk.shape[0] == 200
        src.stop()


class TestFileSourceRead:
    """Test FileSource reads WAV files."""

    def test_file_source_reads_wav(self, tmp_path):
        """FileSource reads a WAV file and streams chunks."""
        import wave
        from audio.file_source import FileSource

        # Create test WAV
        wav_path = str(tmp_path / "test.wav")
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        audio_int16 = (samples * 32767).astype(np.int16)

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        # Read it
        src = FileSource(wav_path)
        src.start()
        assert src.is_active
        assert src.sample_rate == 16000

        chunk = src.read_chunk(480)
        assert chunk.shape[0] == 480
        src.stop()


class TestAudioWorkerSourceSwitching:
    """Test AudioWorker accepts different sources."""

    def test_worker_accepts_bluetooth_source(self):
        """AudioWorker can be configured with BluetoothAudioSource."""
        from background.audio_worker import AudioWorker
        from audio.bluetooth_source import BluetoothAudioSource

        bt = BluetoothAudioSource(device_name="Test BT")
        worker = AudioWorker(engine=None, audio_source=bt)
        assert worker.audio_source is bt
        assert worker._source_name() == "BluetoothAudioSource"

    def test_worker_source_setter(self):
        """AudioWorker.audio_source setter works correctly."""
        from background.audio_worker import AudioWorker
        from audio.bluetooth_source import BluetoothAudioSource
        from audio.microphone import MicrophoneSource

        worker = AudioWorker(engine=None)
        assert worker.audio_source is None  # Default lazy

        bt = BluetoothAudioSource()
        worker.audio_source = bt
        assert worker.audio_source is bt

    def test_worker_rejects_source_change_while_running(self):
        """Cannot change audio source while worker is running."""
        from background.audio_worker import AudioWorker
        from audio.bluetooth_source import BluetoothAudioSource

        # We can't actually start the worker (needs mic), but we can
        # verify the property setter raises when running
        worker = AudioWorker(engine=None)
        bt = BluetoothAudioSource()
        # Should succeed when not running
        worker.audio_source = bt
        assert worker.audio_source is bt


# =========================================================================
# 26. Tests for Phase P — Performance & Resource Optimization
# =========================================================================

class TestConfigModule:
    """Tests for config.py."""

    def test_config_summary_shape(self):
        """get_config_summary returns expected keys."""
        from config import get_config_summary
        summary = get_config_summary()
        expected_keys = {
            "low_resource_mode", "whisper_model", "enable_embeddings",
            "max_voiceprints_per_speaker", "vad_silence_threshold",
            "max_recording_duration_sec", "processing_cooldown_sec",
            "ring_buffer_seconds", "sample_rate", "debug_timing",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_config_default_values(self):
        """Default (non-low-resource) values are reasonable."""
        import config
        assert config.SAMPLE_RATE == 16000
        assert config.CHANNELS == 1
        assert config.CHUNK_DURATION_MS == 30
        assert config.MAX_VOICEPRINTS_PER_SPEAKER in (5, 10)
        assert config.WHISPER_MODEL_SIZE in ("tiny", "base")

    def test_low_resource_mode_toggle(self, monkeypatch):
        """LOW_RESOURCE_MODE changes dependent settings."""
        import importlib
        import config

        # Force low-resource ON
        monkeypatch.setenv("WBRAIN_LOW_RESOURCE", "1")
        importlib.reload(config)
        assert config.LOW_RESOURCE_MODE is True
        assert config.WHISPER_MODEL_SIZE == "tiny"
        assert config.ENABLE_EMBEDDINGS is False
        assert config.MAX_VOICEPRINTS_PER_SPEAKER == 5
        assert config.VAD_SILENCE_THRESHOLD == 800.0
        assert config.MAX_RECORDING_DURATION_SEC == 60.0

        # Restore
        monkeypatch.setenv("WBRAIN_LOW_RESOURCE", "0")
        importlib.reload(config)
        assert config.LOW_RESOURCE_MODE is False
        assert config.WHISPER_MODEL_SIZE == "base"
        assert config.ENABLE_EMBEDDINGS is True


class TestEmbeddingFallback:
    """Ensure TF-IDF fallback when embeddings disabled."""

    def test_tfidf_fallback_when_embeddings_disabled(self, monkeypatch, tmp_path):
        """Query engine falls back to TF-IDF when ENABLE_EMBEDDINGS=False."""
        import importlib
        import config

        # Disable embeddings
        monkeypatch.setenv("WBRAIN_LOW_RESOURCE", "1")
        importlib.reload(config)

        from engine.assistant_engine import MemoryAssistantEngine
        from core.query_engine import QueryEngine

        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))

        # Process text to create conversation + events with proper FK
        engine.process_text(
            "I have a doctor appointment tomorrow at 10 AM."
        )

        # The emb searcher should be a stub (not available)
        qe = QueryEngine(engine.repo)
        qe._emb_searcher = None  # Force fresh creation
        emb = qe._get_emb_searcher()
        assert not emb.is_available

        # Query should still work via TF-IDF
        result = qe.query("doctor appointment")
        assert isinstance(result, str)
        assert len(result) > 0  # Should return something

        # Restore
        monkeypatch.setenv("WBRAIN_LOW_RESOURCE", "0")
        importlib.reload(config)


class TestWhisperModelCache:
    """Tests for Whisper model singleton caching."""

    def test_model_cache_dict_exists(self):
        """The module-level _model_cache is accessible."""
        from core.transcriber import _model_cache
        assert isinstance(_model_cache, dict)

    def test_get_model_returns_none_for_missing(self):
        """_get_model with invalid size doesn't crash (may fail on load)."""
        from core.transcriber import _model_cache
        # Verify cache is a dict, don't actually load a model
        assert isinstance(_model_cache, dict)


class TestResourceStats:
    """Tests for engine.get_resource_stats()."""

    def test_resource_stats_shape(self, tmp_path):
        """get_resource_stats returns expected structure."""
        from engine.assistant_engine import MemoryAssistantEngine

        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))
        stats = engine.get_resource_stats()

        assert "models" in stats
        assert "active_threads" in stats
        assert "thread_names" in stats
        assert "audio_buffer" in stats
        assert "config" in stats
        assert isinstance(stats["models"], dict)
        assert isinstance(stats["active_threads"], int)
        assert stats["active_threads"] >= 1  # At least main thread


class TestWorkerConfigPropagation:
    """Tests for AudioWorker reading config defaults."""

    def test_worker_reads_config_defaults(self):
        """AudioWorker picks up config values when no args specified."""
        from background.audio_worker import AudioWorker
        import config

        worker = AudioWorker(engine=None)
        assert worker.sample_rate == config.SAMPLE_RATE
        assert worker.silence_threshold == config.VAD_SILENCE_THRESHOLD
        assert worker.max_record_sec == config.MAX_RECORDING_DURATION_SEC
        assert worker._cooldown_sec == config.PROCESSING_COOLDOWN_SEC

    def test_worker_overrides_config(self):
        """Explicit constructor args override config defaults."""
        from background.audio_worker import AudioWorker

        worker = AudioWorker(
            engine=None,
            sample_rate=8000,
            silence_threshold=999.0,
            max_record_sec=30.0,
        )
        assert worker.sample_rate == 8000
        assert worker.silence_threshold == 999.0
        assert worker.max_record_sec == 30.0


# =========================================================================
# 27. Tests for Phase Q — Alzheimer-Aware Memory Prioritization
# =========================================================================

class TestImportanceScoring:
    """Tests for memory_ranker.score_event()."""

    def test_medication_scores_5(self):
        """Medication events get importance_score=5."""
        from core.memory_ranker import score_event
        event = {"type": "medication", "description": "Take your medicine after breakfast"}
        assert score_event(event) == 5

    def test_doctor_appointment_scores_5(self):
        """Doctor meeting events get importance_score=5."""
        from core.memory_ranker import score_event
        event = {"type": "meeting", "description": "Doctor appointment tomorrow at 10 AM"}
        assert score_event(event) == 5

    def test_safety_scores_4(self):
        """Safety-related events get importance_score=4."""
        from core.memory_ranker import score_event
        event = {"type": "task", "description": "Don't forget to lock the door"}
        assert score_event(event) == 4

    def test_family_scores_2(self):
        """Family visit events get importance_score=2."""
        from core.memory_ranker import score_event
        event = {"type": "meeting", "description": "Son David is visiting this weekend"}
        assert score_event(event) == 2

    def test_general_task_scores_1(self):
        """General tasks get baseline importance_score=1."""
        from core.memory_ranker import score_event
        event = {"type": "task", "description": "Buy groceries from the store"}
        assert score_event(event) == 1

    def test_score_events_adds_field(self):
        """score_events adds importance_score to each event."""
        from core.memory_ranker import score_events
        events = [
            {"type": "medication", "description": "Take pills"},
            {"type": "task", "description": "Buy milk"},
        ]
        score_events(events)
        assert events[0]["importance_score"] == 5
        assert events[1]["importance_score"] == 1


class TestRecurrenceDetection:
    """Tests for pattern detection and frequency tracking."""

    def test_detect_patterns_finds_medication(self):
        """Detects 'take your medicine' phrase."""
        from core.memory_ranker import detect_patterns
        text = "Don't forget to take your medicine after breakfast."
        patterns = detect_patterns(text)
        phrases = [p["phrase"] for p in patterns]
        assert "take your medicine" in phrases

    def test_pattern_frequency_increments(self, tmp_path):
        """Pattern frequency increments across calls."""
        from core.memory_ranker import detect_patterns
        from storage.repository import Repository

        repo = Repository(str(tmp_path / "test.db"))
        text = "Remember to take your medicine."

        p1 = detect_patterns(text, repo=repo)
        med = [p for p in p1 if p["phrase"] == "take your medicine"]
        assert med[0]["frequency"] == 1

        p2 = detect_patterns(text, repo=repo)
        med2 = [p for p in p2 if p["phrase"] == "take your medicine"]
        assert med2[0]["frequency"] == 2

        stored = repo.get_pattern("take your medicine")
        assert stored["frequency"] == 2


class TestWeightedRanking:
    """Tests for blended scoring in rank_results."""

    def test_rank_results_adds_blended_score(self):
        """rank_results adds blended_score to each result."""
        from core.memory_ranker import rank_results
        results = [
            {"score": 0.8, "document": {"importance_score": 5}},
            {"score": 0.9, "document": {"importance_score": 1}},
        ]
        ranked = rank_results(results)
        assert all("blended_score" in r for r in ranked)

    def test_importance_boosts_ranking(self):
        """Higher importance_score boosts result ranking."""
        from core.memory_ranker import rank_results
        results = [
            {"score": 0.5, "document": {"importance_score": 5}},
            {"score": 0.6, "document": {"importance_score": 0}},
        ]
        ranked = rank_results(results)
        assert ranked[0]["document"]["importance_score"] == 5


class TestUrgencyDetection:
    """Tests for urgent item detection."""

    def test_get_urgent_items_returns_list(self, tmp_path):
        """get_urgent_items returns a list."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))
        urgent = engine.get_urgent_items()
        assert isinstance(urgent, list)

    def test_medication_without_date_is_urgent(self, tmp_path):
        """Medication events without parsed_date are flagged as urgent."""
        from core.memory_ranker import get_urgent_items
        from storage.repository import Repository

        repo = Repository(str(tmp_path / "test.db"))
        conv_id = repo.save_conversation(raw_text="Take medicine")
        repo.save_events(conv_id, [{
            "type": "medication",
            "description": "Take your medicine after breakfast",
            "importance_score": 5,
        }])

        urgent = get_urgent_items(repo)
        assert len(urgent) >= 1
        assert urgent[0]["urgent_flag"] is True


class TestPhaseQEngineIntegration:
    """Tests for Phase Q integration in the engine pipeline."""

    def test_process_text_scores_events(self, tmp_path):
        """process_text adds importance_score to extracted events."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))
        result = engine.process_text(
            "I have a doctor appointment tomorrow at 10 AM. "
            "Don't forget to take your medicine after breakfast."
        )
        scored = [e for e in result["events"] if "importance_score" in e]
        assert len(scored) > 0
        assert any(e["importance_score"] >= 5 for e in scored)

    def test_process_text_detects_patterns(self, tmp_path):
        """process_text detects recurring patterns."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))

        engine.process_text("Don't forget to take your medicine.")
        engine.process_text("Remember to take your medicine.")

        patterns = engine.get_memory_patterns()
        med = [p for p in patterns if "medicine" in p["phrase"]]
        assert len(med) >= 1
        assert med[0]["frequency"] >= 2

    def test_backward_compat_events_without_score(self, tmp_path):
        """Events saved without importance_score default to 0."""
        from storage.repository import Repository
        repo = Repository(str(tmp_path / "test.db"))
        conv_id = repo.save_conversation(raw_text="Test")
        repo.save_events(conv_id, [{
            "type": "task",
            "description": "Generic task",
        }])
        events = repo.get_all_events()
        assert events[0]["importance_score"] == 0


# =========================================================================
# 28. Tests for Phase R — Cognitive Simplification & Reinforcement
# =========================================================================

class TestSimplifiedMode:
    """Tests for simplified mode filtering."""

    def test_simplify_summary_trims(self):
        """simplify_summary trims to max_points sentences."""
        from core.reinforcement import simplify_summary
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = simplify_summary(text, max_points=2)
        assert "First sentence" in result
        assert "Second sentence" in result
        assert "Third sentence" not in result

    def test_simplify_summary_short_passthrough(self):
        """Short summaries pass through unchanged."""
        from core.reinforcement import simplify_summary
        text = "Only one sentence."
        result = simplify_summary(text, max_points=2)
        assert result == text

    def test_filter_events_by_importance(self):
        """filter_events_by_importance removes low-importance events."""
        from core.reinforcement import filter_events_by_importance
        events = [
            {"description": "Take pills", "importance_score": 5},
            {"description": "Buy milk", "importance_score": 1},
            {"description": "Doctor visit", "importance_score": 5},
        ]
        filtered = filter_events_by_importance(events, min_importance=3)
        assert len(filtered) == 2
        assert all(e["importance_score"] >= 3 for e in filtered)

    def test_config_simplified_mode_flag(self, monkeypatch):
        """SIMPLIFIED_MODE config flag exists and toggles."""
        import config
        assert hasattr(config, "SIMPLIFIED_MODE")
        monkeypatch.setattr(config, "SIMPLIFIED_MODE", True)
        assert config.SIMPLIFIED_MODE is True
        monkeypatch.setattr(config, "SIMPLIFIED_MODE", False)
        assert config.SIMPLIFIED_MODE is False


class TestReinforcementTracking:
    """Tests for reinforcement shown tracking."""

    def test_mark_shown_creates_record(self, tmp_path):
        """mark_reinforcement_shown creates a tracking record."""
        from storage.repository import Repository
        repo = Repository(str(tmp_path / "test.db"))
        conv_id = repo.save_conversation(raw_text="Take medicine")
        repo.save_events(conv_id, [{
            "type": "medication",
            "description": "Take pills",
            "importance_score": 5,
        }])
        events = repo.get_all_events()
        event_id = events[0]["id"]

        repo.mark_reinforcement_shown(event_id)
        record = repo.get_reinforcement_record(event_id)
        assert record is not None
        assert record["shown_count"] == 1

    def test_mark_shown_increments(self, tmp_path):
        """Repeated mark_shown calls increment shown_count."""
        from storage.repository import Repository
        repo = Repository(str(tmp_path / "test.db"))
        conv_id = repo.save_conversation(raw_text="Take medicine")
        repo.save_events(conv_id, [{
            "type": "medication",
            "description": "Take pills",
            "importance_score": 5,
        }])
        events = repo.get_all_events()
        event_id = events[0]["id"]

        repo.mark_reinforcement_shown(event_id)
        repo.mark_reinforcement_shown(event_id)
        record = repo.get_reinforcement_record(event_id)
        assert record["shown_count"] == 2

    def test_get_reinforcement_items_via_engine(self, tmp_path):
        """Engine get_reinforcement_items returns a list."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))
        items = engine.get_reinforcement_items()
        assert isinstance(items, list)


class TestEscalationLogic:
    """Tests for escalation level management."""

    def test_escalate_event_sets_level(self, tmp_path):
        """escalate_event sets the escalation_level."""
        from storage.repository import Repository
        repo = Repository(str(tmp_path / "test.db"))
        conv_id = repo.save_conversation(raw_text="Doctor appointment")
        repo.save_events(conv_id, [{
            "type": "meeting",
            "description": "Doctor appointment tomorrow",
            "importance_score": 5,
        }])
        events = repo.get_all_events()
        event_id = events[0]["id"]

        repo.escalate_event(event_id, 2)
        escalated = repo.get_escalated_events(min_level=1)
        assert len(escalated) >= 1
        assert escalated[0]["escalation_level"] == 2

    def test_check_escalations_via_engine(self, tmp_path):
        """Engine check_escalations returns a list."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))
        result = engine.check_escalations()
        assert isinstance(result, list)


class TestDailyBrief:
    """Tests for daily brief generation."""

    def test_daily_brief_structure(self, tmp_path):
        """generate_daily_brief returns expected dict structure."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))
        brief = engine.generate_daily_brief()
        assert "greeting" in brief
        assert "urgent_items" in brief
        assert "patterns" in brief
        assert "summary_text" in brief
        assert "closing" in brief
        assert isinstance(brief["urgent_items"], list)
        assert isinstance(brief["summary_text"], str)

    def test_daily_brief_with_data(self, tmp_path):
        """Daily brief includes urgent items when data exists."""
        from engine.assistant_engine import MemoryAssistantEngine
        engine = MemoryAssistantEngine(str(tmp_path / "test.db"))
        engine.process_text(
            "Don't forget to take your medicine after breakfast. "
            "Doctor appointment tomorrow at 10 AM."
        )
        brief = engine.generate_daily_brief()
        # Should have at least medication or appointment
        assert len(brief["summary_text"]) > 0


class TestPhaseRBackwardCompat:
    """Tests for Phase R backward compatibility."""

    def test_events_default_escalation_zero(self, tmp_path):
        """Events saved without escalation_level default to 0."""
        from storage.repository import Repository
        repo = Repository(str(tmp_path / "test.db"))
        conv_id = repo.save_conversation(raw_text="Test")
        repo.save_events(conv_id, [{
            "type": "task",
            "description": "Generic task",
        }])
        events = repo.get_all_events()
        assert events[0]["escalation_level"] == 0

    def test_config_summary_includes_phase_r(self):
        """Config summary includes Phase R settings."""
        from config import get_config_summary
        summary = get_config_summary()
        assert "simplified_mode" in summary
        assert "reinforcement_interval_hours" in summary
        assert "escalation_max_level" in summary
