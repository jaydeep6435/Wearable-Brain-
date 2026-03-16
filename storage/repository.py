"""
Repository — CRUD Operations for Memory Assistant
====================================================
High-level data access layer built on top of the Database class.

Provides methods for saving/querying conversations, segments, events,
summaries, and reminders — all with proper deduplication.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta

from .db import Database


class Repository:
    """
    Data access layer for the Memory Assistant.

    Usage:
        repo = Repository("memory.db")
        conv_id = repo.save_conversation(raw_text="Hello doctor...")
        repo.save_events(conv_id, [{"type": "meeting", ...}])
        events = repo.get_all_events()
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "memory.db"
            )
        self.db = Database(db_path)

    # ── Fingerprint (dedup) ────────────────────────────────────

    @staticmethod
    def _make_fingerprint(event: dict) -> str:
        """Create MD5 fingerprint from event core fields for dedup."""
        key_parts = [
            (event.get("type") or "").lower().strip(),
            (event.get("description") or "").lower().strip(),
            (
                event.get("parsed_date")
                or event.get("raw_date")
                or event.get("date")
                or ""
            ).strip(),
            (
                event.get("parsed_time")
                or event.get("raw_time")
                or event.get("time")
                or ""
            ).strip(),
        ]
        key = "|".join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()

    # ── Conversations ──────────────────────────────────────────

    def save_conversation(
        self,
        raw_text: str = None,
        audio_path: str = None,
        source: str = "text",
    ) -> str:
        """Save a new conversation. Returns the conversation ID."""
        conv_id = self.db.new_id()
        print(f"[Repo] Saving conversation {conv_id[:8]}... (source={source}, text_len={len(raw_text or '')})")
        self.db.execute(
            "INSERT INTO conversations (id, timestamp, raw_text, audio_path, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, datetime.now().isoformat(), raw_text, audio_path, source),
        )
        # Flush to encrypted file if active
        self.db.save_encrypted()
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        """Get a conversation by ID, including its segments and summary."""
        conv = self.db.fetch_one(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        )
        if not conv:
            return None

        conv["segments"] = self.db.fetch_all(
            "SELECT * FROM segments WHERE conversation_id = ? ORDER BY start_time",
            (conv_id,),
        )
        conv["summary"] = self.db.fetch_one(
            "SELECT * FROM summaries WHERE conversation_id = ? ORDER BY created_at DESC",
            (conv_id,),
        )
        conv["events"] = self.db.fetch_all(
            "SELECT * FROM events WHERE conversation_id = ?", (conv_id,)
        )
        return conv

    def get_all_conversations(self, limit: int = 50) -> list[dict]:
        """Get recent conversations (most recent first)."""
        return self.db.fetch_all(
            "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    # ── Segments ───────────────────────────────────────────────

    def save_segments(self, conv_id: str, segments: list[dict]) -> int:
        """
        Save speaker segments for a conversation.
        Returns count of segments saved.
        """
        count = 0
        for seg in segments:
            seg_id = self.db.new_id()
            self.db.execute(
                "INSERT INTO segments (id, conversation_id, speaker, text, start_time, end_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    seg_id,
                    conv_id,
                    seg.get("speaker"),
                    seg.get("text", ""),
                    seg.get("start"),
                    seg.get("end"),
                ),
            )
            count += 1
        return count

    # ── Events ─────────────────────────────────────────────────

    def save_events(self, conv_id: str, events: list[dict]) -> int:
        """
        Save extracted events with deduplication.
        Returns count of NEW events saved (duplicates skipped).
        """
        saved = 0
        for event in events:
            fp = self._make_fingerprint(event)

            # Check for duplicate
            existing = self.db.fetch_one(
                "SELECT id FROM events WHERE fingerprint = ?", (fp,)
            )
            if existing:
                print(f"[Repo] Duplicate skipped: {event.get('description', '?')[:50]}")
                continue

            event_id = self.db.new_id()
            self.db.execute(
                "INSERT INTO events "
                "(id, conversation_id, type, description, raw_date, raw_time, "
                "parsed_date, parsed_time, person, fingerprint, importance_score, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    conv_id,
                    event.get("type", "unknown"),
                    event.get("description"),
                    event.get("raw_date") or event.get("date"),
                    event.get("raw_time") or event.get("time"),
                    event.get("parsed_date"),
                    event.get("parsed_time"),
                    event.get("person"),
                    fp,
                    event.get("importance_score", 0),
                    datetime.now().isoformat(),
                ),
            )
            saved += 1

        if saved < len(events):
            print(f"[Repo] {len(events) - saved} duplicates skipped out of {len(events)}")
        return saved

    def save_single_event(self, event: dict, conv_id: str = None) -> str | None:
        """Save a single event. Returns event ID or None if duplicate."""
        fp = self._make_fingerprint(event)
        existing = self.db.fetch_one(
            "SELECT id FROM events WHERE fingerprint = ?", (fp,)
        )
        if existing:
            return None

        event_id = self.db.new_id()
        self.db.execute(
            "INSERT INTO events "
            "(id, conversation_id, type, description, raw_date, raw_time, "
            "parsed_date, parsed_time, person, fingerprint, importance_score, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                conv_id,
                event.get("type", "unknown"),
                event.get("description"),
                event.get("raw_date") or event.get("date"),
                event.get("raw_time") or event.get("time"),
                event.get("parsed_date"),
                event.get("parsed_time"),
                event.get("person"),
                fp,
                event.get("importance_score", 0),
                datetime.now().isoformat(),
            ),
        )
        return event_id

    def get_all_events(
        self, type_filter: str = None, search: str = None
    ) -> list[dict]:
        """Get all events, optionally filtered by type or keyword search."""
        if type_filter:
            return self.db.fetch_all(
                "SELECT * FROM events WHERE type = ? ORDER BY recorded_at DESC",
                (type_filter,),
            )
        if search:
            pattern = f"%{search}%"
            return self.db.fetch_all(
                "SELECT * FROM events WHERE "
                "description LIKE ? OR type LIKE ? OR person LIKE ? "
                "ORDER BY recorded_at DESC",
                (pattern, pattern, pattern),
            )
        return self.db.fetch_all(
            "SELECT * FROM events ORDER BY recorded_at DESC"
        )

    def get_upcoming_events(self, minutes: int = 60) -> list[dict]:
        """
        Get events happening in the next N minutes.
        Matches events with parsed_date matching today/tomorrow
        and parsed_time within the window.
        """
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        events = self.db.fetch_all(
            "SELECT * FROM events WHERE parsed_date IN (?, ?) "
            "ORDER BY parsed_date, parsed_time",
            (today, tomorrow),
        )

        # Filter by time window if minutes specified
        results = []
        for event in events:
            event_dict = dict(event)
            # Calculate minutes until event
            if event.get("parsed_date") and event.get("parsed_time"):
                try:
                    event_dt = datetime.strptime(
                        f"{event['parsed_date']} {event['parsed_time']}",
                        "%Y-%m-%d %H:%M",
                    )
                    delta = (event_dt - now).total_seconds() / 60
                    event_dict["minutes_until"] = round(delta)
                    if -30 <= delta <= minutes:  # Include recently passed (30min grace)
                        results.append(event_dict)
                except (ValueError, TypeError):
                    results.append(event_dict)
            else:
                results.append(event_dict)

        return results

    def search_events(self, keyword: str) -> list[dict]:
        """Search events by keyword across all text fields."""
        pattern = f"%{keyword}%"
        return self.db.fetch_all(
            "SELECT * FROM events WHERE "
            "description LIKE ? OR type LIKE ? OR person LIKE ? OR "
            "raw_date LIKE ? OR raw_time LIKE ? "
            "ORDER BY recorded_at DESC",
            (pattern, pattern, pattern, pattern, pattern),
        )

    # ── Summaries ──────────────────────────────────────────────

    def save_summary(
        self,
        conv_id: str,
        summary: str,
        key_points: list[str] = None,
        mode: str = "rule",
    ) -> str:
        """Save a conversation summary. Returns summary ID."""
        sum_id = self.db.new_id()
        kp_json = json.dumps(key_points) if key_points else "[]"
        self.db.execute(
            "INSERT INTO summaries (id, conversation_id, summary, key_points, mode) "
            "VALUES (?, ?, ?, ?, ?)",
            (sum_id, conv_id, summary, kp_json, mode),
        )
        return sum_id

    def get_summaries(self, conv_id: str = None) -> list[dict]:
        """Get summaries, optionally for a specific conversation."""
        if conv_id:
            return self.db.fetch_all(
                "SELECT * FROM summaries WHERE conversation_id = ? "
                "ORDER BY created_at DESC",
                (conv_id,),
            )
        return self.db.fetch_all(
            "SELECT * FROM summaries ORDER BY created_at DESC"
        )

    def search_summaries(self, keyword: str = None) -> list[dict]:
        """
        Search summaries by keyword, or return all if no keyword given.
        Used by semantic search to build the document corpus.
        """
        if keyword:
            pattern = f"%{keyword}%"
            return self.db.fetch_all(
                "SELECT * FROM summaries WHERE "
                "summary LIKE ? OR key_points LIKE ? "
                "ORDER BY created_at DESC",
                (pattern, pattern),
            )
        return self.db.fetch_all(
            "SELECT * FROM summaries ORDER BY created_at DESC"
        )

    def get_recent_summaries(self, limit: int = 5) -> list[dict]:
        """Get most recent summaries with conversation context."""
        return self.db.fetch_all(
            "SELECT s.*, c.recorded_at "
            "FROM summaries s "
            "LEFT JOIN conversations c ON s.conversation_id = c.id "
            "ORDER BY s.created_at DESC LIMIT ?",
            (limit,),
        )

    # ── Reminders ──────────────────────────────────────────────

    def save_reminder(self, event_id: str, trigger_time: str) -> str:
        """Save a reminder. Returns reminder ID."""
        # Prevent duplicate reminder for same event
        existing = self.db.fetch_one(
            "SELECT id FROM reminders WHERE event_id = ? AND status = 'pending'",
            (event_id,),
        )
        if existing:
            return existing["id"]

        rem_id = self.db.new_id()
        self.db.execute(
            "INSERT INTO reminders (id, event_id, trigger_time, status) "
            "VALUES (?, ?, ?, 'pending')",
            (rem_id, event_id, trigger_time),
        )
        return rem_id

    def get_pending_reminders(self) -> list[dict]:
        """Get all pending reminders with their event details."""
        return self.db.fetch_all(
            "SELECT r.*, e.type, e.description, e.parsed_date, e.parsed_time, e.person "
            "FROM reminders r "
            "LEFT JOIN events e ON r.event_id = e.id "
            "WHERE r.status = 'pending' "
            "ORDER BY r.trigger_time",
        )

    def get_reminders_by_status(self, status: str) -> list[dict]:
        """Get reminders by status: 'pending', 'fired', 'snoozed', 'dismissed'."""
        return self.db.fetch_all(
            "SELECT r.*, e.type, e.description, e.parsed_date, e.parsed_time, e.person "
            "FROM reminders r "
            "LEFT JOIN events e ON r.event_id = e.id "
            "WHERE r.status = ? "
            "ORDER BY r.trigger_time",
            (status,),
        )

    def mark_reminder_fired(self, reminder_id: str) -> None:
        """Mark a reminder as fired."""
        self.db.execute(
            "UPDATE reminders SET status = 'fired' WHERE id = ?",
            (reminder_id,),
        )

    def dismiss_reminder(self, reminder_id: str) -> None:
        """Mark a reminder as dismissed (user explicitly silenced it)."""
        self.db.execute(
            "UPDATE reminders SET status = 'dismissed' WHERE id = ?",
            (reminder_id,),
        )

    def snooze_reminder(self, reminder_id: str, new_trigger_time: str) -> None:
        """Snooze a reminder — reset to pending with a new trigger time."""
        self.db.execute(
            "UPDATE reminders SET status = 'pending', trigger_time = ? WHERE id = ?",
            (new_trigger_time, reminder_id),
        )

    def auto_schedule_reminders(self, lead_minutes: int = 15) -> int:
        """
        Create reminders for all events that don't have one yet.
        Trigger time = event datetime minus `lead_minutes`.

        Returns count of new reminders created.
        """
        # Get events with parsed datetime that lack a pending reminder
        events = self.db.fetch_all(
            "SELECT e.id, e.parsed_date, e.parsed_time "
            "FROM events e "
            "WHERE e.parsed_date IS NOT NULL "
            "AND e.parsed_time IS NOT NULL "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM reminders r "
            "  WHERE r.event_id = e.id AND r.status IN ('pending', 'fired')"
            ")",
        )

        count = 0
        for event in events:
            try:
                event_dt = datetime.strptime(
                    f"{event['parsed_date']} {event['parsed_time']}",
                    "%Y-%m-%d %H:%M",
                )
                trigger_dt = event_dt - timedelta(minutes=lead_minutes)

                # Only schedule future reminders
                if trigger_dt > datetime.now():
                    self.save_reminder(event["id"], trigger_dt.isoformat())
                    count += 1
            except (ValueError, TypeError):
                continue

        return count

    # ── Migration from JSON ────────────────────────────────────

    def migrate_from_json(self, json_path: str) -> dict:
        """
        Import events from the legacy memory.json file.
        Returns a summary of what was imported.

        Args:
            json_path: path to the memory.json file
        """
        if not os.path.isfile(json_path):
            print(f"[Repo] JSON file not found: {json_path}")
            return {"status": "error", "message": "File not found"}

        with open(json_path, "r", encoding="utf-8") as f:
            events = json.load(f)

        if not isinstance(events, list):
            return {"status": "error", "message": "Expected a JSON array"}

        # Create a migration conversation
        conv_id = self.save_conversation(
            raw_text="[Migrated from memory.json]",
            source="migration",
        )

        saved = 0
        skipped = 0

        for event in events:
            fp = self._make_fingerprint(event)
            existing = self.db.fetch_one(
                "SELECT id FROM events WHERE fingerprint = ?", (fp,)
            )
            if existing:
                skipped += 1
                continue

            event_id = self.db.new_id()
            self.db.execute(
                "INSERT INTO events "
                "(id, conversation_id, type, description, raw_date, raw_time, "
                "parsed_date, parsed_time, person, fingerprint, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    conv_id,
                    event.get("type", "unknown"),
                    event.get("description"),
                    event.get("raw_date") or event.get("date"),
                    event.get("raw_time") or event.get("time"),
                    event.get("parsed_date"),
                    event.get("parsed_time"),
                    event.get("person"),
                    fp,
                    event.get("recorded_at", datetime.now().isoformat()),
                ),
            )
            saved += 1

        result = {
            "status": "ok",
            "total_in_json": len(events),
            "saved": saved,
            "duplicates_skipped": skipped,
            "conversation_id": conv_id,
        }
        print(f"[Repo] Migration complete: {saved} saved, {skipped} skipped out of {len(events)} events")
        return result

    # ── Voiceprints ─────────────────────────────────────────────

    def save_voiceprint(self, speaker_name: str, embedding_bytes: bytes, max_per_speaker: int = 10) -> str:
        """
        Save a voice embedding for a speaker.

        Multiple embeddings per speaker are allowed (different recordings
        improve matching accuracy). If the count exceeds max_per_speaker,
        the oldest embeddings are rotated out.

        Args:
            speaker_name: Human-readable name (e.g., "Dr. Smith").
            embedding_bytes: Embedding vector as raw bytes (float32).
            max_per_speaker: Maximum embeddings to keep per speaker.

        Returns:
            Voiceprint record ID.
        """
        vp_id = self.db.new_id()
        self.db.execute(
            "INSERT INTO speaker_voiceprints (id, speaker_name, embedding) "
            "VALUES (?, ?, ?)",
            (vp_id, speaker_name, embedding_bytes),
        )
        print(f"[Repo] Saved voiceprint for '{speaker_name}' ({len(embedding_bytes)} bytes)")

        # Auto-rotate: keep only the most recent N embeddings
        self.rotate_voiceprints(speaker_name, max_count=max_per_speaker)

        return vp_id

    def rotate_voiceprints(self, speaker_name: str, max_count: int = 10) -> int:
        """
        Rotate voiceprints for a speaker, keeping only the most recent N.

        Deletes the oldest embeddings beyond max_count.

        Args:
            speaker_name: Speaker to rotate embeddings for.
            max_count: Maximum number of embeddings to keep.

        Returns:
            Number of embeddings deleted.
        """
        existing = self.get_voiceprints_for_speaker(speaker_name)
        excess = len(existing) - max_count

        if excess <= 0:
            return 0

        # existing is ordered by created_at ASC — oldest first
        to_delete = existing[:excess]
        ids = [vp["id"] for vp in to_delete]

        placeholders = ",".join("?" * len(ids))
        self.db.execute(
            f"DELETE FROM speaker_voiceprints WHERE id IN ({placeholders})",
            tuple(ids),
        )
        print(f"[Repo] Rotated {len(ids)} oldest voiceprints for '{speaker_name}' (keeping {max_count})")
        return len(ids)

    def get_all_voiceprints(self) -> list[dict]:
        """
        Get all stored voiceprints.

        Returns:
            List of dicts with keys: id, speaker_name, embedding (bytes), created_at.
        """
        return self.db.fetch_all(
            "SELECT id, speaker_name, embedding, created_at "
            "FROM speaker_voiceprints ORDER BY created_at"
        )

    def get_voiceprints_for_speaker(self, speaker_name: str) -> list[dict]:
        """
        Get all voiceprints for a specific speaker.

        Args:
            speaker_name: The speaker's display name.

        Returns:
            List of voiceprint dicts for that speaker.
        """
        return self.db.fetch_all(
            "SELECT id, speaker_name, embedding, created_at "
            "FROM speaker_voiceprints WHERE speaker_name = ? ORDER BY created_at",
            (speaker_name,),
        )

    def delete_voiceprints(self, speaker_name: str) -> int:
        """
        Delete all voiceprints for a speaker.

        Returns:
            Number of voiceprints deleted.
        """
        existing = self.get_voiceprints_for_speaker(speaker_name)
        count = len(existing)
        if count > 0:
            self.db.execute(
                "DELETE FROM speaker_voiceprints WHERE speaker_name = ?",
                (speaker_name,),
            )
            print(f"[Repo] Deleted {count} voiceprints for '{speaker_name}'")
        return count

    def count_voiceprints(self) -> int:
        """Total number of stored voiceprints."""
        row = self.db.fetch_one("SELECT COUNT(*) as cnt FROM speaker_voiceprints")
        return row["cnt"] if row else 0

    # ── Conversation Embeddings ────────────────────────────────

    def save_conversation_embedding(
        self, conversation_id: str, embedding_bytes: bytes
    ) -> None:
        """
        Save or update a conversation's sentence embedding.

        Uses INSERT OR REPLACE since conversation_id is PRIMARY KEY.

        Args:
            conversation_id: Conversation to associate the embedding with.
            embedding_bytes: Raw float32 embedding as bytes.
        """
        self.db.execute(
            "INSERT OR REPLACE INTO conversation_embeddings "
            "(conversation_id, embedding) VALUES (?, ?)",
            (conversation_id, embedding_bytes),
        )

    def get_conversation_embedding(self, conversation_id: str) -> bytes | None:
        """
        Get the stored embedding for a conversation.

        Returns:
            Raw bytes of the embedding, or None if not found.
        """
        row = self.db.fetch_one(
            "SELECT embedding FROM conversation_embeddings WHERE conversation_id = ?",
            (conversation_id,),
        )
        return row["embedding"] if row else None

    def get_all_conversation_embeddings(self) -> dict[str, bytes]:
        """
        Get all stored conversation embeddings.

        Returns:
            Dict mapping conversation_id → embedding bytes.
        """
        rows = self.db.fetch_all(
            "SELECT conversation_id, embedding FROM conversation_embeddings"
        )
        return {r["conversation_id"]: r["embedding"] for r in rows}

    # ── Memory Patterns (Phase Q) ──────────────────────────────

    def increment_pattern(self, phrase: str, category: str = "general") -> int:
        """
        Insert or increment a recurring phrase pattern.
        Returns the new frequency count.
        """
        existing = self.db.fetch_one(
            "SELECT frequency FROM memory_patterns WHERE phrase = ?",
            (phrase,),
        )
        if existing:
            new_freq = existing["frequency"] + 1
            self.db.execute(
                "UPDATE memory_patterns SET frequency = ?, last_seen = ? "
                "WHERE phrase = ?",
                (new_freq, datetime.now().isoformat(), phrase),
            )
            return new_freq
        else:
            self.db.execute(
                "INSERT INTO memory_patterns (phrase, category, frequency, last_seen) "
                "VALUES (?, ?, 1, ?)",
                (phrase, category, datetime.now().isoformat()),
            )
            return 1

    def get_patterns(self, min_frequency: int = 1) -> list[dict]:
        """Get recurring patterns, optionally filtered by minimum frequency."""
        return self.db.fetch_all(
            "SELECT * FROM memory_patterns WHERE frequency >= ? "
            "ORDER BY frequency DESC",
            (min_frequency,),
        )

    def get_pattern(self, phrase: str) -> dict | None:
        """Get a single pattern by phrase."""
        return self.db.fetch_one(
            "SELECT * FROM memory_patterns WHERE phrase = ?",
            (phrase,),
        )

    def save_pattern(self, phrase: str, category: str = "general") -> int:
        """Compatibility alias used by engine-side pattern detection."""
        return self.increment_pattern(phrase=phrase, category=category)

    def get_memory_stats(self) -> dict:
        """Compatibility alias for debug endpoints."""
        return self.get_stats()

    def get_urgent_events(self, hours: int = 24) -> list[dict]:
        """
        Get medication and meeting events that may be urgent.
        Returns all high-importance events for urgency evaluation.
        """
        return self.db.fetch_all(
            "SELECT * FROM events "
            "WHERE type IN ('medication', 'meeting', 'appointment') "
            "AND importance_score >= 3 "
            "ORDER BY importance_score DESC, recorded_at DESC",
        )

    # ── Memory Reinforcement (Phase R) ──────────────────────────

    def mark_reinforcement_shown(self, event_id: str) -> None:
        """Record that a critical event was shown to the user."""
        existing = self.db.fetch_one(
            "SELECT shown_count FROM memory_reinforcement WHERE event_id = ?",
            (event_id,),
        )
        now = datetime.now().isoformat()
        if existing:
            self.db.execute(
                "UPDATE memory_reinforcement SET last_shown = ?, shown_count = shown_count + 1 "
                "WHERE event_id = ?",
                (now, event_id),
            )
        else:
            self.db.execute(
                "INSERT INTO memory_reinforcement (event_id, last_shown, shown_count) "
                "VALUES (?, ?, 1)",
                (event_id, now),
            )

    def get_reinforcement_candidates(self, interval_hours: int = 12) -> list[dict]:
        """
        Get high-importance events not shown within interval_hours.
        Returns events needing re-display.
        """
        cutoff = (datetime.now() - timedelta(hours=interval_hours)).isoformat()
        return self.db.fetch_all(
            "SELECT e.*, r.last_shown, r.shown_count "
            "FROM events e "
            "LEFT JOIN memory_reinforcement r ON e.id = r.event_id "
            "WHERE e.importance_score >= 5 "
            "AND (r.last_shown IS NULL OR r.last_shown < ?) "
            "ORDER BY e.importance_score DESC",
            (cutoff,),
        )

    def get_reinforcement_record(self, event_id: str) -> dict | None:
        """Get reinforcement tracking record for an event."""
        return self.db.fetch_one(
            "SELECT * FROM memory_reinforcement WHERE event_id = ?",
            (event_id,),
        )

    # ── Escalation (Phase R) ──────────────────────────────────

    def escalate_event(self, event_id: str, level: int) -> None:
        """Set escalation level for an event."""
        self.db.execute(
            "UPDATE events SET escalation_level = ? WHERE id = ?",
            (level, event_id),
        )

    def get_escalation_candidates(self) -> list[dict]:
        """Get urgent events that may need escalation."""
        return self.db.fetch_all(
            "SELECT * FROM events "
            "WHERE type IN ('medication', 'meeting', 'appointment') "
            "AND importance_score >= 5 "
            "AND escalation_level < 3 "
            "ORDER BY importance_score DESC, recorded_at DESC",
        )

    def get_escalated_events(self, min_level: int = 1) -> list[dict]:
        """Get events at or above a certain escalation level."""
        return self.db.fetch_all(
            "SELECT * FROM events WHERE escalation_level >= ? "
            "ORDER BY escalation_level DESC, importance_score DESC",
            (min_level,),
        )

    # ── Stats ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get database statistics."""
        return self.db.get_stats()


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    repo = Repository("test_repo.db")

    # Test conversation
    cid = repo.save_conversation(raw_text="Doctor said take medicine at 10 AM")
    print(f"Conversation: {cid}")

    # Test events with dedup
    events = [
        {"type": "meeting", "description": "Doctor appointment", "date": "tomorrow", "time": "10 AM"},
        {"type": "medication", "description": "Take medicine after breakfast"},
        {"type": "meeting", "description": "Doctor appointment", "date": "tomorrow", "time": "10 AM"},  # Duplicate!
    ]
    saved = repo.save_events(cid, events)
    print(f"Saved {saved} events (1 should be duplicate)")

    # Test summary
    repo.save_summary(cid, "Patient has doctor appointment tomorrow", ["medicine", "appointment"])

    # Test retrieval
    all_events = repo.get_all_events()
    print(f"Total events: {len(all_events)}")

    # Test search
    found = repo.search_events("doctor")
    print(f"Search 'doctor': {len(found)} results")

    # Test full conversation
    conv = repo.get_conversation(cid)
    print(f"Full conversation: {conv['id'][:8]}... with {len(conv['events'])} events")

    print(f"\nStats: {repo.get_stats()}")

    # Cleanup
    os.remove("test_repo.db")
    print("Test passed ✅")
