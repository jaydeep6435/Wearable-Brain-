"""
SQLite Database Core
====================

Provides a thin database layer used by Repository:
- schema initialization
- query helpers returning dict rows
- UUID ID generation
- basic table statistics
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from typing import Any


class Database:
    """Lightweight SQLite wrapper for the Memory Assistant repository."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                recorded_at TEXT DEFAULT (datetime('now')),
                raw_text TEXT,
                audio_path TEXT,
                source TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                speaker TEXT,
                text TEXT,
                start_time REAL,
                end_time REAL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                type TEXT,
                description TEXT,
                raw_date TEXT,
                raw_time TEXT,
                parsed_date TEXT,
                parsed_time TEXT,
                person TEXT,
                fingerprint TEXT UNIQUE,
                importance_score INTEGER DEFAULT 0,
                escalation_level INTEGER DEFAULT 0,
                recorded_at TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                summary TEXT,
                key_points TEXT,
                mode TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                event_id TEXT,
                trigger_time TEXT,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY(event_id) REFERENCES events(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS speaker_voiceprints (
                id TEXT PRIMARY KEY,
                speaker_name TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_embeddings (
                conversation_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_patterns (
                phrase TEXT PRIMARY KEY,
                category TEXT,
                frequency INTEGER DEFAULT 1,
                last_seen TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_reinforcement (
                event_id TEXT PRIMARY KEY,
                last_shown TEXT,
                shown_count INTEGER DEFAULT 0,
                FOREIGN KEY(event_id) REFERENCES events(id)
            )
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_date_time ON events(parsed_date, parsed_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_segments_conversation ON segments(conversation_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_reminders_status_time ON reminders(status, trigger_time)")

        self.conn.commit()

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        cur = self.conn.cursor()
        cur.execute(query, params)
        self.conn.commit()

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict | None:
        cur = self.conn.cursor()
        row = cur.execute(query, params).fetchone()
        return dict(row) if row is not None else None

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict]:
        cur = self.conn.cursor()
        rows = cur.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def save_encrypted(self) -> None:
        """Compatibility hook. Encryption manager can be integrated here later."""
        return

    def get_stats(self) -> dict:
        tables = [
            "conversations",
            "segments",
            "events",
            "summaries",
            "reminders",
            "speaker_voiceprints",
            "conversation_embeddings",
            "memory_patterns",
            "memory_reinforcement",
        ]
        out: dict[str, int] = {}
        for t in tables:
            row = self.fetch_one(f"SELECT COUNT(*) AS cnt FROM {t}")
            out[t] = int(row["cnt"]) if row else 0
        return out
