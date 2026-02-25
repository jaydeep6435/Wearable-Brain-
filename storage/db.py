"""
SQLite Database Manager — With Optional Encryption
=====================================================
Connection manager and schema creation for the Memory Assistant.

Tables:
  - conversations       : Top-level conversation records
  - segments            : Speaker-labeled text segments
  - events              : Extracted events (meetings, medications, tasks)
  - summaries           : Conversation summaries and key points
  - reminders           : Scheduled reminder triggers
  - speaker_voiceprints : Voice embedding vectors

Encryption:
  If a passphrase is provided, the database file is encrypted at rest
  using AES-256-GCM. On open, the file is decrypted to a temp location.
  On close (or atexit), the temp file is re-encrypted and deleted.
  Without a passphrase, the database operates in plaintext mode.

All tables use TEXT PRIMARY KEY with UUID4 identifiers.
"""

import atexit
import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime


# Default database path (next to project root)
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory.db")

# ── Schema SQL ─────────────────────────────────────────────────

SCHEMA_SQL = """
-- Conversations: top-level record for each processed input
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    timestamp   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_text    TEXT,
    audio_path  TEXT,
    source      TEXT DEFAULT 'text'  -- 'text', 'audio', 'vad'
);

-- Segments: speaker-labeled pieces within a conversation
CREATE TABLE IF NOT EXISTS segments (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    speaker         TEXT,
    text            TEXT NOT NULL,
    start_time      REAL,
    end_time        REAL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Events: extracted structured events
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    type            TEXT NOT NULL,
    description     TEXT,
    raw_date        TEXT,
    raw_time        TEXT,
    parsed_date     TEXT,
    parsed_time     TEXT,
    person          TEXT,
    fingerprint     TEXT UNIQUE,
    importance_score INTEGER DEFAULT 0,
    escalation_level INTEGER DEFAULT 0,
    recorded_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
);

-- Summaries: conversation summaries
CREATE TABLE IF NOT EXISTS summaries (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    summary         TEXT,
    key_points      TEXT,
    mode            TEXT DEFAULT 'rule',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Reminders: scheduled alerts
CREATE TABLE IF NOT EXISTS reminders (
    id              TEXT PRIMARY KEY,
    event_id        TEXT,
    trigger_time    DATETIME,
    status          TEXT DEFAULT 'pending',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

-- Speaker voiceprints: voice embedding vectors for auto-identification
CREATE TABLE IF NOT EXISTS speaker_voiceprints (
    id              TEXT PRIMARY KEY,
    speaker_name    TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Conversation embeddings: precomputed sentence vectors for semantic search
CREATE TABLE IF NOT EXISTS conversation_embeddings (
    conversation_id TEXT PRIMARY KEY,
    embedding       BLOB NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_segments_conv    ON segments(conversation_id);
CREATE INDEX IF NOT EXISTS idx_events_conv      ON events(conversation_id);
CREATE INDEX IF NOT EXISTS idx_events_type      ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_fp        ON events(fingerprint);
CREATE INDEX IF NOT EXISTS idx_summaries_conv   ON summaries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status);
CREATE INDEX IF NOT EXISTS idx_reminders_time   ON reminders(trigger_time);
CREATE INDEX IF NOT EXISTS idx_voiceprint_speaker ON speaker_voiceprints(speaker_name);

-- Memory patterns: recurring conversation phrases (Alzheimer prioritization)
CREATE TABLE IF NOT EXISTS memory_patterns (
    phrase      TEXT PRIMARY KEY,
    category    TEXT DEFAULT 'general',
    frequency   INTEGER DEFAULT 1,
    last_seen   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Memory reinforcement: tracks when critical items were shown (Phase R)
CREATE TABLE IF NOT EXISTS memory_reinforcement (
    event_id    TEXT PRIMARY KEY,
    last_shown  DATETIME,
    shown_count INTEGER DEFAULT 0,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_conv_emb          ON conversation_embeddings(conversation_id);
"""


class Database:
    """
    SQLite database connection manager with optional encryption.

    Modes:
      - Plaintext (no passphrase): Opens db_path directly, no encryption.
      - Encrypted (with passphrase): Decrypts to temp file on open,
        re-encrypts on close/save.

    Usage:
        # Plaintext mode (backward compatible)
        db = Database("memory.db")

        # Encrypted mode
        db = Database("memory.db", passphrase="secret")

        # Or auto-detect from environment
        from storage.encryption import get_passphrase
        db = Database("memory.db", passphrase=get_passphrase())
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, passphrase: str = None):
        self.db_path = db_path
        self._passphrase = passphrase
        self._encryptor = None
        self._temp_path = None
        self._encrypted = False
        self._closed = False

        # Determine working path
        if passphrase:
            self._setup_encrypted(passphrase)
        else:
            self._working_path = db_path

        self._ensure_directory()
        self._init_schema()

        mode = "encrypted" if self._encrypted else "plaintext"
        print(f"[DB] SQLite ready ({mode}): {self.db_path}")

    def _setup_encrypted(self, passphrase: str):
        """Set up encrypted database lifecycle."""
        try:
            from storage.encryption import DatabaseEncryption
        except ImportError:
            print("[DB] WARNING: cryptography library not installed — running unencrypted")
            self._working_path = self.db_path
            return

        self._encryptor = DatabaseEncryption(passphrase)
        enc_path = self.db_path + ".enc"

        if DatabaseEncryption.is_encrypted(enc_path):
            # Encrypted file exists → decrypt to temp
            self._temp_path = self._make_temp_path()
            self._encryptor.decrypt_file(enc_path, self._temp_path)
            self._working_path = self._temp_path
            self._encrypted = True
            print("[DB] Decrypted database loaded from encrypted file")

        elif os.path.isfile(self.db_path) and DatabaseEncryption.is_sqlite(self.db_path):
            # Plaintext file exists → migrate (open plaintext, will encrypt on save)
            self._temp_path = self._make_temp_path()
            shutil.copy2(self.db_path, self._temp_path)
            self._working_path = self._temp_path
            self._encrypted = True
            print("[DB] Plaintext database detected — will encrypt on save")

        else:
            # No existing DB → create new encrypted DB
            self._temp_path = self._make_temp_path()
            self._working_path = self._temp_path
            self._encrypted = True

        # Register cleanup
        if self._encrypted:
            atexit.register(self._atexit_handler)

    def _make_temp_path(self) -> str:
        """Create a secure temp file path for the decrypted database."""
        fd, path = tempfile.mkstemp(suffix=".db", prefix="wbrain_")
        os.close(fd)
        return path

    def _ensure_directory(self):
        """Create parent directory if needed."""
        parent = os.path.dirname(self._working_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _init_schema(self):
        """Create all tables and indexes if they don't exist."""
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)
            # Migrations for existing databases
            self._migrate(conn)

    def _migrate(self, conn):
        """Run lightweight migrations for schema upgrades."""
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()]
            # Phase Q: Add importance_score to events if missing
            if "importance_score" not in cols:
                conn.execute("ALTER TABLE events ADD COLUMN importance_score INTEGER DEFAULT 0")
            # Phase R: Add escalation_level to events if missing
            if "escalation_level" not in cols:
                conn.execute("ALTER TABLE events ADD COLUMN escalation_level INTEGER DEFAULT 0")
        except Exception:
            pass

        # Create indexes on migrated columns (safe after ALTER TABLE)
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_importance ON events(importance_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_escalation ON events(escalation_level)")
        except Exception:
            pass

    @contextmanager
    def connection(self):
        """
        Context manager for database connections.
        Enables WAL mode and foreign keys for performance + integrity.
        """
        conn = sqlite3.connect(self._working_path)
        conn.row_factory = sqlite3.Row  # Dict-like access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a single SQL statement."""
        with self.connection() as conn:
            conn.execute(sql, params)

    def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement with multiple parameter sets."""
        with self.connection() as conn:
            conn.executemany(sql, params_list)

    def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Fetch a single row as a dict."""
        with self.connection() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Fetch all rows as a list of dicts."""
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def count(self, table: str) -> int:
        """Count rows in a table."""
        result = self.fetch_one(f"SELECT COUNT(*) as cnt FROM {table}")
        return result["cnt"] if result else 0

    @staticmethod
    def new_id() -> str:
        """Generate a new UUID4 identifier."""
        return str(uuid.uuid4())

    def get_stats(self) -> dict:
        """Get table row counts."""
        return {
            "conversations": self.count("conversations"),
            "segments": self.count("segments"),
            "events": self.count("events"),
            "summaries": self.count("summaries"),
            "reminders": self.count("reminders"),
        }

    # ── Encryption Lifecycle ───────────────────────────────────

    @property
    def is_encrypted(self) -> bool:
        """Whether the database is running in encrypted mode."""
        return self._encrypted

    def save_encrypted(self) -> None:
        """
        Flush the current database state to the encrypted file.
        Call periodically (e.g., after each conversation) for durability.
        No-op if not in encrypted mode.
        """
        if not self._encrypted or not self._encryptor:
            return

        enc_path = self.db_path + ".enc"
        try:
            # Checkpoint WAL to ensure all data is in main DB file
            try:
                conn = sqlite3.connect(self._working_path)
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except Exception:
                pass

            self._encryptor.encrypt_file(self._working_path, enc_path)
        except Exception as e:
            print(f"[DB] WARNING: Failed to save encrypted file: {e}")

    def close(self) -> None:
        """
        Final save + cleanup.
        Encrypts the database back and removes the temp file.
        """
        if self._closed:
            return

        if self._encrypted and self._encryptor:
            # Final encrypt (must happen BEFORE setting _closed)
            self.save_encrypted()

            # Now mark as closed (prevents double-close)
            self._closed = True

            # Migrate: remove old plaintext file if encrypted version now exists
            enc_path = self.db_path + ".enc"
            if os.path.isfile(enc_path) and os.path.isfile(self.db_path):
                from storage.encryption import DatabaseEncryption
                if DatabaseEncryption.is_sqlite(self.db_path):
                    try:
                        os.remove(self.db_path)
                        print("[DB] Plaintext database removed after encryption migration")
                    except OSError:
                        pass

            # Clean up temp file
            self._cleanup_temp()
        else:
            self._closed = True

    def _cleanup_temp(self):
        """Remove temporary decrypted file."""
        if self._temp_path and os.path.isfile(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass

            # Also clean WAL/SHM temp files
            for suffix in ["-wal", "-shm"]:
                wal = self._temp_path + suffix
                if os.path.isfile(wal):
                    try:
                        os.remove(wal)
                    except OSError:
                        pass

    def _atexit_handler(self):
        """Emergency cleanup on process exit."""
        if not self._closed:
            self.close()


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    db = Database("test_memory.db")
    print(f"Stats: {db.get_stats()}")

    # Test insert
    eid = db.new_id()
    db.execute(
        "INSERT INTO events (id, type, description) VALUES (?, ?, ?)",
        (eid, "meeting", "Doctor appointment tomorrow at 10 AM"),
    )
    print(f"Inserted event: {eid}")
    print(f"Events: {db.fetch_all('SELECT * FROM events')}")
    print(f"Stats: {db.get_stats()}")

    # Cleanup test file
    os.remove("test_memory.db")
    print("Test passed ✅")
