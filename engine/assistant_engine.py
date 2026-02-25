"""
MemoryAssistantEngine — Central Orchestrator
===============================================
Pure Python engine that wraps all core modules.
No Flask. No network. No API. Just direct method calls.

Usage:
    from engine.assistant_engine import MemoryAssistantEngine

    engine = MemoryAssistantEngine()
    result = engine.process_text("I have a doctor appointment tomorrow at 10 AM")
    result = engine.process_audio("recording.wav")
    answer = engine.query("When is my appointment?")
    events = engine.get_upcoming_events()
"""

import os
import sys

# Ensure project root is on the path so core/ and storage/ imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from storage.repository import Repository
from core.transcriber import transcribe_audio
from core.summarizer import summarize, summarize_with_highlights
from core.event_extractor import extract_structured_events
from core.query_engine import QueryEngine
from core.reminder_manager import ReminderManager
from core import llm_engine
from diarization.diarizer import SpeakerDiarizer
from conversation.builder import ConversationBuilder
from speaker_identity.identity_manager import IdentityManager
from background.audio_worker import AudioWorker


class RepositoryAdapter:
    """
    Adapter that makes Repository look like MemoryManager.

    QueryEngine and ReminderManager expect a MemoryManager with:
      .get_all_events()  → list of dicts
      .search_events(keyword) → list of dicts

    This adapter wraps Repository to provide the same interface
    so we don't need to modify the existing modules.
    """

    def __init__(self, repo: Repository):
        self._repo = repo

    def get_all_events(self) -> list[dict]:
        """Return all events from SQLite."""
        return self._repo.get_all_events()

    def search_events(self, keyword: str) -> list[dict]:
        """Search events by keyword."""
        return self._repo.search_events(keyword)

    def count(self) -> int:
        """Count total events."""
        stats = self._repo.get_stats()
        return stats.get("events", 0)


class MemoryAssistantEngine:
    """
    Central orchestrator for the Conversational Memory Assistant.

    All processing is done locally via direct method calls.
    No network, no API, no Flask — pure Python engine.

    Public Methods:
        process_text(text, use_llm)     → dict (summary + events + highlights)
        process_audio(file_path, use_llm) → dict (transcription + summary + events)
        query(question, use_llm)        → dict (answer)
        get_events(type_filter, search) → list[dict]
        get_upcoming_events(minutes)    → dict (upcoming + alerts)
        get_llm_status()                → dict (status + models)
        get_stats()                     → dict (table counts)
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the engine with all modules.

        Args:
            db_path: Path to SQLite database. Defaults to memory.db in project root.
        """
        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "memory.db")

        # Storage layer
        self.repo = Repository(db_path)

        # Adapter for backward-compatible modules
        self._memory_adapter = RepositoryAdapter(self.repo)

        # Query & reminder engines (use adapter + repo for persistence)
        self.query_engine = QueryEngine(self._memory_adapter)
        self.reminder_mgr = ReminderManager(self._memory_adapter, repo=self.repo)

        # Auto-schedule reminders for existing events
        self.reminder_mgr.auto_schedule()

        # Speaker diarization + conversation builder
        self.diarizer = SpeakerDiarizer()
        self.conv_builder = ConversationBuilder()

        # Speaker identity mapping
        self.identity_mgr = IdentityManager(self.repo.db)

        print(f"[Engine] MemoryAssistantEngine ready")
        print(f"[Engine]   Database: {db_path}")
        print(f"[Engine]   Events in DB: {self._memory_adapter.count()}")
        print(f"[Engine]   Speaker profiles: {self.identity_mgr.profile_count}")
        print(f"[Engine]   Diarization: {'pyannote' if self.diarizer.is_available else 'fallback (single speaker)'}")

    # ── Process Text ───────────────────────────────────────────

    def process_text(self, text: str, use_llm: bool = False) -> dict:
        """
        Process conversation text through the full NLP pipeline.

        Pipeline: Text → Summarize → Extract Events → Store

        Args:
            text: Conversation text to process.
            use_llm: Whether to use LLM for enhanced processing.

        Returns:
            dict with keys: summary, highlights, events, conversation_id
        """
        if not text or not text.strip():
            return {"error": "Empty text provided"}

        print(f"[Engine] Processing text ({len(text)} chars, LLM={'ON' if use_llm else 'OFF'})...")

        # Step 1: Summarize
        summary_text = summarize(text, use_llm=use_llm)
        highlights = summarize_with_highlights(text)

        # Step 2: Extract events
        events = extract_structured_events(text, use_llm=use_llm)

        # Step 2b: Score events by importance (Phase Q)
        from core.memory_ranker import score_events, detect_patterns
        score_events(events)

        # Step 3: Store in database
        conv_id = self.repo.save_conversation(raw_text=text, source="text")
        saved_count = self.repo.save_events(conv_id, events)

        # Step 3b: Detect recurring patterns (Phase Q)
        detect_patterns(text, repo=self.repo)

        # Store summary
        key_points = [
            h["sentence"] for h in highlights if h.get("important")
        ]
        mode = "llm" if use_llm else "rule"
        self.repo.save_summary(conv_id, summary_text, key_points, mode)

        print(f"[Engine] Done: {saved_count} events saved, conv={conv_id[:8]}...")

        # Phase R: Apply simplified mode filtering
        display_summary = summary_text
        display_events = events
        try:
            from config import SIMPLIFIED_MODE, MAX_SUMMARY_POINTS, MIN_DISPLAY_IMPORTANCE
            if SIMPLIFIED_MODE:
                from core.reinforcement import simplify_summary, filter_events_by_importance
                display_summary = simplify_summary(summary_text, MAX_SUMMARY_POINTS)
                display_events = filter_events_by_importance(events, MIN_DISPLAY_IMPORTANCE)
        except ImportError:
            pass

        return {
            "conversation_id": conv_id,
            "summary": display_summary,
            "highlights": highlights,
            "events": display_events,
            "events_saved": saved_count,
            "mode": mode,
        }

    # ── Process Audio ──────────────────────────────────────────

    def process_audio(self, file_path: str, use_llm: bool = False) -> dict:
        """
        Process an audio file through the full pipeline.

        Pipeline: Audio -> Diarize -> Voice Fingerprint -> Transcribe
                       -> Build Conversation -> Summarize -> Extract Events -> Store

        Args:
            file_path: Path to the audio file (wav, mp3, m4a).
            use_llm: Whether to use LLM for enhanced processing.

        Returns:
            dict with keys: transcription, speakers, summary, highlights,
                            events, conversation_id
        """
        if not os.path.isfile(file_path):
            return {"error": f"Audio file not found: {file_path}"}

        print(f"[Engine] Processing audio: {file_path}")

        # Step 1: Speaker diarization (who spoke when)
        print("[Engine]   Step 1/6: Speaker diarization...")
        dia_segments = self.diarizer.diarize(file_path)

        # Step 2: Voice fingerprinting (auto-identify returning speakers)
        print("[Engine]   Step 2/6: Voice fingerprinting...")
        dia_segments = self.identity_mgr.auto_identify_speakers(
            dia_segments, file_path, repo=self.repo,
        )

        # Step 3: Transcribe audio -> text
        print("[Engine]   Step 3/6: Whisper transcription...")
        asr_result = transcribe_audio(file_path)
        text = asr_result["text"]

        if not text.strip():
            return {
                "transcription": "",
                "speakers": [],
                "summary": "No speech detected in the audio.",
                "highlights": [],
                "events": [],
                "events_saved": 0,
                "warning": "No speech detected",
            }

        # Step 4: Build structured conversation (merge diarization + ASR)
        print("[Engine]   Step 4/6: Building conversation...")
        conversation = self.conv_builder.build(
            dia_segments, asr_result, identity_manager=self.identity_mgr
        )
        speaker_text = self.conv_builder.build_text(conversation)
        num_speakers = len(set(s["speaker"] for s in conversation))
        print(f"[Engine]   Found {num_speakers} speaker(s), {len(conversation)} segments")

        # Step 5: Summarize (use speaker-labeled text for better context)
        print("[Engine]   Step 5/6: Summarizing...")
        summary_text = summarize(speaker_text if num_speakers > 1 else text, use_llm=use_llm)
        highlights = summarize_with_highlights(text)

        # Step 6: Extract events
        print("[Engine]   Step 6/6: Extracting events...")
        events = extract_structured_events(text, use_llm=use_llm)

        # Score events by importance (Phase Q)
        from core.memory_ranker import score_events, detect_patterns
        score_events(events)

        # Store in database
        conv_id = self.repo.save_conversation(
            raw_text=speaker_text, audio_path=file_path, source="audio"
        )
        self.repo.save_segments(conv_id, conversation)
        saved_count = self.repo.save_events(conv_id, events)

        # Detect recurring patterns (Phase Q)
        detect_patterns(text, repo=self.repo)

        key_points = [
            h["sentence"] for h in highlights if h.get("important")
        ]
        mode = "llm" if use_llm else "rule"
        self.repo.save_summary(conv_id, summary_text, key_points, mode)

        print(f"[Engine] Done: {num_speakers} speakers, {saved_count} events")

        return {
            "conversation_id": conv_id,
            "transcription": text,
            "speakers": conversation,
            "num_speakers": num_speakers,
            "summary": summary_text,
            "highlights": highlights,
            "events": events,
            "events_saved": saved_count,
            "mode": mode,
        }

    # ── Query ──────────────────────────────────────────────────

    def query(self, question: str, use_llm: bool = False) -> dict:
        """
        Answer a natural language question about stored memories.

        Args:
            question: The user's question (e.g., "When is my appointment?")
            use_llm: Whether to use LLM for enhanced answering.

        Returns:
            dict with keys: question, answer
        """
        if not question or not question.strip():
            return {"question": question, "answer": "Please ask a question."}

        print(f"[Engine] Query: '{question}'")
        answer = self.query_engine.query(question, use_llm=use_llm)
        print(f"[Engine] Answer: {answer[:100]}...")

        return {
            "question": question,
            "answer": answer,
        }

    def get_memory_count(self) -> dict:
        """Debug method to get total memory items."""
        return self.repo.get_memory_stats()

    # ── Events ─────────────────────────────────────────────────

    def get_events(
        self, type_filter: str = None, search: str = None
    ) -> list[dict]:
        """
        Get all stored events, optionally filtered.

        Args:
            type_filter: Filter by event type (meeting, medication, task, visit).
            search: Search keyword.

        Returns:
            List of event dicts.
        """
        return self.repo.get_all_events(
            type_filter=type_filter, search=search
        )

    # ── Upcoming Events / Reminders ────────────────────────────

    def get_upcoming_events(self, minutes: int = 60) -> dict:
        """
        Get upcoming events and alerts.

        Args:
            minutes: Look-ahead window in minutes.

        Returns:
            dict with keys: upcoming, alerts, total_events
        """
        upcoming = self.reminder_mgr.get_upcoming_events(minutes=minutes)
        alerts = self.reminder_mgr.check_due_events(window_minutes=5)

        return {
            "upcoming": upcoming,
            "alerts": alerts,
            "total_events": self._memory_adapter.count(),
        }

    # ── LLM Status ─────────────────────────────────────────────

    def get_llm_status(self) -> dict:
        """Check if local LLM (Ollama) is available."""
        available = llm_engine.is_available()
        models = llm_engine.get_models() if available else []

        return {
            "status": "online" if available else "offline",
            "models": models,
            "default_model": llm_engine.DEFAULT_MODEL,
        }

    # ── Statistics ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get engine and database statistics."""
        db_stats = self.repo.get_stats()
        llm_status = self.get_llm_status()

        return {
            "database": db_stats,
            "llm": llm_status,
            "speaker_profiles": self.identity_mgr.profile_count,
            "diarization": "pyannote" if self.diarizer.is_available else "fallback",
            "version": "2.2.0",
            "architecture": "offline-engine",
        }

    def get_resource_stats(self) -> dict:
        """
        Get resource usage statistics for debugging.

        Returns approximate info about loaded models, active threads,
        audio buffer state, and estimated memory usage.
        No external profiling libraries required.
        """
        import sys
        import threading

        # Loaded models
        models = {}

        # Whisper model cache
        try:
            from core.transcriber import _model_cache
            for size, model in _model_cache.items():
                models[f"whisper_{size}"] = {
                    "loaded": True,
                    "approx_mb": round(sys.getsizeof(model) / 1024 / 1024, 1),
                }
        except ImportError:
            pass

        # Diarization model
        models["pyannote"] = {"loaded": self.diarizer.is_available}

        # Embedding model
        try:
            from core.semantic_search import EmbeddingSearch
            emb = EmbeddingSearch()
            models["sentence_transformer"] = {"loaded": emb.is_available}
        except Exception:
            models["sentence_transformer"] = {"loaded": False}

        # Active threads
        active_threads = threading.active_count()
        thread_names = [t.name for t in threading.enumerate()]

        # Audio buffer info
        buffer_info = {}
        if hasattr(self, '_worker') and self._worker:
            src = self._worker.audio_source
            if src and hasattr(src, 'get_stats'):
                buffer_info = src.get_stats()

        # Config
        try:
            from config import get_config_summary
            config_summary = get_config_summary()
        except ImportError:
            config_summary = {}

        return {
            "models": models,
            "active_threads": active_threads,
            "thread_names": thread_names,
            "audio_buffer": buffer_info,
            "config": config_summary,
        }

    def get_urgent_items(self, hours: int = 24) -> list[dict]:
        """
        Get events that are urgent (medication/appointments within `hours`).

        Returns list of event dicts with urgent_flag = True.
        """
        from core.memory_ranker import get_urgent_items
        return get_urgent_items(self.repo, hours=hours)

    def get_memory_patterns(self, min_frequency: int = 1) -> list[dict]:
        """Get recurring conversation patterns."""
        return self.repo.get_patterns(min_frequency=min_frequency)

    # ── Cognitive Reinforcement (Phase R) ─────────────────────

    def get_reinforcement_items(self) -> list[dict]:
        """
        Get critical events needing re-display to the user.

        Returns events with importance >= 5 not shown within
        REINFORCEMENT_INTERVAL_HOURS.
        """
        from core.reinforcement import get_reinforcement_items
        try:
            from config import REINFORCEMENT_INTERVAL_HOURS
        except ImportError:
            REINFORCEMENT_INTERVAL_HOURS = 12
        return get_reinforcement_items(self.repo, interval_hours=REINFORCEMENT_INTERVAL_HOURS)

    def mark_item_shown(self, event_id: str) -> None:
        """Record that a critical event was shown to the user."""
        from core.reinforcement import mark_shown
        mark_shown(self.repo, event_id)

    def check_escalations(self) -> list[dict]:
        """
        Check for missed/overdue events and escalate.

        Returns list of newly escalated events.
        """
        from core.reinforcement import check_escalation
        try:
            from config import ESCALATION_MAX_LEVEL
        except ImportError:
            ESCALATION_MAX_LEVEL = 3
        return check_escalation(self.repo, max_level=ESCALATION_MAX_LEVEL)

    def generate_daily_brief(self) -> dict:
        """
        Generate a calm, structured daily summary.

        Returns dict with greeting, urgent_items, patterns,
        summary_text, and closing.
        """
        from core.reinforcement import generate_daily_brief
        return generate_daily_brief(self.repo)

    def set_config_flag(self, key: str, value: bool) -> dict:
        """
        Toggle a runtime config flag.

        Supported keys:
          - SIMPLIFIED_MODE
          - LOW_RESOURCE_MODE
          - DEBUG_TIMING

        Args:
            key: Config flag name (case-insensitive).
            value: True to enable, False to disable.

        Returns:
            dict with status, key, and new value.
        """
        import config
        key_upper = key.upper()
        allowed = {"SIMPLIFIED_MODE", "LOW_RESOURCE_MODE", "DEBUG_TIMING"}
        if key_upper not in allowed:
            return {"error": f"Unknown flag '{key}'. Allowed: {sorted(allowed)}"}
        setattr(config, key_upper, value)
        # Update dependent values when SIMPLIFIED_MODE changes
        if key_upper == "SIMPLIFIED_MODE":
            config.MAX_SUMMARY_POINTS = 2 if value else 5
            config.MIN_DISPLAY_IMPORTANCE = 3 if value else 0
        return {"status": "ok", "key": key_upper, "value": value}

    # ── Speaker Identity ──────────────────────────────────────

    def assign_speaker_label(self, raw_label: str, display_name: str) -> dict:
        """
        Map a raw diarization label to a display name.

        Args:
            raw_label: The raw label from diarization (e.g., "SPEAKER_00").
            display_name: Human-readable name (e.g., "Dr. Smith").

        Returns:
            dict with status and current profiles count.
        """
        self.identity_mgr.assign_label(raw_label, display_name)
        return {
            "status": "ok",
            "speaker_label": raw_label,
            "display_name": display_name,
            "total_profiles": self.identity_mgr.profile_count,
        }

    def get_speaker_profiles(self) -> list[dict]:
        """
        Get all speaker identity mappings.

        Returns:
            List of dicts with: speaker_label, display_name, created_at
        """
        return self.identity_mgr.get_all_profiles()

    def remove_speaker_profile(self, raw_label: str) -> dict:
        """
        Remove a speaker identity mapping.

        Args:
            raw_label: The raw label to unmap.

        Returns:
            dict with status.
        """
        removed = self.identity_mgr.remove_profile(raw_label)
        return {
            "status": "ok" if removed else "not_found",
            "speaker_label": raw_label,
            "total_profiles": self.identity_mgr.profile_count,
        }

    # ── Backup & Restore ─────────────────────────────────────────

    def create_backup(self, destination_path: str) -> dict:
        """
        Create a secure backup of the entire database.

        Includes all conversations, events, voiceprints, embeddings,
        reminders, and summaries. Encryption is preserved.

        Args:
            destination_path: File path for the .wbbak backup archive.

        Returns:
            dict with: status, path, size_bytes, sha256, timestamp
        """
        from storage.backup_manager import BackupManager
        mgr = BackupManager(self.repo.db)
        return mgr.create_backup(destination_path)

    def restore_backup(self, source_path: str) -> dict:
        """
        Restore a database from a backup archive.

        WARNING: This replaces the current database!

        Steps:
          1. Stop background worker (if running)
          2. Verify backup integrity
          3. Replace current database
          4. Reinitialize engine

        Args:
            source_path: Path to the .wbbak backup file.

        Returns:
            dict with: status, restored_from, timestamp, needs_restart
        """
        # Stop background worker before restore
        if hasattr(self, '_worker') and self._worker and self._worker.is_running():
            self._worker.stop()
            print("[Engine] Background worker stopped for restore")

        from storage.backup_manager import BackupManager
        mgr = BackupManager(self.repo.db)
        result = mgr.restore_backup(source_path)

        if result.get("status") == "success":
            # Reinitialize engine with same db_path
            print("[Engine] Reinitializing after restore...")
            self.__init__(self.repo.db.db_path)
            result["needs_restart"] = False  # Engine already restarted

        return result

    def verify_backup(self, file_path: str) -> dict:
        """
        Verify the integrity of a backup file.

        Args:
            file_path: Path to the .wbbak file.

        Returns:
            dict with: valid, manifest, sha256_match, errors
        """
        from storage.backup_manager import BackupManager
        mgr = BackupManager(self.repo.db)
        return mgr.verify_backup(file_path)

    def list_backups(self, directory: str) -> list[dict]:
        """
        List all backup files in a directory.

        Args:
            directory: Directory to scan.

        Returns:
            List of dicts with: path, filename, size_bytes, modified.
        """
        from storage.backup_manager import BackupManager
        mgr = BackupManager(self.repo.db)
        return mgr.list_backups(directory)

    # ── Audio Source Management ──────────────────────────────────

    def set_audio_source(self, source_type: str, **config) -> dict:
        """
        Switch the audio source at runtime.

        Args:
            source_type: "microphone", "bluetooth", or "file"
            config: Source-specific settings:
                bluetooth: device_name (str)
                file: file_path (str), loop (bool)
                microphone: device (int|None)

        Returns:
            dict with status and source info.
        """
        # Stop worker if running
        if hasattr(self, '_worker') and self._worker and self._worker.is_running():
            self._worker.stop()

        source_type = source_type.lower()

        if source_type == "bluetooth":
            from audio.bluetooth_source import BluetoothAudioSource
            source = BluetoothAudioSource(
                device_name=config.get("device_name", "Bluetooth Device"),
                sample_rate=config.get("sample_rate", 16000),
            )
            self._bt_source = source  # Keep reference for push_bluetooth_audio

        elif source_type == "file":
            from audio.file_source import FileSource
            file_path = config.get("file_path")
            if not file_path:
                return {"status": "error", "error": "file_path required"}
            source = FileSource(
                file_path=file_path,
                sample_rate=config.get("sample_rate", 16000),
            )

        elif source_type == "microphone":
            from audio.microphone import MicrophoneSource
            source = MicrophoneSource(
                sample_rate=config.get("sample_rate", 16000),
                device=config.get("device"),
            )

        else:
            return {"status": "error", "error": f"Unknown source type: {source_type}"}

        # Inject into worker
        worker = self._ensure_worker()
        worker.audio_source = source
        print(f"[Engine] Audio source → {type(source).__name__}")

        return {
            "status": "ok",
            "source_type": source_type,
            "source": type(source).__name__,
        }

    def push_bluetooth_audio(self, pcm_data: bytes) -> dict:
        """
        Push raw PCM audio from Flutter (Bluetooth device).

        Called by MethodChannel when BLE audio frames arrive.

        Args:
            pcm_data: Raw int16 little-endian PCM bytes.

        Returns:
            dict with samples_written count.
        """
        if not hasattr(self, '_bt_source') or self._bt_source is None:
            return {"status": "error", "error": "No Bluetooth source active"}

        n = self._bt_source.push_audio(pcm_data)
        return {"status": "ok", "samples_written": n}

    def get_audio_source_info(self) -> dict:
        """Get info about the current audio source."""
        if not hasattr(self, '_worker') or self._worker is None:
            return {"source": "MicrophoneSource (default)", "active": False}

        src = self._worker.audio_source
        if src is None:
            return {"source": "MicrophoneSource (default)", "active": False}

        info = {
            "source": type(src).__name__,
            "active": src.is_active,
            "sample_rate": src.sample_rate,
        }

        # Add Bluetooth-specific stats
        if hasattr(src, 'get_stats'):
            info["stats"] = src.get_stats()

        return info

    # ── Recording & Background Listening ─────────────────────────

    def _ensure_worker(self, **kwargs):
        """Get or create the AudioWorker singleton."""
        if not hasattr(self, '_worker') or self._worker is None:
            self._worker = AudioWorker(engine=self, **kwargs)
        return self._worker

    def start_recording(self, **kwargs) -> dict:
        """
        Start recording a full conversation session.

        Records continuously until stop_recording() is called.
        The complete recording is saved and processed through the
        full pipeline (diarize → transcribe → summarize → extract).

        Returns:
            dict with recording status.
        """
        worker = self._ensure_worker(**kwargs)
        if worker.is_running():
            return {"status": "already_running", **worker.status()}

        return worker.start_recording()

    def stop_recording(self) -> dict:
        """
        Stop recording and process the full conversation.

        Saves the recording to recordings/ directory, then runs:
        diarization → transcription → summarization → event extraction → storage.

        Returns:
            dict with file path, duration, transcription, events, summary.
        """
        if not hasattr(self, '_worker') or not self._worker:
            return {"status": "not_recording"}

        return self._worker.stop_recording(process=True)

    def start_background_listening(self, **kwargs) -> dict:
        """
        Start VAD-based background listening (hands-free mode).

        Auto-detects speech, records chunks, and processes them.
        For wearable/hands-free use cases.

        Returns:
            dict with worker status.
        """
        worker = self._ensure_worker(**kwargs)
        if worker.is_running():
            return {"status": "already_running", **worker.status()}

        worker.start_vad_listening()
        return {"status": "started", **worker.status()}

    def stop_background_listening(self) -> dict:
        """Stop the VAD background listener."""
        if not hasattr(self, '_worker') or not self._worker:
            return {"status": "not_running"}

        was_running = self._worker.is_running()
        self._worker.stop()
        return {
            "status": "stopped" if was_running else "not_running",
            **self._worker.status(),
        }

    def get_worker_status(self) -> dict:
        """Get worker status (works for both session and VAD modes)."""
        if not hasattr(self, '_worker') or not self._worker:
            return {"running": False, "status": "no_worker"}
        return self._worker.status()

    def list_recordings(self) -> list[dict]:
        """List all saved conversation recordings."""
        worker = self._ensure_worker()
        return worker.list_recordings()


# ── CLI Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    engine = MemoryAssistantEngine()

    print("\n" + "=" * 60)
    print("  Engine Stats")
    print("=" * 60)
    import json
    print(json.dumps(engine.get_stats(), indent=2))

    # Process sample text
    print("\n" + "=" * 60)
    print("  Processing Sample Text")
    print("=" * 60)
    sample = (
        "I have a doctor appointment tomorrow at 10 AM. "
        "Don't forget to take your medicine after breakfast. "
        "We need to call the pharmacy to refill the prescription. "
        "Your son David is visiting this weekend."
    )
    result = engine.process_text(sample)
    print(f"\n  Summary: {result['summary'][:100]}...")
    print(f"  Events found: {len(result['events'])}")
    print(f"  Events saved: {result['events_saved']}")

    # Query
    print("\n" + "=" * 60)
    print("  Query Test")
    print("=" * 60)
    questions = [
        "When is my doctor appointment?",
        "What medicine do I need to take?",
        "Who is visiting this weekend?",
    ]
    for q in questions:
        answer = engine.query(q)
        print(f"\n  Q: {q}")
        print(f"  A: {answer['answer']}")

    print("\n" + "=" * 60)
    print(f"  Final Stats: {engine.get_stats()['database']}")
    print("=" * 60)
