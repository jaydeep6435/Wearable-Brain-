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
import json
from datetime import datetime

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
        self.reminder_mgr = ReminderManager(self._memory_adapter)

        # Auto-schedule reminders for existing events
        try:
            self.repo.auto_schedule_reminders(lead_minutes=15)
        except Exception as e:
            print(f"[Engine] Reminder auto-schedule skipped: {e}")

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
        try:
            from core.memory_ranker import score_events, detect_patterns

            score_events(events)
        except Exception as e:
            print(f"[Engine]   memory_ranker unavailable (text path): {e}")

            def detect_patterns(*args, **kwargs):
                return None

        # Step 2c: LLM Event Refinement (Intelligence Mode)
        try:
            from core.llm_engine import refine_events_llm, is_available as llm_available
            if use_llm and llm_available():
                refined = refine_events_llm(text, events)
                if refined:
                    print(f"[Engine]   LLM found {len(refined)} additional events")
                    events.extend(refined)
                    score_events(refined)
        except Exception as e:
            print(f"[Engine]   LLM refinement skipped: {e}")

        # Step 2d: LLM Validation Layer (Hybrid AI)
        llm_validation = None
        try:
            from core.llm_engine import validate_memory, is_available as llm_available
            if use_llm and llm_available():
                print("[Engine]   Step 2d: LLM memory validation...")
                llm_validation = validate_memory(text, events)
                if llm_validation:
                    # Use validated events if available
                    validated = llm_validation.get("validated_events", [])
                    if validated:
                        print(f"[Engine]   LLM validated {len(validated)} events")
                        # Merge validated info into existing events
                        for ve in validated:
                            ve.setdefault("validated", True)
                            score_events([ve])

                    # Log risk flags
                    risks = llm_validation.get("risk_flags", [])
                    if risks:
                        print(f"[Engine]   ⚠ Risk flags: {risks}")

                    # Use clean summary if available
                    clean = llm_validation.get("clean_summary", "")
                    if clean:
                        summary_text = clean
                        print(f"[Engine]   Using LLM clean summary")
        except Exception as e:
            print(f"[Engine]   LLM validation skipped: {e}")

        # Step 3: Store in database
        conv_id = self.repo.save_conversation(raw_text=text, source="text")
        saved_count = self.repo.save_events(conv_id, events)

        # Step 3b: Detect recurring patterns (Phase Q)
        detect_patterns(text, repo=self.repo)

        # Store summary
        key_points = [
            h["sentence"] for h in highlights if h.get("important")
        ]
        mode = "llm" if use_llm or llm_validation else "rule"
        self.repo.save_summary(conv_id, summary_text, key_points, mode)

        # Step 4: Auto-schedule reminders
        reminder_count = self.repo.auto_schedule_reminders(lead_minutes=15)
        if reminder_count:
            print(f"[Engine]   Created {reminder_count} new reminders")

        print(f"[Engine] Done: {saved_count} events saved, conv={conv_id[:8]}...")

        # Return Intelligence Mode result
        return self._finalize_processing_result(conv_id, text, saved_count)

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

        print(f"[Engine] ━━━ processAudio ━━━━━━━━━━━━━━━━━━━")
        print(f"[Engine]   File: {file_path}")
        print(f"[Engine]   DB path: {self.repo.db.db_path}")

        # Step 1: Transcribe + diarize
        print("[Engine]   Step 1/6: Whisper transcription + diarization...")
        asr_result = transcribe_audio(file_path, enable_diarization=True)
        text = asr_result["text"]

        # Build diarization segments from ASR speaker-labeled spans.
        dia_segments = [
            {
                "speaker": seg.get("speaker", "Speaker 1"),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
            }
            for seg in asr_result.get("segments", [])
        ]

        # Step 2: Voice fingerprinting / label normalization
        print("[Engine]   Step 2/6: Voice fingerprinting...")
        dia_segments = self.identity_mgr.auto_identify_speakers(
            dia_segments, file_path, repo=self.repo,
        )

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

        # Step 3: Build structured conversation (merge diarization + ASR)
        print("[Engine]   Step 3/6: Building conversation...")
        conversation = self.conv_builder.build(
            dia_segments, asr_result, identity_manager=self.identity_mgr
        )
        speaker_text = self.conv_builder.build_text(conversation)
        num_speakers = max(1, len(set(s["speaker"] for s in conversation)))
        print(f"[Engine]   Found {num_speakers} speaker(s), {len(conversation)} segments")

        # Step 5: Summarize (use speaker-labeled text for better context)
        print("[Engine]   Step 5/8: Summarizing...")
        summary_text = summarize(speaker_text if num_speakers > 1 else text, use_llm=use_llm)
        highlights = summarize_with_highlights(text)

        # Step 6: Extract events
        print("[Engine]   Step 6/8: Extracting events...")
        events = extract_structured_events(text, use_llm=use_llm)

        # Score events by importance (Phase Q)
        try:
            from core.memory_ranker import score_events, detect_patterns

            score_events(events)
        except Exception as e:
            print(f"[Engine]   memory_ranker unavailable (audio path): {e}")

            def detect_patterns(*args, **kwargs):
                return None

        # Step 7: LLM Event Refinement (Intelligence Mode)
        print("[Engine]   Step 7/8: LLM event refinement...")
        try:
            from core.llm_engine import refine_events_llm, is_available as llm_available
            if use_llm and llm_available():
                refined = refine_events_llm(text, events)
                if refined:
                    print(f"[Engine]   LLM found {len(refined)} additional events/warnings")
                    events.extend(refined)
                    score_events(refined)  # Score the new events too
        except Exception as e:
            print(f"[Engine]   LLM refinement skipped: {e}")

        # Step 7b: LLM Validation Layer (Hybrid AI)
        llm_validation = None
        try:
            from core.llm_engine import validate_memory, is_available as llm_available
            if use_llm and llm_available():
                print("[Engine]   Step 7b: LLM memory validation...")
                llm_validation = validate_memory(text, events)
                if llm_validation:
                    validated = llm_validation.get("validated_events", [])
                    if validated:
                        print(f"[Engine]   LLM validated {len(validated)} events")
                        for ve in validated:
                            ve.setdefault("validated", True)
                            score_events([ve])

                    risks = llm_validation.get("risk_flags", [])
                    if risks:
                        print(f"[Engine]   ⚠ Risk flags: {risks}")

                    clean = llm_validation.get("clean_summary", "")
                    if clean:
                        summary_text = clean
                        print(f"[Engine]   Using LLM clean summary")
        except Exception as e:
            print(f"[Engine]   LLM validation skipped: {e}")

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
        mode = "llm" if use_llm or llm_validation else "rule"
        self.repo.save_summary(conv_id, summary_text, key_points, mode)

        # Step 8: Auto-schedule reminders
        print("[Engine]   Step 8/8: Auto-scheduling reminders...")
        reminder_count = self.repo.auto_schedule_reminders(lead_minutes=15)
        if reminder_count:
            print(f"[Engine]   Created {reminder_count} new reminders")

        # Finalize result (Intelligence Mode)
        result = self._finalize_processing_result(conv_id, text, saved_count, num_speakers)
        
        print(f"[Engine]   Transcript length: {len(text)}")
        print(f"[Engine]   Events saved: {saved_count}")
        print(f"[Engine]   Urgent items: {len(result.get('urgent_items', []))}")
        print(f"[Engine] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return result

    def _finalize_processing_result(self, conv_id: str, raw_text: str, events_saved: int, num_speakers: int = 2) -> dict:
        """Helper to build a unified Intelligence Mode result dict."""
        conv = self.repo.get_conversation(conv_id) or {}
        segments = conv.get("segments", [])
        speaker_names = list(dict.fromkeys(
            s.get("speaker", "Speaker 1") for s in segments if s.get("speaker")
        ))
        summary_obj = conv.get("summary") or {}
        summary_text = summary_obj.get("summary", "No summary generated.")
        key_points = json.loads(summary_obj.get("key_points", "[]"))

        # Find urgent and important items
        try:
            from core.memory_ranker import get_urgent_items

            urgent_items = get_urgent_items(self.repo, hours=48)  # 48h lookahead
        except Exception as e:
            print(f"[Engine]   memory_ranker unavailable (finalize path): {e}")
            urgent_items = []
        
        # High importance items (importance >= 4)
        important_events = [e for e in conv.get("events", []) if e.get("importance_score", 0) >= 4]

        return {
            "status": "ok",
            "conversation_id": conv_id,
            "transcription": raw_text,
            "transcript_length": len(raw_text),
            "segments": segments,
            "speakers": speaker_names,
            "num_speakers": max(num_speakers, len(speaker_names) or 1),
            "speakers_detected": num_speakers,
            "summary": summary_text,
            "key_points": key_points,
            "events": conv.get("events", []),
            "events_saved": events_saved,
            "urgent_items": urgent_items[:5],      # Limit for UI
            "important_items": important_events[:5], # Limit for UI
            "timestamp": datetime.now().isoformat(),
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

    # ── LLM Validation Layer ───────────────────────────────────

    def llm_validate_memory(self, transcript: str, extracted_events: list) -> dict:
        """
        Standalone LLM validation of extracted events.

        Args:
            transcript: Original conversation text.
            extracted_events: Events from rule-based extraction.

        Returns:
            Validation result with validated_events, refined_reminders,
            risk_flags, and clean_summary.
        """
        try:
            from core.llm_engine import validate_memory, is_available
            if not is_available():
                return {
                    "status": "llm_unavailable",
                    "message": "LLM not available. Using rule-based only.",
                    "validated_events": extracted_events,
                }

            result = validate_memory(transcript, extracted_events)
            if result:
                print(f"[Engine] LLM validated {len(result.get('validated_events', []))} events")
                print(f"[Engine] Risk flags: {result.get('risk_flags', [])}")
                return {"status": "ok", **result}
            else:
                return {
                    "status": "llm_error",
                    "message": "LLM returned no result.",
                    "validated_events": extracted_events,
                }
        except Exception as e:
            print(f"[Engine] LLM validation error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "validated_events": extracted_events,
            }

    # ── Chat With Memory (LLM-Powered) ─────────────────────────

    def chat_with_memory(self, question: str) -> dict:
        """
        LLM-powered conversational memory Q&A.

        Flow:
          1. Retrieve relevant memories (embedding + keyword search)
          2. Rank by importance + recency
          3. Build structured context (top 5)
          4. Send to LLM with Alzheimer-specific system prompt
          5. Return structured answer

        Args:
            question: Patient's natural language question.

        Returns:
            dict with answer, related_events, confidence.
        """
        if not question or not question.strip():
            return {
                "answer": "Please ask me a question about your day or memories.",
                "related_events": [],
                "confidence": "low",
            }

        print(f"[Engine] Chat: '{question}'")

        # Step 1: Retrieve relevant memories
        memory_context = self._build_memory_context(question)

        if not memory_context.strip():
            return {
                "answer": "I don't have any memories stored yet. Try recording a conversation first.",
                "related_events": [],
                "confidence": "low",
            }

        # Step 2: Call LLM
        try:
            from core.llm_engine import chat_with_memory as llm_chat, is_available
            if not is_available():
                # Fallback to rule-based query
                print("[Engine]   LLM unavailable, falling back to rule-based")
                rule_answer = self.query_engine.query(question, use_llm=False)
                return {
                    "answer": rule_answer,
                    "related_events": [],
                    "confidence": "medium",
                    "mode": "rule-based",
                }

            result = llm_chat(question, memory_context)
            if result:
                result["mode"] = "llm"
                print(f"[Engine]   Chat answer ({result.get('confidence', '?')}): {result['answer'][:80]}")
                return result

        except Exception as e:
            print(f"[Engine]   Chat error: {e}")

        # Final fallback
        rule_answer = self.query_engine.query(question, use_llm=False)
        return {
            "answer": rule_answer,
            "related_events": [],
            "confidence": "low",
            "mode": "fallback",
        }

    def _build_memory_context(self, question: str, top_k: int = 5) -> str:
        """
        Build structured memory context for LLM chat.

        Uses embedding search + importance ranking + recency boost
        to find the most relevant memories.
        """
        context_parts = []

        # Get all events and summaries for search
        all_events = self._memory_adapter.get_all_events()
        if not all_events:
            return ""

        # Try embedding-based semantic search first
        try:
            from core.semantic_search import EmbeddingSearch
            emb = EmbeddingSearch()
            if emb.is_available():
                results = emb.search(question, all_events, top_k=top_k)
                if results:
                    # Re-rank with importance + recency
                    from core.memory_ranker import rank_results
                    ranked = rank_results(results)
                    for r in ranked[:top_k]:
                        doc = r["document"]
                        context_parts.append(self._format_memory_entry(doc))
        except Exception as e:
            print(f"[Engine]   Embedding search failed: {e}")

        # Fallback: keyword search if embedding returned nothing
        if not context_parts:
            keywords = question.lower().split()
            for kw in keywords:
                if len(kw) > 3:  # Skip short words
                    matches = self._memory_adapter.search_events(kw)
                    for m in matches[:top_k]:
                        entry = self._format_memory_entry(m)
                        if entry not in context_parts:
                            context_parts.append(entry)

        # Also add recent summaries
        try:
            recent = self.repo.get_recent_summaries(limit=3)
            for s in recent:
                summary_text = s.get("summary", "")
                if summary_text:
                    ts = s.get("recorded_at", "unknown date")
                    context_parts.append(f"[Summary from {ts}]: {summary_text}")
        except Exception:
            pass

        return "\n".join(context_parts[:top_k + 3])

    def _format_memory_entry(self, event: dict) -> str:
        """Format a single memory event for LLM context."""
        parts = []
        etype = event.get("type", "event")
        desc = event.get("description", event.get("text", ""))
        date = event.get("parsed_date", event.get("raw_date", ""))
        time = event.get("parsed_time", event.get("time", ""))
        person = event.get("person", "")
        importance = event.get("importance_score", 0)
        recorded = event.get("recorded_at", "")

        entry = f"[{etype.upper()}]"
        if date:
            entry += f" Date: {date}"
        if time:
            entry += f" Time: {time}"
        if person:
            entry += f" Person: {person}"
        entry += f" — {desc}"
        if importance >= 4:
            entry += f" [IMPORTANT]"
        if recorded:
            entry += f" (recorded: {recorded})"

        return entry

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
        try:
            from core.memory_ranker import get_urgent_items

            return get_urgent_items(self.repo, hours=hours)
        except Exception as e:
            print(f"[Engine]   memory_ranker unavailable (urgent path): {e}")
            return []

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
        print("[Engine] ━━━ startListening ━━━━━━━━━━━━━━━━━━")
        print(f"[Engine]   DB path: {self.repo.db.db_path}")
        worker = self._ensure_worker(**kwargs)
        if worker.is_running():
            print("[Engine]   Already recording!")
            return {"status": "already_running", **worker.status()}

        result = worker.start_recording()
        print(f"[Engine]   ✓ Recording started: {result.get('status')}")
        print("[Engine] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return result

    def stop_recording(self) -> dict:
        """
        Stop recording and process the full conversation.

        Saves the recording to recordings/ directory, then runs:
        diarization → transcription → summarization → event extraction → storage.

        Returns:
            dict with file path, duration, transcription, events, summary.
        """
        print("[Engine] ━━━ stopListening ━━━━━━━━━━━━━━━━━━━")
        if not hasattr(self, '_worker') or not self._worker:
            print("[Engine]   ⚠ No worker — not recording")
            return {"status": "not_recording"}

        result = self._worker.stop_recording(process=True)
        transcript = result.get('transcription', '')
        events_saved = result.get('events_saved', 0)
        conv_id = result.get('conversation_id', 'N/A')
        print(f"[Engine]   Transcript length: {len(transcript)}")
        print(f"[Engine]   Events saved: {events_saved}")
        print(f"[Engine]   Conversation ID: {conv_id}")
        stats = self.repo.get_memory_stats()
        print(f"[Engine]   DB counts: {stats}")
        print(f"[Engine]   DB path: {self.repo.db.db_path}")
        print("[Engine] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return result

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

    # ── Conversational Memory (Intelligence Mode) ────────────────

    def chat_with_memory(self, question: str) -> dict:
        """
        Memory-aware LLM chat.
        Retrieves relevant context, ranks it, and answers using LLM.

        Returns:
            dict with answer, related_events, and confidence.
        """
        print(f"[Engine] ━━━ chat_with_memory ━━━━━━━━━━━━━━━")
        print(f"[Engine]   Question: {question}")
        
        from core.query_engine import QueryEngine
        query_engine = QueryEngine(self.repo)
        
        # 1. Semantic search + ranking
        # This uses EmbeddingSearch (if configured) or TF-IDF as fallback
        semantic_results = query_engine._semantic_search(question, top_k=5)
        
        # 2. Extract best context
        context_parts = []
        related_events = []
        
        if semantic_results:
            print(f"[Engine]   Found {len(semantic_results)} semantic matches")
            for sr in semantic_results:
                doc = sr["document"]
                # Build context line
                desc = doc.get("description") or doc.get("summary") or doc.get("text", "")
                date = doc.get("parsed_date") or doc.get("raw_date") or ""
                etype = doc.get("type", "event")
                context_parts.append(f"- [{etype.upper()}] {desc} (Date: {date})")
                
                # If it's an event, add to related list for UI cards
                if etype in ("meeting", "task", "medication", "event"):
                    related_events.append(doc)
        else:
            print("[Engine]   No semantic matches found")

        context = "\n".join(context_parts) if context_parts else "No relevant memories found."
        
        # 3. Answer via LLM
        try:
            from core.llm_engine import answer_query_llm, is_available
            if is_available():
                print("[Engine]   Generating LLM answer...")
                answer = answer_query_llm(question, context)
                confidence = "high" if context_parts else "low"
            else:
                print("[Engine]   LLM unavailable, falling back to rule-based")
                answer = query_engine.query(question, use_llm=False)
                confidence = "medium"
        except Exception as e:
            print(f"[Engine]   Chat error: {e}")
            answer = "I'm having trouble accessing my intelligence module right now."
            confidence = "none"

        print(f"[Engine]   Confidence: {confidence}")
        print(f"[Engine] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return {
            "answer": answer,
            "related_events": related_events[:3],
            "confidence": confidence,
            "context_used": len(context_parts) > 0
        }

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
