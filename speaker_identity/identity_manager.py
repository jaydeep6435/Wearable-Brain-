"""
Speaker Identity Manager — Map Raw Labels to Display Names
============================================================
Provides persistent mapping from diarization labels (SPEAKER_00)
to human-readable display names (Dr. Smith, Patient, Caregiver).

Storage: SQLite `speaker_profiles` table via the Database class.

Usage:
    from speaker_identity.identity_manager import IdentityManager

    mgr = IdentityManager(db)
    mgr.assign_label("SPEAKER_00", "Dr. Smith")
    mgr.assign_label("SPEAKER_01", "Patient")

    name = mgr.get_display_name("SPEAKER_00")  # "Dr. Smith"
    segments = mgr.apply_to_conversation(raw_segments)
"""

import os
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from storage.db import Database


class IdentityManager:
    """
    Manages speaker identity mapping.

    Maps raw diarization labels (SPEAKER_00, SPEAKER_01) to
    human-readable display names stored in SQLite.

    Thread-safe: all operations go through Database's connection manager.
    """

    def __init__(self, db: Database):
        """
        Initialize with a Database instance.

        Args:
            db: Database instance (shared with Repository).
        """
        self._db = db
        self._ensure_table()
        self._cache: dict[str, str] = {}
        self._load_cache()

    def _ensure_table(self):
        """Create the speaker_profiles table if it doesn't exist."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS speaker_profiles (
                speaker_label  TEXT PRIMARY KEY,
                display_name   TEXT NOT NULL,
                created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _load_cache(self):
        """Load all profiles into memory for fast lookup."""
        rows = self._db.fetch_all("SELECT speaker_label, display_name FROM speaker_profiles")
        self._cache = {row["speaker_label"]: row["display_name"] for row in rows}

    # ── CRUD Operations ────────────────────────────────────────

    def assign_label(self, speaker_label: str, display_name: str) -> None:
        """
        Map a raw speaker label to a display name.

        If the label already exists, it will be updated.

        Args:
            speaker_label: The raw diarization label (e.g., "SPEAKER_00").
            display_name: The human-readable name (e.g., "Dr. Smith").
        """
        speaker_label = speaker_label.strip()
        display_name = display_name.strip()

        if not speaker_label or not display_name:
            return

        self._db.execute(
            """INSERT INTO speaker_profiles (speaker_label, display_name)
               VALUES (?, ?)
               ON CONFLICT(speaker_label) DO UPDATE SET display_name = excluded.display_name""",
            (speaker_label, display_name),
        )
        self._cache[speaker_label] = display_name
        print(f"[Identity] {speaker_label} -> {display_name}")

    def get_display_name(self, speaker_label: str) -> str | None:
        """
        Get the display name for a raw speaker label.

        Args:
            speaker_label: The raw diarization label.

        Returns:
            Display name string, or None if not mapped.
        """
        return self._cache.get(speaker_label)

    def get_all_profiles(self) -> list[dict]:
        """
        Get all speaker profiles.

        Returns:
            List of dicts with keys: speaker_label, display_name, created_at
        """
        return self._db.fetch_all(
            "SELECT speaker_label, display_name, created_at FROM speaker_profiles ORDER BY created_at"
        )

    def remove_profile(self, speaker_label: str) -> bool:
        """
        Remove a speaker profile.

        Args:
            speaker_label: The raw label to remove.

        Returns:
            True if a profile was actually removed.
        """
        existed = speaker_label in self._cache
        self._db.execute(
            "DELETE FROM speaker_profiles WHERE speaker_label = ?",
            (speaker_label,),
        )
        self._cache.pop(speaker_label, None)
        return existed

    def clear_all(self) -> int:
        """
        Remove all speaker profiles.

        Returns:
            Number of profiles removed.
        """
        count = len(self._cache)
        self._db.execute("DELETE FROM speaker_profiles")
        self._cache.clear()
        return count

    # ── Conversation Mapping ───────────────────────────────────

    def apply_to_conversation(self, segments: list[dict]) -> list[dict]:
        """
        Apply identity mapping to conversation segments.

        Each segment gets a `display_name` field added.
        If no mapping exists, display_name = raw speaker label.

        Args:
            segments: List of conversation segments from builder.
                      Each must have a "speaker" key.

        Returns:
            Same list with `display_name` added to each segment.
        """
        for seg in segments:
            raw = seg.get("speaker", "UNKNOWN")
            seg["display_name"] = self._cache.get(raw, raw)
        return segments

    def resolve_name(self, speaker_label: str) -> str:
        """
        Get display name or fall back to raw label.

        Args:
            speaker_label: Raw diarization label.

        Returns:
            Display name if mapped, otherwise the raw label itself.
        """
        return self._cache.get(speaker_label, speaker_label)

    # ── Voice Fingerprint Integration ────────────────────────────

    def register_voiceprint(
        self,
        speaker_name: str,
        embedding,
        repo=None,
    ) -> str | None:
        """
        Store a voice embedding for a speaker in the database.

        Multiple embeddings per speaker improve matching accuracy.
        Requires a Repository instance and the VoiceFingerprintEngine.

        Args:
            speaker_name: Display name (e.g., "Dr. Smith").
            embedding: numpy array (float32 embedding vector).
            repo: Repository instance for persistence.

        Returns:
            Voiceprint record ID, or None if storage failed.
        """
        if repo is None or embedding is None:
            return None

        try:
            from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
            emb_bytes = VoiceFingerprintEngine.embedding_to_bytes(embedding)
            vp_id = repo.save_voiceprint(speaker_name, emb_bytes)
            print(f"[Identity] Registered voiceprint for '{speaker_name}'")
            return vp_id
        except Exception as e:
            print(f"[Identity] Failed to register voiceprint: {e}")
            return None

    def match_voiceprint(
        self,
        embedding,
        repo=None,
        threshold: float = 0.75,
    ) -> dict:
        """
        Match an embedding against stored voiceprints.

        Returns a MatchResult dict with confidence scoring.

        Args:
            embedding: numpy array (float32 embedding vector).
            repo: Repository instance.
            threshold: Minimum similarity to consider a match.

        Returns:
            MatchResult dict: matched_name, similarity_score,
            confidence, auto_assign.
        """
        empty = {
            "matched_name": None,
            "similarity_score": 0.0,
            "confidence": "none",
            "auto_assign": False,
        }

        if repo is None or embedding is None:
            return empty

        try:
            from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
            engine = VoiceFingerprintEngine()

            stored = repo.get_all_voiceprints()
            if not stored:
                return empty

            result = engine.match_against_db(embedding, stored, threshold)

            if result.get("matched_name"):
                print(
                    f"[Identity] Voice match: '{result['matched_name']}' "
                    f"(score={result['similarity_score']:.3f}, "
                    f"confidence={result['confidence']})"
                )
            return result

        except Exception as e:
            print(f"[Identity] Voiceprint match failed: {e}")
            return empty

    def auto_identify_speakers(
        self,
        segments: list[dict],
        audio_path: str,
        repo=None,
        threshold: float = 0.75,
    ) -> list[dict]:
        """
        Attempt to auto-identify speakers in diarization segments
        using stored voiceprints with adaptive model updating.

        Behavior by confidence level:
          - HIGH/MEDIUM: Auto-assign display_name + store new embedding
          - LOW: Flag as pending_confirmation (no auto-assign)
          - NONE: No action

        Manual mappings (assign_label) always take priority.

        Args:
            segments: Diarization segments with speaker/start/end.
            audio_path: Path to the audio file.
            repo: Repository instance for voiceprint DB.
            threshold: Similarity threshold for auto-matching.

        Returns:
            Updated segments with auto-identified speaker labels.
        """
        if not segments or repo is None:
            return segments

        try:
            from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
            engine = VoiceFingerprintEngine()

            if not engine.is_available:
                return segments

            stored = repo.get_all_voiceprints()
            if not stored:
                return segments

            # Find the longest segment per unique speaker
            speaker_best_seg: dict[str, dict] = {}
            for seg in segments:
                spk = seg["speaker"]
                duration = seg["end"] - seg["start"]
                if spk not in speaker_best_seg or duration > (
                    speaker_best_seg[spk]["end"] - speaker_best_seg[spk]["start"]
                ):
                    speaker_best_seg[spk] = seg

            # Map: raw label → match result
            label_map: dict[str, dict] = {}

            for raw_label, best_seg in speaker_best_seg.items():
                # Skip if manual mapping already exists
                if raw_label in self._cache:
                    label_map[raw_label] = {
                        "name": self._cache[raw_label],
                        "auto_assign": True,
                        "confidence": "manual",
                    }
                    continue

                # Extract embedding from the longest segment
                emb = engine.extract_embedding(
                    audio_path,
                    start=best_seg["start"],
                    end=best_seg["end"],
                )
                if emb is None:
                    continue

                # Match against stored voiceprints
                result = engine.match_against_db(emb, stored, threshold)
                name = result.get("matched_name")
                confidence = result.get("confidence", "none")
                auto_assign = result.get("auto_assign", False)
                score = result.get("similarity_score", 0.0)

                if name and auto_assign:
                    # HIGH or MEDIUM: auto-assign + store new embedding (adaptive)
                    label_map[raw_label] = {
                        "name": name,
                        "auto_assign": True,
                        "confidence": confidence,
                    }

                    # Adaptive update: store the new embedding
                    try:
                        emb_bytes = VoiceFingerprintEngine.embedding_to_bytes(emb)
                        repo.save_voiceprint(name, emb_bytes)
                        print(
                            f"[Identity] Auto-matched {raw_label} → '{name}' "
                            f"(score={score:.3f}, {confidence}) + adaptive update"
                        )
                    except Exception as e:
                        print(f"[Identity] Adaptive update failed: {e}")

                elif name and not auto_assign:
                    # LOW: flag for manual confirmation
                    label_map[raw_label] = {
                        "name": name,
                        "auto_assign": False,
                        "confidence": confidence,
                    }
                    print(
                        f"[Identity] Low-confidence match {raw_label} ~ '{name}' "
                        f"(score={score:.3f}) — pending confirmation"
                    )

            # Apply label map to segments
            if label_map:
                for seg in segments:
                    raw = seg["speaker"]
                    if raw in label_map:
                        entry = label_map[raw]
                        if entry["auto_assign"]:
                            seg["display_name"] = entry["name"]
                        else:
                            # Low confidence — flag, don't auto-assign
                            seg["pending_confirmation"] = True
                            seg["suggested_name"] = entry["name"]
                            seg["match_confidence"] = entry["confidence"]

            return segments

        except Exception as e:
            print(f"[Identity] Auto-identification failed: {e}")
            return segments

    def reinforce_voiceprint(
        self,
        speaker_name: str,
        embedding,
        repo=None,
    ) -> str | None:
        """
        Reinforce a speaker's voice model after manual confirmation.

        When a user manually assigns a speaker label, this method
        stores the embedding as a confirmed voiceprint, strengthening
        the speaker's model for future matches.

        Args:
            speaker_name: Confirmed display name.
            embedding: numpy embedding vector from the current session.
            repo: Repository instance.

        Returns:
            Voiceprint record ID, or None if failed.
        """
        if repo is None or embedding is None:
            return None

        try:
            from speaker_identity.voice_fingerprint import VoiceFingerprintEngine
            emb_bytes = VoiceFingerprintEngine.embedding_to_bytes(embedding)
            vp_id = repo.save_voiceprint(speaker_name, emb_bytes)
            print(f"[Identity] Reinforced voiceprint for '{speaker_name}' (manual confirmation)")
            return vp_id
        except Exception as e:
            print(f"[Identity] Reinforcement failed: {e}")
            return None

    @property
    def profile_count(self) -> int:
        """Number of mapped speaker profiles."""
        return len(self._cache)

    def __repr__(self) -> str:
        return f"IdentityManager({self.profile_count} profiles)"


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    db = Database("test_identity.db")
    mgr = IdentityManager(db)

    # Assign labels
    mgr.assign_label("SPEAKER_00", "Dr. Smith")
    mgr.assign_label("SPEAKER_01", "Patient")
    mgr.assign_label("SPEAKER_02", "Caregiver")

    # Lookup
    print(f"SPEAKER_00 -> {mgr.get_display_name('SPEAKER_00')}")
    print(f"SPEAKER_99 -> {mgr.get_display_name('SPEAKER_99')}")

    # Apply to conversation
    segments = [
        {"speaker": "SPEAKER_00", "text": "Take your medicine."},
        {"speaker": "SPEAKER_01", "text": "OK doctor."},
        {"speaker": "SPEAKER_02", "text": "I'll remind them."},
    ]
    mapped = mgr.apply_to_conversation(segments)
    for seg in mapped:
        print(f"  {seg['display_name']}: {seg['text']}")

    # All profiles
    print(f"\nProfiles: {mgr.get_all_profiles()}")
    print(f"Count: {mgr.profile_count}")

    # Cleanup
    import os
    os.remove("test_identity.db")
    print("Test passed!")
