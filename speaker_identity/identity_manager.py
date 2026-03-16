"""
Identity Manager
================

Lightweight speaker label normalization for diarization output.
Keeps stable friendly names across one runtime session.
"""

from __future__ import annotations

from datetime import datetime


class IdentityManager:
    """Maps raw diarization labels to stable human-friendly speaker names."""

    def __init__(self, _db=None):
        self._label_map: dict[str, str] = {}
        self._created_at: dict[str, str] = {}

    @property
    def profile_count(self) -> int:
        return len(self._label_map)

    def resolve_label(self, raw_label: str) -> str:
        key = str(raw_label or "SPEAKER_00")
        if key in self._label_map:
            return self._label_map[key]

        friendly = f"Speaker {len(self._label_map) + 1}"
        self._label_map[key] = friendly
        self._created_at.setdefault(key, datetime.now().isoformat())
        return friendly

    def assign_label(self, raw_label: str, display_name: str) -> None:
        """Assign a user-friendly name to a raw speaker label."""
        key = str(raw_label or "SPEAKER_00")
        name = str(display_name or "Speaker 1").strip() or "Speaker 1"
        self._label_map[key] = name
        self._created_at.setdefault(key, datetime.now().isoformat())

    def get_all_profiles(self) -> list[dict]:
        """Return current speaker label mappings."""
        rows = []
        for raw, name in self._label_map.items():
            rows.append(
                {
                    "speaker_label": raw,
                    "display_name": name,
                    "created_at": self._created_at.get(raw, ""),
                }
            )
        return rows

    def remove_profile(self, raw_label: str) -> bool:
        """Remove mapping for a raw speaker label."""
        key = str(raw_label or "")
        if key not in self._label_map:
            return False
        self._label_map.pop(key, None)
        self._created_at.pop(key, None)
        return True

    def auto_identify_speakers(
        self,
        diarization_segments: list[dict],
        audio_path: str | None = None,
        repo=None,
    ) -> list[dict]:
        """
        Normalize diarization speaker labels.

        Voiceprint-based re-identification can be added later without changing
        the engine contract.
        """
        _ = audio_path
        _ = repo

        normalized: list[dict] = []
        for seg in diarization_segments or []:
            updated = dict(seg)
            updated["speaker"] = self.resolve_label(seg.get("speaker", "SPEAKER_00"))
            normalized.append(updated)
        return normalized
