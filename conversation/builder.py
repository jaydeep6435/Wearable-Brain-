"""
Conversation Builder
====================

Merges diarization timing and ASR segments into a single speaker-labeled
conversation structure used by the engine and persistence layer.
"""

from __future__ import annotations

from typing import Any


class ConversationBuilder:
    """Builds normalized speaker-segment conversation entries."""

    @staticmethod
    def _best_speaker(start: float, end: float, diarization: list[dict[str, Any]]) -> str:
        if not diarization:
            return "Speaker 1"

        best = diarization[0].get("speaker", "Speaker 1")
        best_overlap = -1.0
        for seg in diarization:
            d_start = float(seg.get("start", 0.0))
            d_end = float(seg.get("end", 0.0))
            overlap = max(0.0, min(end, d_end) - max(start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = seg.get("speaker", best)
        return str(best)

    def build(
        self,
        diarization_segments: list[dict[str, Any]],
        asr_result: dict[str, Any],
        identity_manager=None,
    ) -> list[dict[str, Any]]:
        """Return conversation segments with keys: speaker/start/end/text."""
        asr_segments = asr_result.get("segments", []) or []

        conversation: list[dict[str, Any]] = []

        for seg in asr_segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            speaker = str(seg.get("speaker") or self._best_speaker(start, end, diarization_segments))

            if identity_manager is not None:
                speaker = identity_manager.resolve_label(speaker)

            conversation.append(
                {
                    "speaker": speaker,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                }
            )

        if not conversation and diarization_segments:
            # Keep timing-only diarization output if ASR did not emit segments.
            for seg in diarization_segments:
                speaker = str(seg.get("speaker", "Speaker 1"))
                if identity_manager is not None:
                    speaker = identity_manager.resolve_label(speaker)
                conversation.append(
                    {
                        "speaker": speaker,
                        "start": round(float(seg.get("start", 0.0)), 2),
                        "end": round(float(seg.get("end", 0.0)), 2),
                        "text": "",
                    }
                )

        return conversation

    @staticmethod
    def build_text(conversation: list[dict[str, Any]]) -> str:
        """Render speaker-attributed transcript text for summarization/storage."""
        lines = []
        for seg in conversation:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            speaker = str(seg.get("speaker", "Speaker 1"))
            lines.append(f"[{speaker}] {text}")
        return "\n".join(lines).strip()
