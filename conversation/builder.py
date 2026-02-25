"""
Conversation Builder — Merge Diarization + Transcription
==========================================================
Takes speaker diarization segments and Whisper transcription,
and produces a structured conversation with speaker labels.

Input:
    diarization_segments: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5}]
    whisper_segments:     [{"text": "hello", "start": 0.1, "end": 1.2}]

Output:
    [{"speaker": "SPEAKER_00", "text": "hello world", "start": 0.0, "end": 3.5}]
"""


class ConversationBuilder:
    """
    Merges speaker diarization output with Whisper transcription segments
    to produce a speaker-labeled conversation.

    Usage:
        builder = ConversationBuilder()
        conversation = builder.build(diarization_segments, whisper_result)
    """

    def build(
        self,
        diarization_segments: list[dict],
        whisper_result: dict,
        identity_manager=None,
    ) -> list[dict]:
        """
        Merge diarization and transcription into structured conversation.

        Args:
            diarization_segments: Output from SpeakerDiarizer.diarize()
                [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5}, ...]

            whisper_result: Output from transcribe_audio()
                {"text": "full text", "segments": [{"text": "...", "start": ..., "end": ...}]}

        Returns:
            List of speaker-labeled segments:
            [
                {"speaker": "SPEAKER_00", "text": "hello world", "start": 0.0, "end": 3.5},
                {"speaker": "SPEAKER_01", "text": "how are you", "start": 3.5, "end": 6.0},
            ]
        """
        whisper_segments = whisper_result.get("segments", [])

        if not diarization_segments:
            # No diarization — return full text as single speaker
            return [{
                "speaker": "SPEAKER_00",
                "text": whisper_result.get("text", "").strip(),
                "start": 0.0,
                "end": whisper_segments[-1]["end"] if whisper_segments else 0.0,
            }]

        if not whisper_segments:
            # No transcription segments — can't merge
            return [{
                "speaker": seg["speaker"],
                "text": "",
                "start": seg["start"],
                "end": seg["end"],
            } for seg in diarization_segments]

        # Assign each Whisper segment to the best-matching diarization segment
        conversation = []
        for dia_seg in diarization_segments:
            # Collect all Whisper words that overlap with this speaker segment
            texts = []
            for w_seg in whisper_segments:
                overlap = self._compute_overlap(
                    dia_seg["start"], dia_seg["end"],
                    w_seg["start"], w_seg["end"],
                )
                # If more than 50% of the Whisper segment overlaps → assign it
                w_duration = w_seg["end"] - w_seg["start"]
                if w_duration > 0 and overlap / w_duration > 0.5:
                    texts.append(w_seg["text"].strip())

            combined_text = " ".join(texts).strip()
            if combined_text:
                conversation.append({
                    "speaker": dia_seg["speaker"],
                    "text": combined_text,
                    "start": dia_seg["start"],
                    "end": dia_seg["end"],
                })

        # Merge consecutive segments from the same speaker
        conversation = self._merge_consecutive(conversation)

        # Apply identity mapping if available
        if identity_manager is not None:
            conversation = identity_manager.apply_to_conversation(conversation)
        else:
            # Add display_name = raw speaker label as default
            for seg in conversation:
                seg["display_name"] = seg["speaker"]

        return conversation

    def build_text(self, conversation: list[dict]) -> str:
        """
        Convert structured conversation into readable text.

        Uses display_name if available, falls back to raw speaker label.

        Args:
            conversation: Output from build()

        Returns:
            Formatted string like:
            "Dr. Smith: Hello world.
             Patient: How are you?"
        """
        lines = []
        for seg in conversation:
            name = seg.get("display_name", seg.get("speaker", "UNKNOWN"))
            text = seg.get("text", "")
            lines.append(f"{name}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _compute_overlap(
        start1: float, end1: float,
        start2: float, end2: float,
    ) -> float:
        """Compute overlap duration between two time ranges."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0.0, overlap_end - overlap_start)

    @staticmethod
    def _merge_consecutive(segments: list[dict]) -> list[dict]:
        """Merge consecutive segments from the same speaker."""
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            last = merged[-1]
            if seg["speaker"] == last["speaker"]:
                last["text"] = f"{last['text']} {seg['text']}".strip()
                last["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        return merged


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    builder = ConversationBuilder()

    # Simulated diarization output (2 speakers)
    dia_segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0},
        {"speaker": "SPEAKER_01", "start": 4.0, "end": 8.0},
        {"speaker": "SPEAKER_00", "start": 8.0, "end": 12.0},
        {"speaker": "SPEAKER_01", "start": 12.0, "end": 16.0},
    ]

    # Simulated Whisper output
    whisper_result = {
        "text": "I have a doctor appointment tomorrow at 10 AM. "
                "Don't forget to take your medicine after breakfast. "
                "We need to call the pharmacy to refill. "
                "Your son David is visiting this weekend.",
        "segments": [
            {"text": "I have a doctor appointment tomorrow at 10 AM.", "start": 0.5, "end": 3.5},
            {"text": "Don't forget to take your medicine after breakfast.", "start": 4.2, "end": 7.5},
            {"text": "We need to call the pharmacy to refill.", "start": 8.5, "end": 11.0},
            {"text": "Your son David is visiting this weekend.", "start": 12.3, "end": 15.5},
        ],
    }

    conversation = builder.build(dia_segments, whisper_result)
    print("--- Structured Conversation ---")
    print(json.dumps(conversation, indent=2))

    print("\n--- Readable Text ---")
    print(builder.build_text(conversation))
