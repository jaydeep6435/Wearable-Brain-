"""
run_pipeline.py — Day 1 MVP Entry Point
=========================================
This script ties together the three core modules:
  1. Transcriber  — converts audio → text  (Whisper)
  2. Summarizer   — extracts key sentences  (pure Python)
  3. Event Extractor — finds dates, times, meetings, tasks  (regex)

Usage:
  python run_pipeline.py                   # uses sample text file
  python run_pipeline.py path/to/audio.wav # transcribes real audio
"""

import os
import sys

# ── Import our core modules ──────────────────────────────────────────────
from core.summarizer import summarize
from core.event_extractor import extract_events


# ═══════════════════════════════════════════════════════════════════════════
# Helper: load text either from audio or from the sample file
# ═══════════════════════════════════════════════════════════════════════════

def load_text(source: str | None = None) -> str:
    """
    If `source` is an audio file → transcribe it with Whisper.
    If `source` is a .txt file  → read it directly.
    If `source` is None          → use the built-in sample text.
    """
    # --- No argument: use the sample conversation ---
    if source is None:
        sample_path = os.path.join(
            os.path.dirname(__file__), "sample", "sample_conversation.txt"
        )
        print(f"[Pipeline] No input provided — using sample: {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    # --- Text file ---
    if source.lower().endswith(".txt"):
        print(f"[Pipeline] Reading text file: {source}")
        with open(source, "r", encoding="utf-8") as f:
            return f.read().strip()

    # --- Audio file: use Whisper ---
    print(f"[Pipeline] Audio file detected — starting transcription...")
    from core.transcriber import transcribe_audio
    result = transcribe_audio(source)
    return result["text"]


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(source: str | None = None):
    """Run the full pipeline: Transcribe → Summarize → Extract Events."""

    # ── Step 1: Get the transcription text ───────────────────────────────
    print("\n" + "=" * 60)
    print("  CONVERSATIONAL MEMORY ASSISTANT -- Day 1 MVP")
    print("=" * 60)

    text = load_text(source)

    print("\n[TRANSCRIPTION] FULL TRANSCRIPTION")
    print("-" * 40)
    print(text)

    # -- Step 2: Generate summary ------------------------------------------
    summary = summarize(text, num_sentences=3)

    print("\n[SUMMARY] Top 3 sentences")
    print("-" * 40)
    print(summary)

    # -- Step 3: Extract events --------------------------------------------
    events = extract_events(text)

    print("\n[EVENTS] EXTRACTED EVENTS")
    print("-" * 40)

    if not events:
        print("  No events detected.")
    else:
        # Group events by type for cleaner output
        event_types = {}
        for e in events:
            event_types.setdefault(e["type"], []).append(e)

        type_icons = {
            "date": "[DATE]",
            "time": "[TIME]",
            "meeting": "[MEETING]",
            "task": "[TASK]",
            "medication": "[MEDS]",
        }

        for etype, elist in event_types.items():
            icon = type_icons.get(etype, "*")
            print(f"\n  {icon} {etype.upper()} ({len(elist)} found)")
            for e in elist:
                print(f"     -> {e['value']}")
                print(f"        Context: \"{e['context']}\"")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60 + "\n")

    # Return structured results (useful for programmatic access)
    return {
        "transcription": text,
        "summary": summary,
        "events": events,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    source_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(source_file)
