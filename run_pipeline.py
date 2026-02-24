"""
run_pipeline.py -- Day 2 MVP Entry Point
=========================================
Upgraded pipeline with:
  1. Transcriber       -- audio -> text  (Whisper, offline)
  2. Summarizer        -- text -> highlighted summary  (pure Python)
  3. Event Extractor   -- text -> structured JSON events  (regex)
  4. Memory Manager    -- event storage, search, persistence

Usage:
  python run_pipeline.py                   # uses sample text file
  python run_pipeline.py path/to/audio.wav # transcribes real audio
"""

import json
import os
import sys

# -- Import our core modules -----------------------------------------------
from core.summarizer import summarize, summarize_with_highlights
from core.event_extractor import extract_structured_events
from core.memory_manager import MemoryManager


# =========================================================================
# Helper: load text either from audio or from the sample file
# =========================================================================

def load_text(source: str | None = None) -> str:
    """
    If `source` is an audio file -> transcribe it with Whisper.
    If `source` is a .txt file  -> read it directly.
    If `source` is None          -> use the built-in sample text.
    """
    if source is None:
        sample_path = os.path.join(
            os.path.dirname(__file__), "sample", "sample_conversation.txt"
        )
        print(f"[Pipeline] No input provided -- using sample: {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    if source.lower().endswith(".txt"):
        print(f"[Pipeline] Reading text file: {source}")
        with open(source, "r", encoding="utf-8") as f:
            return f.read().strip()

    print(f"[Pipeline] Audio file detected -- starting transcription...")
    from core.transcriber import transcribe_audio
    result = transcribe_audio(source)
    return result["text"]


# =========================================================================
# Main pipeline
# =========================================================================

def run_pipeline(source: str | None = None):
    """Run the full pipeline: Transcribe -> Summarize -> Extract -> Store."""

    # -- Banner ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CONVERSATIONAL MEMORY ASSISTANT -- Day 2 MVP")
    print("=" * 60)

    # -- Step 1: Transcription ---------------------------------------------
    text = load_text(source)

    print("\n[TRANSCRIPTION]")
    print("-" * 40)
    print(text)

    # -- Step 2: Highlighted summary ---------------------------------------
    summary_text = summarize(text, num_sentences=3)
    highlights = summarize_with_highlights(text, num_sentences=3)

    print("\n[HIGHLIGHTED SUMMARY]")
    print("-" * 40)
    for item in highlights:
        marker = "[IMPORTANT]" if item["important"] else "           "
        tags = f"  ({', '.join(item['tags'])})" if item["tags"] else ""
        print(f"  {marker} {item['sentence']}{tags}")

    print(f"\n  Quick Summary: {summary_text}")

    # -- Step 3: Structured event extraction --------------------------------
    events = extract_structured_events(text)

    print("\n[STRUCTURED EVENTS] (JSON)")
    print("-" * 40)

    if not events:
        print("  No events detected.")
    else:
        print(json.dumps(events, indent=2, ensure_ascii=False))

    # -- Step 4: Store in memory manager ------------------------------------
    memory = MemoryManager()
    memory.add_events(events)

    # Save to file
    memory_path = os.path.join(os.path.dirname(__file__), "memory.json")
    memory.save_to_file(memory_path)

    print(f"\n[MEMORY] {memory.count()} events stored")
    print("-" * 40)

    # Show today's events
    today = memory.get_today_events()
    if today:
        print(f"\n  Today/Tomorrow events ({len(today)}):")
        for e in today:
            print(f"    -> [{e['type'].upper()}] {e['description']}")
    else:
        print("  No events for today/tomorrow.")

    # -- Done --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60 + "\n")

    return {
        "transcription": text,
        "summary": summary_text,
        "highlights": highlights,
        "events": events,
        "memory": memory,
    }


# =========================================================================
# CLI entry point
# =========================================================================

if __name__ == "__main__":
    source_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(source_file)
