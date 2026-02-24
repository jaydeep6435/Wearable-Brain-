"""
run_pipeline.py -- Day 4 MVP Entry Point
=========================================
Full pipeline with:
  1. Transcriber       -- audio -> text  (Whisper, offline)
  2. Summarizer        -- text -> highlighted summary  (pure Python)
  3. Event Extractor   -- text -> structured JSON with real dates  (regex + dateparser)
  4. Memory Manager    -- event storage, search, persistence
  5. Query Engine      -- natural language memory queries
  6. Reminder Manager  -- scheduled alerts for upcoming events

Usage:
  python run_pipeline.py                   # process sample text
  python run_pipeline.py path/to/audio.wav # process audio file
  python run_pipeline.py -i                # interactive query mode
  python run_pipeline.py -r                # enable reminders
  python run_pipeline.py -i -r            # interactive + reminders
"""

import json
import os
import sys

# -- Import our core modules -----------------------------------------------
from core.summarizer import summarize, summarize_with_highlights
from core.event_extractor import extract_structured_events
from core.memory_manager import MemoryManager
from core.query_engine import QueryEngine
from core.reminder_manager import ReminderManager


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

def run_pipeline(source: str | None = None) -> dict:
    """Run the full pipeline: Transcribe -> Summarize -> Extract -> Store."""

    # -- Banner ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CONVERSATIONAL MEMORY ASSISTANT -- Day 4 MVP")
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

    # -- Step 3: Structured event extraction with real dates ----------------
    events = extract_structured_events(text)

    print("\n[STRUCTURED EVENTS] (JSON with parsed dates)")
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
            desc = e.get("description", "Unknown")
            parsed = e.get("parsed_date", "")
            print(f"    -> [{e['type'].upper()}] {desc}  ({parsed})")
    else:
        print("  No events for today/tomorrow.")

    # -- Step 5: Reminder preview ------------------------------------------
    reminder = ReminderManager(memory)

    schedule = reminder.get_todays_schedule()
    if schedule:
        print(f"\n[TODAY'S SCHEDULE]")
        print("-" * 40)
        print(reminder.format_schedule(schedule))

    upcoming = reminder.get_upcoming_events(minutes=24 * 60)
    if upcoming:
        print(f"\n[UPCOMING (next 24h)]")
        print("-" * 40)
        for e in upcoming:
            print(f"  ⏰ {e['description']} (in {e['minutes_until']} min)")

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
        "reminder": reminder,
    }


# =========================================================================
# Interactive query mode
# =========================================================================

def interactive_query(memory: MemoryManager):
    """
    Start an interactive query session.
    User can type natural language questions about stored memories.
    """
    engine = QueryEngine(memory)

    print("\n" + "=" * 60)
    print("  INTERACTIVE QUERY MODE")
    print("  Ask me anything about your conversations!")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    print("\n  Example questions:")
    print("    - What meetings do I have tomorrow?")
    print("    - Do I have any tasks?")
    print("    - Did I take medicine?")
    print("    - Give me a summary")
    print("    - Tell me about the pharmacy")
    print()

    while True:
        try:
            question = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "bye", "q"):
            print("  Goodbye!")
            break

        answer = engine.query(question)
        print(f"  Bot > {answer}\n")


# =========================================================================
# Reminder mode
# =========================================================================

def start_reminders(reminder: ReminderManager):
    """Start the reminder loop and wait."""
    print("\n" + "=" * 60)
    print("  REMINDER MODE ACTIVE")
    print("  Checking for upcoming events every 60 seconds...")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    reminder.start_reminder_loop(interval=60)

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        reminder.stop()
        print("\n  Reminders stopped.")


# =========================================================================
# CLI entry point
# =========================================================================

if __name__ == "__main__":
    # Parse arguments
    flags = {"-i", "--interactive", "-r", "--reminders"}
    args = [a for a in sys.argv[1:] if a not in flags]
    interactive = "-i" in sys.argv or "--interactive" in sys.argv
    reminders = "-r" in sys.argv or "--reminders" in sys.argv

    source_file = args[0] if args else None

    # Run the pipeline
    result = run_pipeline(source_file)

    # Enter interactive mode if requested
    if interactive:
        if reminders:
            result["reminder"].start_reminder_loop(interval=60)
            print("  [Reminders active in background]")
        interactive_query(result["memory"])
        if reminders:
            result["reminder"].stop()
    elif reminders:
        start_reminders(result["reminder"])
