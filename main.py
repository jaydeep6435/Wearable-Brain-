"""
Memory Assistant — Main Entry Point
======================================
CLI interface for the offline MemoryAssistantEngine.

Usage:
    python main.py                           # Interactive mode
    python main.py --record                  # Record conversation (stop with Ctrl+C)
    python main.py --process-text "text..."  # Process text
    python main.py --process-audio file.wav  # Process audio
    python main.py --query "question?"       # Ask a question
    python main.py --events                  # List all events
    python main.py --stats                   # Show statistics
    python main.py --start-listening         # Start background VAD listener
    python main.py --recordings              # List saved recordings
    python main.py --migrate                 # Import from memory.json
    python main.py --serve                   # Start local API bridge
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Conversational Memory Assistant — Offline Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--process-text", "-t",
        type=str, metavar="TEXT",
        help="Process conversation text through the pipeline",
    )
    parser.add_argument(
        "--process-audio", "-a",
        type=str, metavar="FILE",
        help="Process an audio file (wav, mp3, m4a)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str, metavar="QUESTION",
        help="Ask a question about stored memories",
    )
    parser.add_argument(
        "--events", "-e",
        action="store_true",
        help="List all stored events",
    )
    parser.add_argument(
        "--upcoming", "-u",
        type=int, nargs="?", const=60, metavar="MINUTES",
        help="Show upcoming events (default: next 60 min)",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show engine statistics",
    )
    parser.add_argument(
        "--migrate", "-m",
        action="store_true",
        help="Migrate data from memory.json to SQLite",
    )

    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-enhanced processing (requires Ollama)",
    )
    parser.add_argument(
        "--db",
        type=str, default=None,
        help="Path to SQLite database (default: memory.db)",
    )

    # Session recording
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start recording a conversation (Ctrl+C to stop and process)",
    )
    parser.add_argument(
        "--recordings",
        action="store_true",
        help="List all saved conversation recordings",
    )

    # Background VAD listening
    parser.add_argument(
        "--start-listening",
        action="store_true",
        help="Start VAD-based background listening (hands-free)",
    )
    parser.add_argument(
        "--worker-status",
        action="store_true",
        help="Show background worker status",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float, default=500.0,
        help="VAD silence threshold (higher = less sensitive, default: 500)",
    )

    args = parser.parse_args()

    # Import engine (lazy — only when needed)
    from engine.assistant_engine import MemoryAssistantEngine

    engine = MemoryAssistantEngine(db_path=args.db)
    use_llm = args.llm

    # ── Dispatch commands ──────────────────────────────────────



    if args.record:
        import time as _time
        result = engine.start_recording()
        print(f"\n  Recording started: {result['status']}")
        print("  Speak naturally into your microphone.")
        print("  Press Ctrl+C when the conversation is done.\n")
        try:
            while engine.get_worker_status().get("running", False):
                duration = engine.get_worker_status().get("session_duration", 0)
                print(f"\r  Recording... {duration:.0f}s", end="", flush=True)
                _time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n  Stopping and processing...")
            result = engine.stop_recording()
            print(f"\n  Status: {result.get('status')}")
            if result.get('file'):
                print(f"  Saved:  {result['file']}")
                print(f"  Duration: {result.get('duration', 0):.1f}s")
            if result.get('processing'):
                p = result['processing']
                print(f"  Speakers: {p.get('num_speakers', 1)}")
                print(f"  Summary: {p.get('summary', 'N/A')}")
                print(f"  Events: {p.get('events_saved', 0)}")
        return

    if args.recordings:
        recordings = engine.list_recordings()
        if recordings:
            print(f"\n  Saved recordings ({len(recordings)}):")
            for i, r in enumerate(recordings, 1):
                print(f"  {i:3d}. {r['file']}  ({r['size_mb']} MB)  {r['modified']}")
        else:
            print("\n  No recordings found.")
        return

    if args.start_listening:
        import time as _time
        result = engine.start_background_listening(
            silence_threshold=args.silence_threshold
        )
        print(f"\n  VAD listening: {result['status']}")
        print(f"  Threshold: {args.silence_threshold}")
        print("  Press Ctrl+C to stop...\n")
        try:
            while engine.get_worker_status().get("running", False):
                _time.sleep(1)
        except KeyboardInterrupt:
            print("\n")
            engine.stop_background_listening()
            final = engine.get_worker_status()
            print(f"  Recordings captured: {final.get('recordings_captured', 0)}")
            print(f"  Recordings processed: {final.get('recordings_processed', 0)}")
        return

    if args.worker_status:
        status = engine.get_worker_status()
        print(json.dumps(status, indent=2))
        return

    if args.migrate:
        json_path = os.path.join(PROJECT_ROOT, "memory.json")
        result = engine.repo.migrate_from_json(json_path)
        print(json.dumps(result, indent=2))
        return

    if args.stats:
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2))
        return

    if args.events:
        events = engine.get_events()
        print(f"\n  Total events: {len(events)}")
        for i, e in enumerate(events, 1):
            desc = e.get("description", "?")
            etype = e.get("type", "?")
            person = e.get("person") or ""
            print(f"  {i:3d}. [{etype}] {desc}" + (f" (with {person})" if person else ""))
        return

    if args.upcoming is not None:
        result = engine.get_upcoming_events(minutes=args.upcoming)
        upcoming = result["upcoming"]
        alerts = result["alerts"]
        if upcoming:
            print(f"\n  Upcoming events (next {args.upcoming} min):")
            for e in upcoming:
                print(f"    - {e.get('description', '?')} (in {e.get('minutes_until', '?')} min)")
        else:
            print(f"\n  No events in the next {args.upcoming} minutes.")
        if alerts:
            print(f"\n  ALERTS:")
            for a in alerts:
                print(f"    ! {a}")
        return

    if args.process_text:
        result = engine.process_text(args.process_text, use_llm=use_llm)
        print(f"\n  Summary: {result.get('summary', 'N/A')}")
        print(f"  Events found: {len(result.get('events', []))}")
        print(f"  Events saved: {result.get('events_saved', 0)}")
        if result.get("events"):
            print("\n  Extracted events:")
            for e in result["events"]:
                print(f"    - [{e.get('type')}] {e.get('description', '?')}")
        return

    if args.process_audio:
        result = engine.process_audio(args.process_audio, use_llm=use_llm)
        if result.get("error"):
            print(f"\n  Error: {result['error']}")
            return
        print(f"\n  Transcription: {result.get('transcription', 'N/A')[:100]}...")
        print(f"  Speakers: {result.get('num_speakers', 1)}")
        print(f"  Summary: {result.get('summary', 'N/A')}")
        print(f"  Events saved: {result.get('events_saved', 0)}")
        if result.get("speakers"):
            print("\n  Speaker segments:")
            for seg in result["speakers"]:
                print(f"    {seg['speaker']}: {seg['text'][:80]}")
        return

    if args.query:
        result = engine.query(args.query, use_llm=use_llm)
        print(f"\n  Q: {result['question']}")
        print(f"  A: {result['answer']}")
        return

    # ── Interactive mode (no arguments) ────────────────────────
    _interactive_mode(engine, use_llm)


def _interactive_mode(engine, use_llm: bool = False):
    """Run the interactive CLI loop."""
    print("\n" + "=" * 60)
    print("  Memory Assistant — Interactive Mode")
    print("=" * 60)
    print("  Commands:")
    print("    /record          Start recording conversation")
    print("    /stop            Stop recording & process")
    print("    /recordings      List saved recordings")
    print("    /text <text>     Process conversation text")
    print("    /audio <file>    Process audio file")
    print("    /events          List all events")
    print("    /upcoming [min]  Show upcoming events")
    print("    /listen          Start VAD listening (hands-free)")
    print("    /stop-listen     Stop VAD listening")
    print("    /status          Show worker status")
    print("    /stats           Show statistics")
    print("    /llm             Toggle LLM mode")
    print("    /quit            Exit")
    print("  Or just type a question to query memories.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("  Goodbye!")
            break

        if user_input.lower() == "/llm":
            use_llm = not use_llm
            print(f"  LLM mode: {'ON' if use_llm else 'OFF'}")
            continue

        if user_input.lower() == "/stats":
            stats = engine.get_stats()
            print(f"  {json.dumps(stats, indent=4)}")
            continue

        if user_input.lower() == "/events":
            events = engine.get_events()
            print(f"  Total events: {len(events)}")
            for i, e in enumerate(events, 1):
                print(f"  {i}. [{e.get('type')}] {e.get('description', '?')}")
            continue

        if user_input.lower().startswith("/upcoming"):
            parts = user_input.split()
            minutes = int(parts[1]) if len(parts) > 1 else 60
            result = engine.get_upcoming_events(minutes=minutes)
            upcoming = result["upcoming"]
            if upcoming:
                for e in upcoming:
                    print(f"    - {e.get('description', '?')} (in {e.get('minutes_until', '?')} min)")
            else:
                print(f"  No events in the next {minutes} minutes.")
            continue

        if user_input.lower().startswith("/text "):
            text = user_input[6:].strip()
            result = engine.process_text(text, use_llm=use_llm)
            print(f"  Summary: {result.get('summary', 'N/A')}")
            print(f"  Events saved: {result.get('events_saved', 0)}")
            continue

        if user_input.lower().startswith("/audio "):
            path = user_input[7:].strip()
            result = engine.process_audio(path, use_llm=use_llm)
            print(f"  Transcription: {result.get('transcription', 'N/A')[:100]}...")
            print(f"  Events saved: {result.get('events_saved', 0)}")
            continue

        if user_input.lower() == "/record":
            result = engine.start_recording()
            print(f"  Recording: {result['status']}")
            if result['status'] == 'recording':
                print("  Speak naturally. Type /stop when done.")
            continue

        if user_input.lower() == "/stop":
            result = engine.stop_recording()
            print(f"  Status: {result.get('status')}")
            if result.get('file'):
                print(f"  Saved: {result['file']}")
                print(f"  Duration: {result.get('duration', 0):.1f}s")
            if result.get('processing'):
                p = result['processing']
                print(f"  Summary: {p.get('summary', 'N/A')}")
                print(f"  Events: {p.get('events_saved', 0)}")
            continue

        if user_input.lower() == "/recordings":
            recordings = engine.list_recordings()
            if recordings:
                print(f"  Saved recordings ({len(recordings)}):")
                for i, r in enumerate(recordings, 1):
                    print(f"  {i}. {r['file']}  ({r['size_mb']} MB)")
            else:
                print("  No recordings yet.")
            continue

        if user_input.lower() == "/listen":
            result = engine.start_background_listening()
            print(f"  VAD: {result['status']}")
            continue

        if user_input.lower() == "/stop-listen":
            result = engine.stop_background_listening()
            print(f"  VAD: {result['status']}")
            continue

        if user_input.lower() == "/status":
            status = engine.get_worker_status()
            print(f"  {json.dumps(status, indent=4)}")
            continue

        # Default: treat as a query
        result = engine.query(user_input, use_llm=use_llm)
        print(f"  {result['answer']}")


if __name__ == "__main__":
    main()
