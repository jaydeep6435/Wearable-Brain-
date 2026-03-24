"""
api.py — Flask REST API for Conversational Memory Assistant
=============================================================
Exposes the AI pipeline as HTTP endpoints for mobile/web integration.

Endpoints:
  POST /process_audio   — Upload audio file → transcription + summary + events
  POST /process_text    — Send text → summary + events
  POST /query           — Ask a question → get answer
  GET  /events          — Get all stored events
  GET  /reminders       — Get upcoming reminders

Usage:
  python api.py

Server runs at http://localhost:5000
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

# -- Import our core modules -----------------------------------------------
from core.summarizer import summarize, summarize_with_highlights
from core.event_extractor import extract_structured_events
from core.memory_manager import MemoryManager
from core.query_engine import QueryEngine
from core.reminder_manager import ReminderManager


# =========================================================================
# App setup
# =========================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter / mobile app

# Global memory + reminder manager (shared across requests)
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")
memory = MemoryManager()

# Load existing memory if available
if os.path.isfile(MEMORY_FILE):
    memory.load_from_file(MEMORY_FILE)
    print(f"[API] Loaded {memory.count()} events from {MEMORY_FILE}")

query_engine = QueryEngine(memory)
reminder_mgr = ReminderManager(memory)


# =========================================================================
# POST /process_audio — Upload audio → transcription + summary + events
# =========================================================================

@app.route("/process_audio", methods=["POST"])
def process_audio():
    """
    Upload an audio file for processing.

    Expects: multipart/form-data with a file field named 'audio'
    Returns: JSON with transcription, summary, highlights, events
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use field name 'audio'."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Save to temp file
    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    use_llm = request.args.get("use_llm", "false").lower() == "true"

    try:
        # Step 1: Transcribe
        from core.transcriber import transcribe_audio
        result = transcribe_audio(tmp_path)
        text = result["text"]

        # Step 2: Summarize
        summary = summarize(text, num_sentences=3, use_llm=use_llm)
        highlights = summarize_with_highlights(text, num_sentences=3)

        # Step 3: Extract events
        events = extract_structured_events(text, use_llm=use_llm)

        # Step 4: Store in memory
        memory.add_events(events)
        memory.save_to_file(MEMORY_FILE)

        return jsonify({
            "status": "success",
            "transcription": text,
            "summary": summary,
            "highlights": highlights,
            "events": events,
            "llm_used": use_llm,
            "total_events_in_memory": memory.count(),
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)


# =========================================================================
# POST /process_text — Send text → summary + events
# =========================================================================

@app.route("/process_text", methods=["POST"])
def process_text():
    """
    Process a text input.

    Expects: JSON body with {"text": "..."}
    Returns: JSON with summary, highlights, events
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON body."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text is empty."}), 400

    use_llm = request.args.get("use_llm", "false").lower() == "true"

    # Step 1: Summarize
    summary = summarize(text, num_sentences=3, use_llm=use_llm)
    highlights = summarize_with_highlights(text, num_sentences=3)

    # Step 2: Extract events
    events = extract_structured_events(text, use_llm=use_llm)

    # Step 3: Store in memory
    memory.add_events(events)
    memory.save_to_file(MEMORY_FILE)

    return jsonify({
        "status": "success",
        "summary": summary,
        "highlights": highlights,
        "events": events,
        "llm_used": use_llm,
        "total_events_in_memory": memory.count(),
    })


# =========================================================================
# POST /query — Ask a question → get answer
# =========================================================================

@app.route("/query", methods=["POST"])
def query():
    """
    Ask a natural language question about stored memories.

    Expects: JSON body with {"question": "..."}
    Returns: JSON with answer string
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in JSON body."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question is empty."}), 400

    use_llm = request.args.get("use_llm", "false").lower() == "true"
    answer = query_engine.query(question, use_llm=use_llm)

    return jsonify({
        "status": "success",
        "question": question,
        "answer": answer,
        "llm_used": use_llm,
    })


# =========================================================================
# GET /events — Get all stored events
# =========================================================================

@app.route("/events", methods=["GET"])
def get_events():
    """
    Get all events stored in memory.

    Optional query params:
      ?type=meeting     — filter by event type
      ?search=pharmacy  — search events by keyword

    Returns: JSON with list of events
    """
    event_type = request.args.get("type")
    search = request.args.get("search")

    events = memory.get_all_events()

    # Filter by type if specified
    if event_type:
        events = [e for e in events if e.get("type") == event_type.lower()]

    # Search by keyword if specified
    if search:
        events = memory.search_events(search)

    return jsonify({
        "status": "success",
        "count": len(events),
        "events": events,
    })


# =========================================================================
# GET /reminders — Get upcoming reminders
# =========================================================================

@app.route("/reminders", methods=["GET"])
def get_reminders():
    """
    Get upcoming event reminders.

    Optional query params:
      ?minutes=60  — look-ahead window (default: 60 minutes)

    Returns: JSON with upcoming events and alerts
    """
    minutes = request.args.get("minutes", 60, type=int)

    upcoming = reminder_mgr.get_upcoming_events(minutes=minutes)
    today_schedule = reminder_mgr.get_todays_schedule()

    return jsonify({
        "status": "success",
        "upcoming_count": len(upcoming),
        "upcoming": upcoming,
        "todays_schedule": today_schedule,
    })


# =========================================================================
# GET / — Health check / API info
# =========================================================================

@app.route("/", methods=["GET"])
def index():
    """API info and health check."""
    return jsonify({
        "app": "Conversational Memory Assistant API",
        "version": "Day 7 — Hybrid LLM",
        "status": "running",
        "events_in_memory": memory.count(),
        "endpoints": {
            "POST /process_audio": "Upload audio (?use_llm=true)",
            "POST /process_text": "Send text (?use_llm=true)",
            "POST /query": "Ask a question (?use_llm=true)",
            "GET /events": "All events (?type=meeting&search=keyword)",
            "GET /reminders": "Upcoming reminders (?minutes=60)",
            "GET /llm/status": "Check Ollama LLM status",
        },
    })


# =========================================================================
# GET /llm/status — Check LLM availability
# =========================================================================

@app.route("/llm/status", methods=["GET"])
def llm_status():
    """Check if the local LLM (Ollama) is running and available."""
    try:
        from core.llm_engine import is_available, get_models, OLLAMA_URL, DEFAULT_MODEL
        available = is_available()
        models = get_models() if available else []
        return jsonify({
            "status": "online" if available else "offline",
            "ollama_url": OLLAMA_URL,
            "default_model": DEFAULT_MODEL,
            "available_models": models,
            "instructions": None if available else "Install Ollama from https://ollama.com then run: ollama pull qwen2.5:3b-instruct",
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
        })


# =========================================================================
# Run server
# =========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CONVERSATIONAL MEMORY ASSISTANT — API Server")
    print("=" * 60)
    print(f"  Events in memory: {memory.count()}")
    print(f"  Memory file: {MEMORY_FILE}")
    print(f"  Server: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
