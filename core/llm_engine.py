"""
LLM Engine Module — Local LLM via Ollama
==========================================
Connects to a local Ollama instance (http://localhost:11434) to use
lightweight local LLMs (qwen2.5, phi3, gemma) for improved NLP tasks.

Features:
  - Health check (is Ollama running?)
  - Raw text generation
  - Structured event extraction via prompt
  - Conversation summarization via prompt
  - Memory-aware Q&A via prompt

All processing is fully offline. Gracefully returns None if Ollama
is unavailable, so the system can fall back to rule-based methods.

Setup:
  1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull qwen2.5:3b-instruct
  3. Start server:   ollama serve  (or runs automatically)
"""

import json
import os
import re
import requests
from datetime import datetime


# =========================================================================
# Configuration
# =========================================================================

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = os.environ.get("WBRAIN_OLLAMA_MODEL", "qwen2.5:3b-instruct")
"""Preferred default model for local Ollama generation.
Can be overridden with WBRAIN_OLLAMA_MODEL.
"""

# Ordered best-effort model preference list.
MODEL_PREFERENCE = [
    "qwen2.5:3b-instruct",
    "qwen2.5:3b",
    "phi3:mini",
    "phi3",
    "gemma2:2b",
    "mistral:7b-instruct",
    "mistral",
]
TIMEOUT = 60                # Max seconds to wait for LLM response


# =========================================================================
# Health check
# =========================================================================

def is_available() -> bool:
    """Check if Ollama is running and has a model available."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return len(models) > 0
        return False
    except (requests.ConnectionError, requests.Timeout):
        return False


def get_models() -> list[str]:
    """Return list of available model names."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
        return []
    except (requests.ConnectionError, requests.Timeout):
        return []


def _normalize_model_name(name: str) -> str:
    """Normalize model names so matching works across :latest suffixes."""
    return name.lower().strip()


def select_model(requested_model: str | None = None) -> str:
    """Pick the best available local model, preferring lightweight/strong options.

    Resolution order:
      1) Explicit function argument
      2) WBRAIN_OLLAMA_MODEL env / DEFAULT_MODEL
      3) MODEL_PREFERENCE shortlist
      4) First available Ollama model
      5) DEFAULT_MODEL (last-resort when tags call fails)
    """
    available = get_models()
    if not available:
        return requested_model or DEFAULT_MODEL

    normalized = {_normalize_model_name(m): m for m in available}

    candidates = []
    if requested_model:
        candidates.append(requested_model)
    candidates.append(DEFAULT_MODEL)
    candidates.extend(MODEL_PREFERENCE)

    # Exact / prefix / base-name matching.
    for candidate in candidates:
        c = _normalize_model_name(candidate)
        if c in normalized:
            return normalized[c]

        for avail_norm, avail_raw in normalized.items():
            if avail_norm.startswith(c):
                return avail_raw

        c_base = c.split(":", 1)[0]
        for avail_norm, avail_raw in normalized.items():
            if avail_norm.split(":", 1)[0] == c_base:
                return avail_raw

    # Fallback to first installed model.
    return available[0]


# =========================================================================
# Core generation
# =========================================================================

def generate(prompt: str, model: str = DEFAULT_MODEL) -> str | None:
    """
    Send a prompt to Ollama and return the generated text.

    Args:
        prompt: The text prompt to send.
        model: Model name (default: auto-selected from local availability).

    Returns:
        Generated text string, or None if Ollama is unavailable.
    """
    try:
        selected_model = select_model(model)
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": selected_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.15,      # Lower = less hallucination
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": 500,       # Max tokens
                },
            },
            timeout=TIMEOUT,
        )

        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return None

    except (requests.ConnectionError, requests.Timeout):
        return None


# =========================================================================
# Structured event extraction
# =========================================================================

EVENT_EXTRACTION_PROMPT = """You are a helpful assistant that extracts structured events from conversation text.

Extract ALL events (meetings, tasks, medications) from the text below.
Return ONLY a valid JSON array. Each event must have these fields:
- "type": one of "meeting", "task", "medication"
- "raw_date": the date text found (e.g. "tomorrow", "March 15") or null
- "time": the time text found (e.g. "10 AM") or null
- "person": person name mentioned or null
- "description": short description of the event

TEXT:
{text}

RESPOND WITH ONLY THE JSON ARRAY, NO OTHER TEXT:"""


def extract_events_llm(text: str) -> list[dict] | None:
    """
    Use LLM to extract structured events from text.

    Returns:
        List of event dicts, or None if LLM is unavailable.
    """
    prompt = EVENT_EXTRACTION_PROMPT.format(text=text)
    response = generate(prompt)

    if not response:
        return None

    # Try to parse JSON from the response
    return _parse_json_array(response)


# =========================================================================
# Summarization
# =========================================================================

SUMMARY_PROMPT = """You are a helpful assistant for an Alzheimer's patient's caregiver.

Summarize the following conversation in 3-4 short, clear bullet points.
Focus on: appointments, medications, tasks, and visitors.
Keep it simple and easy to understand.

CONVERSATION:
{text}

SUMMARY (bullet points):"""


def summarize_llm(text: str) -> str | None:
    """
    Use LLM to generate a conversation summary.

    Returns:
        Summary string, or None if LLM is unavailable.
    """
    prompt = SUMMARY_PROMPT.format(text=text)
    return generate(prompt)


# =========================================================================
# Query answering
# =========================================================================

QUERY_PROMPT = """You are a memory assistant for an Alzheimer's patient.
Answer the question based ONLY on the memory data provided below.
Provide a clear, simple answer.
If the information is not in the data, say "I don't have that information in my memory right now."

Return the response in JSON format with these fields:
- "answer": your spoken-style answer
- "confidence": "high", "medium", or "low"
- "source_count": number of memories used

MEMORY DATA:
{context}

QUESTION: {question}

JSON RESPONSE:"""


def answer_query_llm(question: str, context: str) -> dict | None:
    """
    Use LLM to answer a question based on memory context.
    Returns a dict with structured answer.
    """
    prompt = QUERY_PROMPT.format(question=question, context=context)
    response = generate(prompt)

    if not response:
        return None

    # Try to parse JSON from the response
    return _parse_json_dict(response)


# =========================================================================
# Event Refinement
# =========================================================================

EVENT_REFINEMENT_PROMPT = """You are a medical safety auditor for an Alzheimer's assistant.
I have a conversation transcript and some events my system already extracted.
Your job is to find MISSED events, especially safety warnings or medical instructions.

TRANSCRIPT:
{text}

EXTRACTED SO FAR:
{extracted}

Return ONLY a JSON array of NEW events found (do not repeat existing ones).
Format each as:
- "type": "meeting", "task", "medication", or "warning"
- "description": clear summary
- "importance": score 1-10 (10 for life safety)

If no new events, return [].
JSON ARRAY:"""


def refine_events_llm(text: str, current_events: list[dict]) -> list[dict]:
    """
    Use LLM to find missed events or safety warnings.
    """
    events_str = json.dumps(current_events, indent=2)
    prompt = EVENT_REFINEMENT_PROMPT.format(text=text, extracted=events_str)
    response = generate(prompt)
    
    if not response:
        return []
        
    return _parse_json_array(response) or []


# =========================================================================
# Memory Validation (Hybrid AI — LLM validates rule-based extraction)
# =========================================================================

VALIDATION_PROMPT = """You are a medical safety auditor for an Alzheimer's patient memory assistant.

I have a conversation transcript and events my rule-based system extracted.
Your job is to VALIDATE and REFINE these events — NOT to invent new ones.

STRICT RULES:
- Do NOT invent dates, times, or events not present in the transcript.
- Do NOT hallucinate people or medications not mentioned.
- You may CORRECT obvious extraction errors (e.g., wrong date parsing).
- You may IMPROVE clarity of descriptions and titles.
- You must CATEGORIZE each event: medical, task, family, safety.
- You must assign PRIORITY: high, medium, or low.
- Flag any RISK items (missed medication, safety concern, urgent appointment).

TRANSCRIPT:
{transcript}

EXTRACTED EVENTS:
{events}

Return ONLY valid JSON with this exact structure:
{{
  "validated_events": [
    {{
      "type": "meeting|task|medication|warning",
      "description": "clear, refined description",
      "category": "medical|task|family|safety",
      "priority": "high|medium|low",
      "raw_date": "date text from transcript or null",
      "time": "time text from transcript or null",
      "person": "person name or null",
      "validated": true
    }}
  ],
  "refined_reminders": [
    {{
      "title": "clear reminder title",
      "category": "medical|task|family|safety",
      "priority": "high|medium|low",
      "raw_date": "date from transcript",
      "time": "time from transcript or null"
    }}
  ],
  "ignored_items": ["reason for any removed events"],
  "risk_flags": ["any safety concerns or urgent items"],
  "clean_summary": "A calm, clear 2-3 sentence summary for the patient"
}}

JSON RESPONSE:"""


def validate_memory(
    transcript: str,
    extracted_events: list[dict],
) -> dict | None:
    """
    LLM validation layer: validates rule-based extraction results.

    Args:
        transcript: Original conversation text.
        extracted_events: Events from rule-based extraction.

    Returns:
        Dict with validated_events, refined_reminders, ignored_items,
        risk_flags, and clean_summary. None if LLM unavailable.
    """
    events_str = json.dumps(extracted_events, indent=2, default=str)
    prompt = VALIDATION_PROMPT.format(
        transcript=transcript,
        events=events_str,
    )
    response = generate(prompt)

    if not response:
        return None

    result = _parse_json_dict(response)
    if not result:
        return None

    # Ensure required keys exist with defaults
    result.setdefault("validated_events", [])
    result.setdefault("refined_reminders", [])
    result.setdefault("ignored_items", [])
    result.setdefault("risk_flags", [])
    result.setdefault("clean_summary", "")

    return result


# =========================================================================
# Chat With Memory (LLM-powered conversational memory assistant)
# =========================================================================

CHAT_MEMORY_PROMPT = """You are a memory assistant for an Alzheimer's patient.
Your name is "Memory Assistant". You are calm, kind, and helpful.

STRICT RULES:
- Answer ONLY using the MEMORY CONTEXT provided below.
- If the information is NOT in the context, say: "I don't have that information in my memory right now, but I can help you check with your family."
- NEVER make up information, dates, names, or events.
- Use simple, short sentences. Be warm and reassuring.
- If the question is about medication, appointments, or safety — emphasize the urgency clearly but calmly.
- Reference specific memories when answering (e.g., "On Tuesday, your son David mentioned...").

MEMORY CONTEXT:
{context}

PATIENT'S QUESTION: {question}

Return ONLY valid JSON:
{{
  "answer": "Your calm, clear answer here",
  "related_events": ["brief event reference 1", "brief event reference 2"],
  "confidence": "high|medium|low"
}}

JSON RESPONSE:"""


def chat_with_memory(question: str, memory_context: str) -> dict | None:
    """
    LLM-powered conversational memory Q&A.

    Args:
        question: Patient's natural language question.
        memory_context: Formatted string of relevant memories
                        (ranked by importance + recency + similarity).

    Returns:
        Dict with answer, related_events, confidence.
        None if LLM unavailable.
    """
    prompt = CHAT_MEMORY_PROMPT.format(
        question=question,
        context=memory_context,
    )
    response = generate(prompt)

    if not response:
        return None

    result = _parse_json_dict(response)
    if result:
        result.setdefault("answer", "I'm having trouble thinking right now. Please ask again.")
        result.setdefault("related_events", [])
        result.setdefault("confidence", "low")
        return result

    # If JSON parsing failed, use raw text as answer
    return {
        "answer": response,
        "related_events": [],
        "confidence": "low",
    }



def _parse_json_array(text: str) -> list[dict] | None:
    """
    Try to extract a JSON array from LLM output.
    LLMs sometimes wrap JSON in markdown code blocks or add extra text.
    """
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _parse_json_dict(text: str) -> dict | None:
    """Try to extract a JSON object from LLM output."""
    try:
        # Fast path
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex path
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# =========================================================================
# Quick test
# =========================================================================

if __name__ == "__main__":
    print("=== LLM Engine Status ===")
    print(f"  Ollama URL: {OLLAMA_URL}")
    print(f"  Available: {is_available()}")
    print(f"  Models: {get_models()}")

    if is_available():
        print("\n=== Test Generation ===")
        result = generate("Say hello in one sentence.")
        print(f"  Response: {result}")

        sample = (
            "I have a doctor appointment tomorrow at 10 AM. "
            "Don't forget to take your medicine after breakfast. "
            "David is visiting this weekend."
        )

        print("\n=== Test Event Extraction ===")
        events = extract_events_llm(sample)
        if events:
            print(json.dumps(events, indent=2))
        else:
            print("  No events extracted")

        print("\n=== Test Summary ===")
        summary = summarize_llm(sample)
        print(f"  {summary}")
    else:
        print("\n  Ollama is not running. Install from: https://ollama.com/download")
        print("  Then run: ollama pull phi3")
