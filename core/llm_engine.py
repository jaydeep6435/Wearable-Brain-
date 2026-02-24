"""
LLM Engine Module — Local LLM via Ollama
==========================================
Connects to a local Ollama instance (http://localhost:11434) to use
lightweight LLMs (phi3, mistral) for improved NLP tasks.

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
  2. Pull a model:   ollama pull phi3
  3. Start server:   ollama serve  (or runs automatically)
"""

import json
import re
import requests
from datetime import datetime


# =========================================================================
# Configuration
# =========================================================================

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi3"      # Lightweight 3.8B — fast on CPU
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


# =========================================================================
# Core generation
# =========================================================================

def generate(prompt: str, model: str = DEFAULT_MODEL) -> str | None:
    """
    Send a prompt to Ollama and return the generated text.

    Args:
        prompt: The text prompt to send.
        model: Model name (default: phi3).

    Returns:
        Generated text string, or None if Ollama is unavailable.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,     # Lower = more factual
                    "num_predict": 500,      # Max tokens
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
If the answer is not in the data, say "I don't have that information in memory."
Keep your answer short and clear.

MEMORY DATA:
{context}

QUESTION: {question}

ANSWER:"""


def answer_query_llm(question: str, context: str) -> str | None:
    """
    Use LLM to answer a question based on memory context.

    Args:
        question: The user's natural language question.
        context: Formatted string of stored events/memories.

    Returns:
        Answer string, or None if LLM is unavailable.
    """
    prompt = QUERY_PROMPT.format(question=question, context=context)
    return generate(prompt)


# =========================================================================
# Internal helpers
# =========================================================================

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
