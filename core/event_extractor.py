"""
Event Extractor Module — Text → Structured Events
===================================================
Scans transcribed text for actionable information:
  • Dates         (tomorrow, next Monday, March 5, 2025-03-05 …)
  • Times         (10 AM, 3:30 PM, at noon …)
  • Meetings      (doctor appointment, meeting with …)
  • Tasks / To-Dos (call pharmacy, pick up groceries …)
  • Medications   (take medicine, refill prescription …)

All detection is regex + keyword based — no ML model required.
"""

import re


# ═══════════════════════════════════════════════════════════════════════════
# Pattern definitions
# ═══════════════════════════════════════════════════════════════════════════

# ── Date patterns ─────────────────────────────────────────────────────────
DATE_PATTERNS = [
    # Relative dates
    r'\b(today|tonight|tomorrow|yesterday)\b',
    r'\b(next|this|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)\b',
    # Month + day (optional year)
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)'
    r'\s+\d{1,2}(?:,?\s*\d{4})?\b',
    # Numeric dates
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b\d{4}-\d{2}-\d{2}\b',
]

# ── Time patterns ─────────────────────────────────────────────────────────
TIME_PATTERNS = [
    r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\b',
    r'\b\d{1,2}\s*(?:am|pm|AM|PM)\b',
    r'\b(?:at|by|around|before|after)\s+(?:noon|midnight)\b',
    r'\b(?:morning|afternoon|evening)\b',
]

# ── Meeting / appointment keywords ───────────────────────────────────────
MEETING_PATTERNS = [
    r'\b(?:doctor|dentist|therapist|clinic|hospital)\s*(?:appointment|visit)?\b',
    r'\bappointment\s+(?:with|at|for)\s+\w+\b',
    r'\bmeeting\s+(?:with|at|about)\s+\w+\b',
    r'\b(?:visit|visiting)\s+(?:with\s+)?\w+\b',
]

# ── Task / to-do patterns ────────────────────────────────────────────────
TASK_PATTERNS = [
    r'\b(?:need to|have to|must|should|don\'t forget to|remember to|going to)\s+.{5,60}?(?=[.!?,]|$)',
    r'\b(?:call|pick up|buy|get|bring|send|submit|finish|complete|prepare)\s+.{3,50}?(?=[.!?,]|$)',
]

# ── Medication patterns ──────────────────────────────────────────────────
MEDICATION_PATTERNS = [
    r'\b(?:take|took)\s+(?:your\s+|the\s+)?(?:medicine|medication|pill|pills|tablet|tablets)\b',
    r'\b(?:refill|renew)\s+(?:the\s+)?(?:prescription|medication|medicine)\b',
    r'\bmedicine\s+(?:after|before|with|at)\s+\w+\b',
]


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _find_matches(text: str, patterns: list[str], event_type: str) -> list[dict]:
    """
    Run a list of regex patterns against the text and return
    deduplicated matches as event dicts.
    """
    events = []
    seen = set()

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(0).strip()
            if value.lower() not in seen:
                seen.add(value.lower())

                # Grab surrounding context (the full sentence)
                start = text.rfind('.', 0, match.start())
                end = text.find('.', match.end())
                context = text[start + 1: end if end != -1 else len(text)].strip()

                events.append({
                    "type": event_type,
                    "value": value,
                    "context": context,
                })
    return events


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def extract_events(text: str) -> list[dict]:
    """
    Extract all structured events from the given text.

    Args:
        text : The full transcription text.

    Returns:
        List of event dicts, each with:
            "type"    — one of: date, time, meeting, task, medication
            "value"   — the matched text
            "context" — the sentence surrounding the match
    """
    events = []

    events.extend(_find_matches(text, DATE_PATTERNS, "date"))
    events.extend(_find_matches(text, TIME_PATTERNS, "time"))
    events.extend(_find_matches(text, MEETING_PATTERNS, "meeting"))
    events.extend(_find_matches(text, TASK_PATTERNS, "task"))
    events.extend(_find_matches(text, MEDICATION_PATTERNS, "medication"))

    return events


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Good morning! I have a doctor appointment tomorrow at 10 AM. "
        "Don't forget to take your medicine after breakfast. "
        "We need to call the pharmacy to refill the prescription. "
        "Your son is visiting this weekend. "
        "Remember to do your morning exercises before lunch. "
        "The meeting with Dr. Smith is on March 15."
    )
    print("--- Extracted Events ---")
    for e in extract_events(sample):
        print(f"  [{e['type'].upper():>10}]  {e['value']}")
        print(f"             Context: {e['context']}\n")
