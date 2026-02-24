"""
Event Extractor Module — Text → Structured Events (v3)
========================================================
Scans transcribed text and returns structured JSON events with fields:
  type, raw_date, parsed_date, time, parsed_time, person, description

Detection methods:
  - Regex patterns for dates, times, meetings, tasks, medications
  - Person detection via capitalized names and title prefixes
  - Cross-referencing within the same sentence for richer output
  - Real date parsing via dateparser (Day 4 upgrade)

All detection is regex + keyword based — no ML model required.
"""

import re
from core.date_parser import parse_date, parse_time


# =========================================================================
# Pattern definitions
# =========================================================================

# -- Date patterns --------------------------------------------------------
DATE_PATTERNS = [
    r'\b(today|tonight|tomorrow|yesterday)\b',
    r'\b(next|this|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)\b',
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)'
    r'\s+\d{1,2}(?:,?\s*\d{4})?\b',
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b\d{4}-\d{2}-\d{2}\b',
]

# -- Time patterns --------------------------------------------------------
TIME_PATTERNS = [
    r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\b',
    r'\b\d{1,2}\s*(?:am|pm|AM|PM)\b',
    r'\b(?:at|by|around|before|after)\s+(?:noon|midnight)\b',
]

# -- Person patterns (names, titles) --------------------------------------
PERSON_PATTERNS = [
    r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+\b',          # Dr. Smith
    r'\bwith\s+([A-Z][a-z]{2,})\b',                          # with Rahul
    r'\b(?:son|daughter|brother|sister|wife|husband)\s+([A-Z][a-z]+)\b',  # son David
    r'\b([A-Z][a-z]{2,})\s+(?:called|said|told|wants|is visiting)\b',     # David called
]

# -- Meeting / appointment patterns ---------------------------------------
MEETING_PATTERNS = [
    r'\b(?:doctor|dentist|therapist|clinic|hospital)\s*(?:appointment|visit)?\b',
    r'\bappointment\s+(?:with|at|for)\s+.{3,30}?(?=[.!?,]|$)',
    r'\bmeeting\s+(?:with|at|about)\s+.{3,30}?(?=[.!?,]|$)',
    r'\b(?:visit|visiting)\s+(?:with\s+)?.{3,20}?(?=[.!?,]|$)',
]

# -- Task / to-do patterns ------------------------------------------------
TASK_PATTERNS = [
    r"\b(?:need to|have to|must|should|don't forget to|remember to|going to)\s+.{5,60}?(?=[.!?,]|$)",
    r'\b(?:call|pick up|buy|get|bring|send|submit|finish|complete|prepare)\s+.{3,50}?(?=[.!?,]|$)',
]

# -- Medication patterns ---------------------------------------------------
MEDICATION_PATTERNS = [
    r'\b(?:take|took)\s+(?:your\s+|the\s+)?(?:medicine|medication|pill|pills|tablet|tablets)\b',
    r'\b(?:refill|renew)\s+(?:the\s+)?(?:prescription|medication|medicine)\b',
    r'\bmedicine\s+(?:after|before|with|at)\s+\w+\b',
]


# =========================================================================
# Internal helpers
# =========================================================================

def _get_sentence(text: str, pos: int) -> str:
    """Return the full sentence surrounding position `pos`."""
    start = text.rfind('.', 0, pos)
    end = text.find('.', pos)
    return text[start + 1: end if end != -1 else len(text)].strip()


def _find_in_sentence(sentence: str, patterns: list[str]) -> str | None:
    """Return the first match from patterns found in the sentence, or None."""
    for pat in patterns:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def _find_persons(sentence: str) -> list[str]:
    """Extract person names from a sentence."""
    persons = []
    seen = set()
    for pat in PERSON_PATTERNS:
        for m in re.finditer(pat, sentence):
            # Use the first captured group if it exists, else the full match
            name = (m.group(1) if m.lastindex else m.group(0)).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                persons.append(name)
    return persons


def _find_all_matches(text: str, patterns: list[str]) -> list[tuple[str, str]]:
    """
    Return list of (matched_value, sentence) tuples for all pattern matches.
    Deduplicates by matched value.
    """
    results = []
    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            value = m.group(0).strip()
            if value.lower() not in seen:
                seen.add(value.lower())
                sentence = _get_sentence(text, m.start())
                results.append((value, sentence))
    return results


def _build_event(event_type: str, value: str, sentence: str) -> dict:
    """Build a structured event dict with parsed date/time fields."""
    raw_date = _find_in_sentence(sentence, DATE_PATTERNS)
    raw_time = _find_in_sentence(sentence, TIME_PATTERNS)
    persons = _find_persons(sentence)
    desc = value[0].upper() + value[1:]  # Capitalize first letter

    return {
        "type": event_type,
        "raw_date": raw_date,
        "parsed_date": parse_date(raw_date),
        "time": raw_time,
        "parsed_time": parse_time(raw_time),
        "person": persons[0] if persons else None,
        "description": desc,
    }


# =========================================================================
# Public API
# =========================================================================

def extract_structured_events(text: str, use_llm: bool = False) -> list[dict]:
    """
    Extract structured events from the given text.

    Args:
        text    : Conversation text to extract events from.
        use_llm : If True, use LLM as fallback when rule-based finds few events.

    Returns a list of event dicts, each with:
        "type"        : meeting | task | medication
        "raw_date"    : original date text or None
        "parsed_date" : ISO date string (YYYY-MM-DD) or None
        "time"        : original time text or None
        "parsed_time" : normalized time (HH:MM) or None
        "person"      : detected person name or None
        "description" : human-readable description of the event
    """
    events = []
    seen_descriptions = set()

    # --- Process meeting patterns -----------------------------------------
    for value, sentence in _find_all_matches(text, MEETING_PATTERNS):
        event = _build_event("meeting", value, sentence)
        if event["description"].lower() not in seen_descriptions:
            seen_descriptions.add(event["description"].lower())
            events.append(event)

    # --- Process task patterns --------------------------------------------
    for value, sentence in _find_all_matches(text, TASK_PATTERNS):
        event = _build_event("task", value, sentence)
        if event["description"].lower() not in seen_descriptions:
            seen_descriptions.add(event["description"].lower())
            events.append(event)

    # --- Process medication patterns --------------------------------------
    for value, sentence in _find_all_matches(text, MEDICATION_PATTERNS):
        event = _build_event("medication", value, sentence)
        if event["description"].lower() not in seen_descriptions:
            seen_descriptions.add(event["description"].lower())
            events.append(event)

    # --- LLM enhancement (always runs when enabled) -------------------------
    if use_llm:
        try:
            from core.llm_engine import extract_events_llm, is_available
            if is_available():
                llm_events = extract_events_llm(text)
                if llm_events:
                    for e in llm_events:
                        desc = (e.get("description") or "").lower()
                        if desc and desc not in seen_descriptions:
                            seen_descriptions.add(desc)
                            # Ensure required fields exist
                            e.setdefault("type", "task")
                            e.setdefault("raw_date", None)
                            e.setdefault("parsed_date", parse_date(e.get("raw_date")))
                            e.setdefault("time", None)
                            e.setdefault("parsed_time", parse_time(e.get("time")))
                            e.setdefault("person", None)
                            e["source"] = "llm"
                            events.append(e)
        except Exception:
            pass  # LLM failed — keep rule-based results

    return events


# Legacy API (backward compatibility with Day 1 tests)
def extract_events(text: str) -> list[dict]:
    """
    Legacy wrapper: converts structured events back to the old format
    with {type, value, context} keys.
    """
    structured = extract_structured_events(text)
    legacy = []
    for e in structured:
        legacy.append({
            "type": e["type"],
            "value": e["description"],
            "context": e["description"],
        })
    return legacy


# -------------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    sample = (
        "Good morning! I have a doctor appointment tomorrow at 10 AM. "
        "Don't forget to take your medicine after breakfast. "
        "We need to call the pharmacy to refill the prescription. "
        "Your son David is visiting this weekend. "
        "Remember to do your morning exercises before lunch. "
        "The meeting with Dr. Smith is on March 15 at 11 AM. "
        "Rahul said he will come for dinner next Sunday."
    )
    print("--- Structured Events (JSON) ---")
    events = extract_structured_events(sample)
    print(json.dumps(events, indent=2))
