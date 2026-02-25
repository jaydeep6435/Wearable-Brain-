"""
Memory Ranker — Alzheimer-Aware Prioritization
=================================================
Scores events by medical/safety importance and detects
recurring conversation patterns for an Alzheimer's patient.

Scoring rules:
  +5 — Medication instructions (take medicine, refill prescription)
  +5 — Doctor/medical appointments
  +4 — Safety-related instructions (don't leave stove on, etc.)
  +3 — Repeated/recurring reminders
  +2 — Family visits
  +1 — General notes/tasks

No ML models — pure keyword/regex logic.
"""

import re
from datetime import datetime, timedelta


# ── Importance Scoring ────────────────────────────────────────

# Keywords by category (checked against event type + description)
_MEDICATION_KEYWORDS = [
    "medicine", "medication", "pill", "tablet", "prescription",
    "refill", "pharmacy", "dose", "dosage", "drug",
]

_MEDICAL_APPOINTMENT_KEYWORDS = [
    "doctor", "dentist", "hospital", "clinic", "therapist",
    "appointment", "checkup", "check-up", "specialist", "surgeon",
]

_SAFETY_KEYWORDS = [
    "stove", "oven", "fire", "gas", "lock", "door", "fall",
    "emergency", "careful", "dangerous", "don't forget",
    "safety", "alert", "warning",
]

_FAMILY_KEYWORDS = [
    "son", "daughter", "wife", "husband", "brother", "sister",
    "grandchild", "family", "visiting", "visit",
]

# Recurring phrase patterns (normalized)
_RECURRING_PHRASES = [
    "take your medicine",
    "take medicine",
    "doctor appointment",
    "call pharmacy",
    "refill prescription",
    "don't forget",
    "remember to",
]


def score_event(event: dict) -> int:
    """
    Calculate importance score for a single event.

    Args:
        event: Event dict with 'type' and 'description' keys.

    Returns:
        Integer importance score (0-10).
    """
    score = 0
    event_type = (event.get("type") or "").lower()
    desc = (event.get("description") or "").lower()
    combined = f"{event_type} {desc}"

    # Medication: +5
    if event_type == "medication" or any(kw in combined for kw in _MEDICATION_KEYWORDS):
        score = max(score, 5)

    # Medical appointment: +5
    if event_type == "meeting" and any(kw in combined for kw in _MEDICAL_APPOINTMENT_KEYWORDS):
        score = max(score, 5)

    # Safety: +4
    if any(kw in combined for kw in _SAFETY_KEYWORDS):
        score = max(score, 4)

    # Family: +2
    if any(kw in combined for kw in _FAMILY_KEYWORDS):
        score = max(score, 2)

    # General tasks: +1 (baseline for anything with a type)
    if score == 0 and event_type:
        score = 1

    return score


def score_events(events: list[dict]) -> list[dict]:
    """
    Add importance_score to each event in a list.

    Args:
        events: List of event dicts.

    Returns:
        Same list with 'importance_score' key added to each.
    """
    for event in events:
        event["importance_score"] = score_event(event)
    return events


# ── Recurrence Detection ──────────────────────────────────────

def detect_patterns(text: str, repo=None) -> list[dict]:
    """
    Detect recurring conversation phrases and update pattern counts.

    Args:
        text: Conversation text to scan.
        repo: Repository instance (optional). If provided, updates
              the memory_patterns table.

    Returns:
        List of detected patterns: [{phrase, category, frequency}]
    """
    text_lower = text.lower()
    detected = []

    for phrase in _RECURRING_PHRASES:
        if phrase in text_lower:
            category = _categorize_phrase(phrase)
            freq = 1

            if repo and hasattr(repo, "increment_pattern"):
                freq = repo.increment_pattern(phrase, category)

            detected.append({
                "phrase": phrase,
                "category": category,
                "frequency": freq,
            })

    return detected


def _categorize_phrase(phrase: str) -> str:
    """Categorize a recurring phrase."""
    phrase = phrase.lower()
    if any(kw in phrase for kw in ["medicine", "prescription", "pharmacy"]):
        return "medication"
    if any(kw in phrase for kw in ["doctor", "appointment"]):
        return "medical"
    if any(kw in phrase for kw in ["forget", "remember"]):
        return "reminder"
    return "general"


# ── Weighted Ranking ──────────────────────────────────────────

# Default weights (configurable)
WEIGHT_SEMANTIC = 0.5
WEIGHT_IMPORTANCE = 0.3
WEIGHT_RECENCY = 0.2


def rank_results(
    results: list[dict],
    weight_semantic: float = WEIGHT_SEMANTIC,
    weight_importance: float = WEIGHT_IMPORTANCE,
    weight_recency: float = WEIGHT_RECENCY,
) -> list[dict]:
    """
    Re-rank search results using blended scoring.

    final_score = w_semantic * similarity + w_importance * importance + w_recency * recency

    Args:
        results: List of search result dicts with 'score' and 'document' keys.
        weight_semantic: Weight for semantic similarity (0-1).
        weight_importance: Weight for importance score (0-1).
        weight_recency: Weight for recency (0-1).

    Returns:
        Results sorted by blended score (highest first).
    """
    now = datetime.now()

    for r in results:
        doc = r.get("document", {})

        # Semantic similarity (already 0-1)
        semantic = r.get("score", 0.0)

        # Importance (normalize 0-10 to 0-1)
        importance = doc.get("importance_score", 0) / 10.0

        # Recency (decay over 30 days)
        recency = _compute_recency(doc, now)

        # Blend
        r["blended_score"] = (
            weight_semantic * semantic
            + weight_importance * importance
            + weight_recency * recency
        )

    # Sort by blended score
    results.sort(key=lambda x: x.get("blended_score", 0), reverse=True)
    return results


def _compute_recency(doc: dict, now: datetime) -> float:
    """
    Compute recency score (0-1) based on recorded_at timestamp.
    1.0 = today, decays to ~0.0 after 30 days.
    """
    recorded = doc.get("recorded_at") or doc.get("timestamp")
    if not recorded:
        return 0.5  # Default for unknown

    try:
        if isinstance(recorded, str):
            # Handle ISO format
            recorded_dt = datetime.fromisoformat(recorded.replace("Z", "+00:00"))
            # Remove timezone if present for comparison
            if recorded_dt.tzinfo:
                recorded_dt = recorded_dt.replace(tzinfo=None)
        else:
            return 0.5

        days_ago = (now - recorded_dt).total_seconds() / 86400
        # Exponential decay: 1.0 today, ~0.37 at 30 days
        return max(0.0, min(1.0, 2.718 ** (-days_ago / 30.0)))

    except (ValueError, TypeError):
        return 0.5


# ── Urgency Detection ─────────────────────────────────────────

def get_urgent_items(repo, hours: int = 24) -> list[dict]:
    """
    Find events that are urgent (within `hours` from now).

    Urgent = medication or appointment with parsed_date within the window.

    Args:
        repo: Repository instance with get_all_events().
        hours: Look-ahead window in hours.

    Returns:
        List of urgent event dicts with 'urgent_flag' = True.
    """
    if not hasattr(repo, "get_all_events"):
        return []

    now = datetime.now()
    cutoff = now + timedelta(hours=hours)
    urgent = []

    all_events = repo.get_all_events()
    for event in all_events:
        ev = dict(event)
        event_type = (ev.get("type") or "").lower()

        # Only check medication and meeting types
        if event_type not in ("medication", "meeting", "appointment"):
            continue

        # Check parsed_date
        parsed_date = ev.get("parsed_date")
        if not parsed_date:
            # No date = assume it's recurring/important
            ev["urgent_flag"] = True
            ev["urgency_reason"] = f"No date — {event_type} reminder"
            urgent.append(ev)
            continue

        try:
            event_dt = datetime.fromisoformat(parsed_date)
            if now <= event_dt <= cutoff:
                ev["urgent_flag"] = True
                ev["urgency_reason"] = f"{event_type} within {hours}h"
                urgent.append(ev)
        except (ValueError, TypeError):
            pass

    # Sort by importance
    urgent.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
    return urgent
