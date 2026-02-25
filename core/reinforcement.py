"""
Reinforcement Engine — Cognitive Simplification for Alzheimer's
================================================================
Reduces cognitive load by:
  - Simplifying summaries to N key points
  - Tracking whether critical items have been shown to user
  - Escalating missed/overdue events
  - Generating calm, structured daily briefs

No ML models — pure logic with datetime checks.
"""

from datetime import datetime, timedelta


# ── Simplified Summary ────────────────────────────────────────

def simplify_summary(summary: str, max_points: int = 2) -> str:
    """
    Trim a summary to at most `max_points` sentences.

    Keeps the first N sentences for simplicity and readability.
    Uses period/exclamation/question mark as sentence boundaries.

    Args:
        summary: Full summary text.
        max_points: Maximum sentences to keep.

    Returns:
        Shortened summary string.
    """
    if not summary:
        return ""

    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary) if s.strip()]

    if len(sentences) <= max_points:
        return summary

    return " ".join(sentences[:max_points])


def filter_events_by_importance(events: list[dict], min_importance: int = 3) -> list[dict]:
    """
    Filter events to only show those above minimum importance.

    Args:
        events: List of event dicts with 'importance_score'.
        min_importance: Minimum score to include.

    Returns:
        Filtered list of high-importance events.
    """
    return [e for e in events if e.get("importance_score", 0) >= min_importance]


# ── Reinforcement Tracking ────────────────────────────────────

def get_reinforcement_items(repo, interval_hours: int = 12) -> list[dict]:
    """
    Find critical events that need to be re-shown to the user.

    An event needs reinforcement if:
      - importance_score >= 5 OR urgent
      - Not shown within `interval_hours`

    Args:
        repo: Repository instance.
        interval_hours: Hours since last show to trigger reinforcement.

    Returns:
        List of event dicts with 'reinforcement_needed' = True.
    """
    if not hasattr(repo, "get_reinforcement_candidates"):
        return []

    candidates = repo.get_reinforcement_candidates(interval_hours)
    for c in candidates:
        c["reinforcement_needed"] = True

    return candidates


def mark_shown(repo, event_id: str) -> None:
    """
    Record that an event was shown to the user.

    Args:
        repo: Repository instance.
        event_id: The event ID that was displayed.
    """
    if hasattr(repo, "mark_reinforcement_shown"):
        repo.mark_reinforcement_shown(event_id)


# ── Escalation Logic ─────────────────────────────────────────

def check_escalation(repo, max_level: int = 3) -> list[dict]:
    """
    Find events that need escalation (missed medication, passed appointments).

    Escalation triggers:
      - Medication with parsed_date in the past and not acknowledged
      - Appointment time passed and not acknowledged
      - Any event marked urgent and not shown

    Increments escalation_level up to max_level.

    Args:
        repo: Repository instance.
        max_level: Maximum escalation level (0-3).

    Returns:
        List of escalated event dicts.
    """
    if not hasattr(repo, "get_escalation_candidates"):
        return []

    candidates = repo.get_escalation_candidates()
    escalated = []
    now = datetime.now()

    for event in candidates:
        ev = dict(event)
        current_level = ev.get("escalation_level", 0)

        if current_level >= max_level:
            ev["escalation_capped"] = True
            escalated.append(ev)
            continue

        # Check if event time has passed
        should_escalate = False
        parsed_date = ev.get("parsed_date")

        if parsed_date:
            try:
                event_dt = datetime.fromisoformat(parsed_date)
                if event_dt < now:
                    should_escalate = True
                    ev["escalation_reason"] = "Event time has passed"
            except (ValueError, TypeError):
                pass

        # No date but high importance = escalate if not shown recently
        if not parsed_date and ev.get("importance_score", 0) >= 5:
            should_escalate = True
            ev["escalation_reason"] = "High importance, no date set"

        if should_escalate and current_level < max_level:
            new_level = current_level + 1
            if hasattr(repo, "escalate_event"):
                repo.escalate_event(ev["id"], new_level)
            ev["escalation_level"] = new_level
            escalated.append(ev)

    return escalated


# ── Daily Brief ───────────────────────────────────────────────

_CALM_TEMPLATES = {
    "greeting": "Here's what's important for you today.",
    "medication": "💊 Remember: {desc}",
    "appointment": "📅 You have: {desc}",
    "reminder": "🔔 {desc}",
    "pattern": "📝 Recurring: {phrase} (mentioned {freq} times)",
    "closing": "That's all for now. Take your time.",
}


def generate_daily_brief(repo, max_items: int = 3) -> dict:
    """
    Generate a calm, structured daily summary.

    Returns:
        Dict with keys:
          - greeting: Opening line
          - urgent_items: List of urgent items (max 3)
          - patterns: Recurring patterns
          - summary_text: Combined brief as single string
          - closing: Closing line
    """
    brief = {
        "greeting": _CALM_TEMPLATES["greeting"],
        "urgent_items": [],
        "patterns": [],
        "summary_text": "",
        "closing": _CALM_TEMPLATES["closing"],
    }

    # Urgent items
    try:
        from core.memory_ranker import get_urgent_items
        urgent = get_urgent_items(repo, hours=24)
        for item in urgent[:max_items]:
            event_type = (item.get("type") or "reminder").lower()
            desc = item.get("description", "Important event")
            template_key = event_type if event_type in _CALM_TEMPLATES else "reminder"
            formatted = _CALM_TEMPLATES[template_key].format(desc=desc)
            brief["urgent_items"].append({
                "text": formatted,
                "type": event_type,
                "importance": item.get("importance_score", 0),
            })
    except Exception:
        pass

    # Recurring patterns
    try:
        if hasattr(repo, "get_patterns"):
            patterns = repo.get_patterns(min_frequency=2)
            for p in patterns[:2]:
                formatted = _CALM_TEMPLATES["pattern"].format(
                    phrase=p["phrase"], freq=p["frequency"]
                )
                brief["patterns"].append({
                    "text": formatted,
                    "phrase": p["phrase"],
                    "frequency": p["frequency"],
                })
    except Exception:
        pass

    # Build summary text
    lines = [brief["greeting"]]
    for item in brief["urgent_items"]:
        lines.append(f"  • {item['text']}")
    for pat in brief["patterns"]:
        lines.append(f"  • {pat['text']}")
    if not brief["urgent_items"] and not brief["patterns"]:
        lines.append("  • Nothing urgent right now. Everything is okay.")
    lines.append(brief["closing"])
    brief["summary_text"] = "\n".join(lines)

    return brief
