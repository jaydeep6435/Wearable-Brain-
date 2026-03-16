"""
Cognitive Reinforcement Helpers
===============================

Simple persistence-friendly helpers for reminder reinforcement paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def get_reinforcement_items(repo, interval_hours: int = 12) -> list[dict]:
    """Return important events that may need re-display."""
    cutoff = datetime.now() - timedelta(hours=max(1, int(interval_hours)))
    items = []
    for e in repo.get_all_events() if repo else []:
        if int(e.get("importance_score", 0) or 0) < 4:
            continue
        rec = e.get("recorded_at")
        try:
            if rec and datetime.fromisoformat(rec) < cutoff:
                items.append(dict(e))
        except Exception:
            items.append(dict(e))
    return items[:10]


def mark_shown(repo, event_id: str) -> None:
    """Best-effort mark shown state."""
    if not repo or not event_id:
        return
    try:
        repo.db.execute(
            "INSERT OR REPLACE INTO memory_reinforcement (event_id, last_shown, shown_count) "
            "VALUES (?, ?, COALESCE((SELECT shown_count FROM memory_reinforcement WHERE event_id=?), 0)+1)",
            (event_id, datetime.now().isoformat(), event_id),
        )
    except Exception:
        return


def check_escalation(repo, max_level: int = 3) -> list[dict]:
    """Return events eligible for escalation; no destructive updates."""
    out = []
    for e in repo.get_all_events() if repo else []:
        lvl = int(e.get("escalation_level", 0) or 0)
        if lvl < max(0, int(max_level)) and int(e.get("importance_score", 0) or 0) >= 4:
            item = dict(e)
            item["next_escalation_level"] = lvl + 1
            out.append(item)
    return out[:10]


def generate_daily_brief(repo) -> dict:
    """Generate a calm, minimal daily brief structure."""
    urgent = []
    try:
        from core.memory_ranker import get_urgent_items

        urgent = get_urgent_items(repo, hours=24)
    except Exception:
        urgent = []

    return {
        "greeting": "Here is your daily memory brief.",
        "urgent_items": urgent[:5],
        "patterns": repo.get_patterns(min_frequency=2) if repo and hasattr(repo, "get_patterns") else [],
        "summary_text": "You can review upcoming important items and recent conversation highlights.",
        "closing": "You are doing well. Let's take it one step at a time.",
    }
