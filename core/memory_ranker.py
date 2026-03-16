"""
Memory Ranker
=============

Lightweight scoring and ranking helpers used by the engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def score_events(events: list[dict]) -> list[dict]:
    """Assign a simple importance score in-place."""
    keywords_high = {"doctor", "medicine", "appointment", "urgent", "hospital"}
    for event in events or []:
        text = f"{event.get('type', '')} {event.get('description', '')}".lower()
        score = 1
        if any(k in text for k in keywords_high):
            score += 2
        if event.get("parsed_date") or event.get("parsed_time"):
            score += 1
        if event.get("person"):
            score += 1
        event["importance_score"] = int(min(5, score))
    return events


def detect_patterns(text: str, repo=None) -> None:
    """Best-effort phrase pattern tracking; no-op if repo is unavailable."""
    if not text or repo is None or not hasattr(repo, "save_pattern"):
        return
    words = [w.strip(".,!?;:").lower() for w in text.split() if len(w) > 4]
    for w in set(words):
        try:
            repo.save_pattern(phrase=w, category="keyword")
        except Exception:
            return


def get_urgent_items(repo, hours: int = 24) -> list[dict]:
    """Return high-priority upcoming events from repository data."""
    items = []
    now = datetime.now()
    upper = now + timedelta(hours=max(1, int(hours)))

    for e in repo.get_all_events() if repo else []:
        date_s = e.get("parsed_date")
        time_s = e.get("parsed_time")
        if not date_s:
            continue

        dt = None
        try:
            if time_s:
                dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M")
            else:
                dt = datetime.strptime(date_s, "%Y-%m-%d")
        except Exception:
            continue

        if now <= dt <= upper:
            item = dict(e)
            item["urgent_flag"] = True
            items.append(item)

    items.sort(key=lambda x: (x.get("parsed_date", ""), x.get("parsed_time", "")))
    return items


def rank_results(results: list[dict]) -> list[dict]:
    """Sort semantic results by score descending if present."""
    return sorted(results or [], key=lambda r: float(r.get("score", 0.0)), reverse=True)
