"""
Semantic Search (Fallback)
==========================

Provides a minimal EmbeddingSearch interface used by the engine.
"""

from __future__ import annotations


class EmbeddingSearch:
    def __init__(self):
        self._available = False

    def is_available(self) -> bool:
        return self._available

    def search(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """Keyword overlap fallback search with stable output schema."""
        q_terms = {t for t in (query or "").lower().split() if len(t) > 2}
        scored = []
        for doc in documents or []:
            text = f"{doc.get('description', '')} {doc.get('type', '')} {doc.get('person', '')}".lower()
            score = 0.0
            for t in q_terms:
                if t in text:
                    score += 1.0
            if score > 0:
                scored.append({"document": doc, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(1, int(top_k))]
