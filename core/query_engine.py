"""
Query Engine Module — Natural Language Memory Queries
======================================================
Lets users ask questions about stored conversation memories using
natural language. Uses simple rule-based NLP (keyword matching + regex).

Supported query types:
  - "What meetings do I have tomorrow?"
  - "Did I take medicine?"
  - "What tasks do I have?"
  - "What happened yesterday?"
  - "Search for pharmacy"

No ML models required — fully offline and lightweight.
"""

import re
from typing import Protocol, runtime_checkable


# =========================================================================
# MemoryStore protocol — any object with get_all_events / search_events
# =========================================================================

@runtime_checkable
class MemoryStore(Protocol):
    """Minimal interface for event storage (Repository, adapter, etc.)."""
    def get_all_events(self) -> list[dict]: ...
    def search_events(self, keyword: str) -> list[dict]: ...


# =========================================================================
# Intent definitions — keywords that map to each intent
# =========================================================================

INTENT_KEYWORDS = {
    "meeting": [
        "meeting", "meetings", "appointment", "appointments", "doctor",
        "visit", "visiting", "visitor", "visitors", "who is coming",
        "scheduled", "clinic", "hospital",
    ],
    "task": [
        "task", "tasks", "to do", "todo", "to-do", "remind", "reminder",
        "reminders", "need to", "have to", "should", "must", "pending",
        "things to do", "what should i do",
    ],
    "medication": [
        "medicine", "medication", "medications", "pill", "pills",
        "tablet", "tablets", "prescription", "drug", "drugs",
        "did i take", "take medicine", "took medicine",
    ],
    "summary": [
        "summary", "summarize", "summarise", "what happened",
        "tell me about", "tell me everything", "overview",
        "what did i discuss", "what was discussed", "recap",
    ],
}

# =========================================================================
# Date keywords — map natural language to filter terms
# =========================================================================

DATE_KEYWORDS = {
    "today":     ["today", "this morning", "this afternoon", "this evening", "tonight"],
    "tomorrow":  ["tomorrow", "next day"],
    "yesterday": ["yesterday", "last day"],
    "this week": ["this week", "this weekend"],
    "next week": ["next week"],
}

# Day names for more specific queries
DAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


# =========================================================================
# QueryEngine class
# =========================================================================

class QueryEngine:
    """
    Hybrid query engine: rule-based + semantic search.

    Priority:
      1. Intent detection (meeting, task, medication, summary)
      2. Keyword search (SQL LIKE)
      3. Semantic search (embedding-based, TF-IDF fallback)
      4. LLM fallback (if enabled)

    Usage:
        engine = QueryEngine(memory_manager)
        answer = engine.query("What meetings do I have tomorrow?")
        print(answer)
    """

    def __init__(self, memory: MemoryStore):
        """
        Initialize with any object implementing MemoryStore protocol.

        Args:
            memory: An object with get_all_events() and search_events() methods.
        """
        self.memory = memory
        self._searcher = None       # Lazy-loaded TF-IDF SemanticSearch
        self._emb_searcher = None   # Lazy-loaded EmbeddingSearch
        self._index_version = 0     # Tracks when to rebuild

    # -- Main query method -------------------------------------------------

    def query(self, question: str, use_llm: bool = False) -> str:
        """
        Process a natural language question and return a human-readable answer.

        Args:
            question: The user's question string.
            use_llm:  If True, use LLM when rule-based fails.

        Returns:
            A formatted answer string.
        """
        question_lower = question.lower().strip()

        # Step 1: Detect the intent
        intent = self._detect_intent(question_lower)

        # Step 2: Extract date reference
        date_filter = self._extract_date(question_lower)

        # Step 3: Route to the appropriate handler
        if intent == "meeting":
            result = self._handle_meeting_query(date_filter)
        elif intent == "task":
            result = self._handle_task_query(date_filter)
        elif intent == "medication":
            result = self._handle_medication_query(date_filter)
        elif intent == "summary":
            # If the query has specific nouns beyond summary keywords,
            # try keyword search first (e.g. "tell me about pharmacy")
            search_result = self._handle_search_query(question_lower)
            if "no events found" not in search_result.lower():
                result = search_result
            else:
                result = self._handle_summary_query(date_filter)
        else:
            # Fallback: keyword search using the most meaningful words
            result = self._handle_search_query(question_lower)

        # Step 4: If LLM is enabled, always use it for a smarter answer
        if use_llm:
            try:
                from core.llm_engine import answer_query_llm, is_available
                if is_available():
                    context = self._format_memory_context()
                    llm_answer = answer_query_llm(question, context)
                    if llm_answer:
                        print(f"[QueryEngine] LLM answered: {llm_answer[:50]}...")
                        return llm_answer
            except Exception as e:
                print(f"[QueryEngine] LLM error: {e}")
                pass  # LLM failed — return rule-based result

        print(f"[QueryEngine] {intent} handling returned: {result[:100]}...")
        return result

    def _format_memory_context(self) -> str:
        """Format stored events as a text block for LLM context."""
        events = self.memory.get_all_events()
        if not events:
            return "No events stored in memory."

        lines = []
        for e in events[-20:]:  # Last 20 events to keep context manageable
            desc = e.get("description", "Unknown")
            etype = e.get("type", "event")
            date = e.get("parsed_date") or e.get("raw_date") or ""
            time = e.get("parsed_time") or e.get("time") or ""
            person = e.get("person") or ""
            line = f"- [{etype.upper()}] {desc}"
            if date:
                line += f" (Date: {date})"
            if time:
                line += f" (Time: {time})"
            if person:
                line += f" (Person: {person})"
            lines.append(line)

        return "\n".join(lines)

    # -- Intent detection --------------------------------------------------

    def _detect_intent(self, question: str) -> str:
        """
        Detect the user's intent from the question text.
        Returns: "meeting", "task", "medication", "summary", or "search"
        """
        scores = {}

        for intent, keywords in INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in question:
                    # Longer keyword matches get higher scores
                    score += len(keyword.split())
            scores[intent] = score

        # Pick the intent with the highest score
        best_intent = max(scores, key=scores.get)

        # If no intent scored above 0, fall back to search
        if scores[best_intent] == 0:
            return "search"

        return best_intent

    # -- Date extraction ---------------------------------------------------

    def _extract_date(self, question: str) -> str | None:
        """
        Extract a date reference from the question.
        Returns the normalized date keyword or None.
        """
        # Check multi-word date phrases first (e.g. "this week")
        for date_key, phrases in DATE_KEYWORDS.items():
            for phrase in phrases:
                if phrase in question:
                    return date_key

        # Check for day names (e.g. "monday", "next friday")
        for day in DAY_NAMES:
            if day in question:
                # Check if "next" precedes it
                if f"next {day}" in question:
                    return f"next {day}"
                return day

        return None

    # -- Query handlers ----------------------------------------------------

    def _handle_meeting_query(self, date_filter: str | None) -> str:
        """Handle meeting-related queries."""
        events = self._filter_events("meeting", date_filter)

        if not events:
            date_text = f" for {date_filter}" if date_filter else ""
            return f"No meetings found{date_text}."

        date_text = f" {date_filter}" if date_filter else ""
        descriptions = self._format_event_list(events)

        if len(events) == 1:
            return f"You have 1 meeting{date_text}: {descriptions}."
        else:
            return f"You have {len(events)} meetings{date_text}: {descriptions}."

    def _handle_task_query(self, date_filter: str | None) -> str:
        """Handle task-related queries."""
        events = self._filter_events("task", date_filter)

        if not events:
            date_text = f" for {date_filter}" if date_filter else ""
            return f"No tasks found{date_text}."

        date_text = f" {date_filter}" if date_filter else ""
        descriptions = self._format_event_list(events)

        if len(events) == 1:
            return f"You have 1 task{date_text}: {descriptions}."
        else:
            return f"You have {len(events)} tasks{date_text}: {descriptions}."

    def _handle_medication_query(self, date_filter: str | None) -> str:
        """Handle medication-related queries."""
        events = self._filter_events("medication", date_filter)

        if not events:
            date_text = f" for {date_filter}" if date_filter else ""
            return f"No medication reminders found{date_text}."

        descriptions = self._format_event_list(events)

        if len(events) == 1:
            return f"You have 1 medication reminder: {descriptions}."
        else:
            return f"You have {len(events)} medication reminders: {descriptions}."

    def _handle_summary_query(self, date_filter: str | None) -> str:
        """Handle summary/overview queries."""
        all_events = self.memory.get_all_events()

        if not all_events:
            return "No events stored in memory yet."

        # Count by type
        type_counts = {}
        for e in all_events:
            t = e.get("type", "other")
            type_counts[t] = type_counts.get(t, 0) + 1

        parts = []
        for t, count in type_counts.items():
            parts.append(f"{count} {t}(s)")

        summary = ", ".join(parts)
        total = len(all_events)

        # If date filter, show filtered events
        if date_filter:
            filtered = [
                e for e in all_events
                if date_filter in (e.get("raw_date") or e.get("date") or "").lower()
                or date_filter in (e.get("description") or "").lower()
            ]
            if filtered:
                descs = self._format_event_list(filtered)
                return f"For {date_filter}, you have {len(filtered)} event(s): {descs}."
            else:
                return f"No events found for {date_filter}. Overall you have {total} events: {summary}."

        return f"You have {total} events in memory: {summary}."

    def _handle_search_query(self, question: str) -> str:
        """Hybrid search: keyword first, then semantic fallback."""
        # Remove common question words to get meaningful search terms
        stop_words = {
            "what", "when", "where", "who", "how", "is", "are", "was",
            "were", "do", "did", "does", "the", "a", "an", "i", "my",
            "me", "have", "has", "had", "about", "for", "any", "tell",
            "show", "find", "can", "you", "please", "know",
        }
        words = re.findall(r'[a-z]+', question)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        if not keywords:
            return "I'm not sure what you're looking for. Try asking about meetings, tasks, or medications."

        # Step 1: Keyword search (exact SQL LIKE)
        all_results = []
        seen_descs = set()

        for keyword in keywords:
            results = self.memory.search_events(keyword)
            for r in results:
                desc = r.get("description", "")
                if desc.lower() not in seen_descs:
                    seen_descs.add(desc.lower())
                    all_results.append(r)

        if all_results:
            descriptions = self._format_event_list(all_results)
            return f"Found {len(all_results)} event(s) matching '{' '.join(keywords)}': {descriptions}."

        # Step 2: Semantic search (embedding-based, then TF-IDF fallback)
        semantic_results = self._semantic_search(question)
        if semantic_results:
            parts = []
            for sr in semantic_results:
                doc = sr["document"]
                desc = doc.get("description") or doc.get("summary") or doc.get("text", "")
                score_pct = int(sr["score"] * 100)
                parts.append(f"{desc} ({score_pct}% match)")

            return f"Found {len(semantic_results)} similar result(s): {'; '.join(parts)}."

        return f"No events found matching: {', '.join(keywords)}."

    # ── Semantic Search ────────────────────────────────────────

    def _get_emb_searcher(self):
        """Get or create the EmbeddingSearch engine (singleton).
        Disabled when config.ENABLE_EMBEDDINGS is False."""
        if self._emb_searcher is None:
            try:
                from config import ENABLE_EMBEDDINGS
            except ImportError:
                ENABLE_EMBEDDINGS = True

            if not ENABLE_EMBEDDINGS:
                # Return stub that reports unavailable
                class _Stub:
                    is_available = False
                self._emb_searcher = _Stub()
            else:
                from core.semantic_search import EmbeddingSearch
                self._emb_searcher = EmbeddingSearch()
        return self._emb_searcher

    def _get_tfidf_searcher(self):
        """Get or create the TF-IDF search engine, rebuilding index if needed."""
        from core.semantic_search import SemanticSearch

        # Rebuild index if new events have been added
        current_events = self.memory.get_all_events()
        current_count = len(current_events)

        if (self._searcher is None or current_count != self._index_version):
            self._searcher = SemanticSearch()

            # Build corpus from events + summaries
            documents = []
            for event in current_events:
                documents.append(dict(event))

            # Also index summaries if available
            if hasattr(self.memory, "search_summaries"):
                try:
                    summaries = self.memory.search_summaries()
                    for s in summaries:
                        documents.append(dict(s))
                except Exception:
                    pass  # Repository may not support this yet

            self._searcher.index(documents)
            self._index_version = current_count

        return self._searcher

    def _get_documents(self) -> list[dict]:
        """Build the document corpus from events + summaries."""
        documents = []
        current_events = self.memory.get_all_events()
        for event in current_events:
            documents.append(dict(event))

        if hasattr(self.memory, "search_summaries"):
            try:
                summaries = self.memory.search_summaries()
                for s in summaries:
                    documents.append(dict(s))
            except Exception:
                pass

        return documents

    def _semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Perform semantic search.

        Strategy:
          1. Try EmbeddingSearch (sentence-transformers) — handles synonyms
          2. Fall back to TF-IDF if embeddings unavailable

        Returns list of {document, score, rank} dicts, or empty list.
        """
        try:
            # Primary: embedding-based search
            emb = self._get_emb_searcher()
            if emb.is_available:
                documents = self._get_documents()
                if not documents:
                    return []

                # Load precomputed embeddings if available
                precomputed = None
                if hasattr(self.memory, "get_all_conversation_embeddings"):
                    try:
                        precomputed = self.memory.get_all_conversation_embeddings()
                    except Exception:
                        precomputed = None

                results = emb.search(
                    query, documents,
                    top_k=top_k, threshold=0.2,
                    precomputed_embeddings=precomputed,
                )
                if results:
                    # Apply importance + recency ranking (Phase Q)
                    from core.memory_ranker import rank_results
                    return rank_results(results)

            # Fallback: TF-IDF
            tfidf = self._get_tfidf_searcher()
            if tfidf.doc_count == 0:
                return []
            results = tfidf.search(query, top_k=top_k, threshold=0.1)
            if results:
                from core.memory_ranker import rank_results
                return rank_results(results)
            return results

        except Exception:
            return []  # Graceful fallback

    # -- Helpers -----------------------------------------------------------

    def _filter_events(self, event_type: str, date_filter: str | None) -> list[dict]:
        """
        Filter memory events by type and optionally by date.
        """
        all_events = self.memory.get_all_events()

        # Filter by type
        filtered = [e for e in all_events if e.get("type") == event_type]

        # Filter by date if specified
        if date_filter:
            date_filtered = []
            for e in filtered:
                event_date = (e.get("raw_date") or e.get("date") or "").lower()
                event_desc = (e.get("description") or "").lower()
                if date_filter in event_date or date_filter in event_desc:
                    date_filtered.append(e)
            filtered = date_filtered

        return filtered

    def _format_event_list(self, events: list[dict]) -> str:
        """Format a list of events into a readable comma-separated string."""
        parts = []
        for e in events:
            desc = e.get("description", "Unknown event")
            time = e.get("time")
            person = e.get("person")

            text = desc
            if time and time.lower() not in desc.lower():
                text += f" at {time}"
            if person and person.lower() not in desc.lower():
                text += f" with {person}"
            parts.append(text)

        if len(parts) <= 2:
            return " and ".join(parts)
        else:
            return ", ".join(parts[:-1]) + ", and " + parts[-1]


# -------------------------------------------------------------------------
# Quick test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple in-memory store for testing
    class _TestStore:
        def __init__(self, events):
            self._events = events
        def get_all_events(self):
            return self._events
        def search_events(self, keyword):
            kw = keyword.lower()
            return [e for e in self._events
                    if kw in (e.get('description','') + e.get('type','')).lower()]

    store = _TestStore([
        {"type": "meeting", "date": "tomorrow", "time": "10 AM",
         "person": "Dr. Smith", "description": "Doctor appointment"},
        {"type": "meeting", "date": None, "time": None,
         "person": "David", "description": "Visiting this weekend"},
        {"type": "task", "date": "tomorrow", "time": None,
         "person": None, "description": "Buy groceries tomorrow"},
        {"type": "task", "date": None, "time": None,
         "person": None, "description": "Call the pharmacy to refill prescription"},
        {"type": "medication", "date": None, "time": None,
         "person": None, "description": "Take your medicine after breakfast"},
    ])

    engine = QueryEngine(store)

    test_questions = [
        "What meetings do I have tomorrow?",
        "Do I have any tasks?",
        "Did I take medicine?",
        "What happened today?",
        "Tell me about the pharmacy",
        "What is scheduled for this weekend?",
    ]

    for q in test_questions:
        print(f"\n  Q: {q}")
        print(f"  A: {engine.query(q)}")
