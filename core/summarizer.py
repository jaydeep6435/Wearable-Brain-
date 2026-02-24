"""
Summarizer Module — Text → Summary
====================================
Uses a simple *extractive* approach (no ML model needed):
  1. Split text into sentences
  2. Score each sentence by word-frequency importance
  3. Return the top-N sentences in their original order

This keeps everything offline and lightweight.
"""

import re
from collections import Counter


# ── Common stop-words to ignore when scoring ────────────────────────────────
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
    "also", "well", "know", "like", "going", "right", "okay", "yeah",
}

# ── Bonus keywords relevant to Alzheimer's memory assistance ────────────────
BONUS_KEYWORDS = {
    "remember", "forgot", "forget", "medicine", "medication", "doctor",
    "appointment", "meeting", "tomorrow", "today", "important", "call",
    "visit", "therapy", "exercise", "eat", "lunch", "dinner", "breakfast",
    "remind", "reminder", "task", "schedule", "family", "son", "daughter",
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    # Split on . ! ? followed by a space or end-of-string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Remove empty strings and very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _score_sentence(sentence: str, word_freq: Counter) -> float:
    """
    Score a sentence based on:
      - Sum of word-frequency scores (content words only)
      - Bonus for memory-assistance keywords
    """
    words = re.findall(r'[a-z]+', sentence.lower())
    if not words:
        return 0.0

    score = 0.0
    for word in words:
        if word not in STOP_WORDS:
            score += word_freq.get(word, 0)
        if word in BONUS_KEYWORDS:
            score += 2  # Extra weight for care-related terms

    # Normalise by sentence length to avoid bias toward long sentences
    return score / len(words)


def summarize(text: str, num_sentences: int = 3, use_llm: bool = False) -> str:
    """
    Generate an extractive summary of the given text.

    Args:
        text           : The full text to summarize.
        num_sentences  : How many sentences to include in the summary.
        use_llm        : If True, append an LLM-generated summary.

    Returns:
        A string containing the top-N most important sentences,
        presented in their original order. If use_llm is True,
        an LLM summary is appended.
    """
    # Step 1: Split into sentences
    sentences = _split_sentences(text)
    if not sentences:
        return text  # Nothing to summarize

    # Step 2: Build word-frequency table (skip stop-words)
    all_words = re.findall(r'[a-z]+', text.lower())
    word_freq = Counter(w for w in all_words if w not in STOP_WORDS)

    # Step 3: Score every sentence
    scored = [(i, sent, _score_sentence(sent, word_freq))
              for i, sent in enumerate(sentences)]

    # Step 4: Pick the top N by score
    top_n = sorted(scored, key=lambda x: x[2], reverse=True)[:num_sentences]

    # Step 5: Re-order by original position so the summary reads naturally
    top_n.sort(key=lambda x: x[0])

    extractive = " ".join(sent for _, sent, _ in top_n)

    # Step 6 (optional): Use LLM for a better summary
    if use_llm:
        try:
            from core.llm_engine import summarize_llm, is_available
            if is_available():
                llm_summary = summarize_llm(text)
                if llm_summary:
                    # Return LLM summary as primary, extractive as reference
                    return llm_summary
        except Exception:
            pass  # LLM failed — return extractive only

    return extractive


# -- Importance tagging keywords (for highlighting) -------------------------
HIGHLIGHT_TAGS = {
    "date": {"today", "tonight", "tomorrow", "yesterday", "monday", "tuesday",
             "wednesday", "thursday", "friday", "saturday", "sunday",
             "january", "february", "march", "april", "may", "june",
             "july", "august", "september", "october", "november", "december",
             "next", "this", "coming", "week", "month"},
    "task": {"call", "buy", "pick", "bring", "send", "submit", "finish",
             "complete", "prepare", "remind", "remember", "forget", "forgot",
             "need", "must", "should"},
    "meeting": {"doctor", "dentist", "therapist", "appointment", "meeting",
                "visit", "visiting", "clinic", "hospital"},
    "medication": {"medicine", "medication", "pill", "pills", "tablet",
                   "prescription", "refill", "dose"},
}


def summarize_with_highlights(text: str, num_sentences: int = 5) -> list[dict]:
    """
    Generate a highlighted summary showing which sentences are important.

    Args:
        text           : The full text to summarize.
        num_sentences  : How many top sentences to mark as important.

    Returns:
        List of dicts, one per sentence:
            "sentence"  : the sentence text
            "important" : True if this is a top-N sentence
            "tags"      : list of categories that triggered importance
                          e.g. ["date", "meeting", "medication"]
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Build word-frequency table
    all_words = re.findall(r'[a-z]+', text.lower())
    word_freq = Counter(w for w in all_words if w not in STOP_WORDS)

    # Score and rank
    scored = [(i, sent, _score_sentence(sent, word_freq))
              for i, sent in enumerate(sentences)]
    top_indices = {
        item[0]
        for item in sorted(scored, key=lambda x: x[2], reverse=True)[:num_sentences]
    }

    # Build highlighted results
    results = []
    for i, sentence in enumerate(sentences):
        words = set(re.findall(r'[a-z]+', sentence.lower()))
        tags = []
        for tag_name, tag_keywords in HIGHLIGHT_TAGS.items():
            if words & tag_keywords:  # set intersection
                tags.append(tag_name)

        results.append({
            "sentence": sentence,
            "important": i in top_indices,
            "tags": tags,
        })

    return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    sample = (
        "Good morning! I have a doctor appointment tomorrow at 10 AM. "
        "Don't forget to take your medicine after breakfast. "
        "We need to call the pharmacy to refill the prescription. "
        "Your son is visiting this weekend. "
        "Remember to do your morning exercises before lunch. "
        "The weather today is sunny, perfect for a walk in the garden."
    )

    print("--- Summary ---")
    print(summarize(sample, num_sentences=3))

    print("\n--- Highlighted Summary ---")
    for item in summarize_with_highlights(sample, num_sentences=3):
        marker = "[IMPORTANT]" if item["important"] else "           "
        tags = ", ".join(item["tags"]) if item["tags"] else ""
        print(f"  {marker} {item['sentence']}")
        if tags:
            print(f"              Tags: {tags}")
