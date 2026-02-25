package com.example.memory_assistant

import android.util.Log

/**
 * SimpleNlpProcessor — Lightweight on-device text analysis.
 *
 * Extracts events (meetings, medications, tasks) from conversation text
 * using simple keyword + regex matching. No ML models needed.
 *
 * This is the Android-side equivalent of nlp/event_extractor.py.
 * It does NOT replace the Python AI logic — it provides a basic
 * bridge-level extraction so data flows correctly on the device.
 */
object SimpleNlpProcessor {

    private const val TAG = "WBrain.NLP"

    // ── Keyword patterns for event type detection ────────────

    private val MEETING_WORDS = listOf(
        "appointment", "meeting", "doctor", "visit", "visiting",
        "hospital", "clinic", "scheduled", "check-up", "checkup"
    )

    private val MEDICATION_WORDS = listOf(
        "medicine", "medication", "pill", "tablet", "prescription",
        "drug", "pharmacy", "refill", "dose", "take your"
    )

    private val TASK_WORDS = listOf(
        "call", "buy", "pick up", "need to", "have to", "should",
        "must", "don't forget", "remember to", "make sure"
    )

    // ── Regex patterns ───────────────────────────────────────

    private val TIME_REGEX = Regex(
        """(\d{1,2})\s*(:\d{2})?\s*(am|pm|AM|PM|a\.m\.|p\.m\.)""",
        RegexOption.IGNORE_CASE
    )

    private val DATE_REGEX = Regex(
        """(today|tomorrow|yesterday|this weekend|next week|""" +
        """monday|tuesday|wednesday|thursday|friday|saturday|sunday)""",
        RegexOption.IGNORE_CASE
    )

    private val PERSON_REGEX = Regex(
        """\b(Dr\.?\s+[A-Z][a-z]+|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)\b"""
    )

    // ── Extract events from text ─────────────────────────────

    data class ExtractedEvent(
        val type: String,
        val description: String,
        val date: String? = null,
        val time: String? = null,
        val person: String? = null
    )

    fun extractEvents(text: String): List<ExtractedEvent> {
        val events = mutableListOf<ExtractedEvent>()
        val sentences = text.split(Regex("[.!?]+"))
            .map { it.trim() }
            .filter { it.length > 5 }

        Log.d(TAG, "Extracting events from ${sentences.size} sentences...")

        for (sentence in sentences) {
            val lower = sentence.lowercase()

            val type = when {
                MEETING_WORDS.any { it in lower } -> "meeting"
                MEDICATION_WORDS.any { it in lower } -> "medication"
                TASK_WORDS.any { it in lower } -> "task"
                else -> null
            }

            if (type != null) {
                val time = TIME_REGEX.find(sentence)?.value
                val date = DATE_REGEX.find(sentence)?.value
                val person = PERSON_REGEX.find(sentence)?.value

                events.add(ExtractedEvent(
                    type = type,
                    description = sentence.trim(),
                    date = date,
                    time = time,
                    person = person
                ))

                Log.d(TAG, "  Found: [$type] ${sentence.take(50)} " +
                    "(date=$date, time=$time, person=$person)")
            }
        }

        Log.i(TAG, "Extracted ${events.size} events from text")
        return events
    }

    // ── Summarize text ───────────────────────────────────────

    fun summarize(text: String): String {
        val sentences = text.split(Regex("[.!?]+"))
            .map { it.trim() }
            .filter { it.length > 5 }

        return when {
            sentences.isEmpty() -> text.take(200)
            sentences.size <= 3 -> sentences.joinToString(". ") + "."
            else -> sentences.take(3).joinToString(". ") + "."
        }
    }

    // ── Extract key points ───────────────────────────────────

    fun extractKeyPoints(text: String): List<String> {
        val points = mutableListOf<String>()
        val sentences = text.split(Regex("[.!?]+"))
            .map { it.trim() }
            .filter { it.length > 5 }

        for (sentence in sentences) {
            val lower = sentence.lowercase()
            // A sentence is a "key point" if it contains actionable/important words
            val isImportant = MEETING_WORDS.any { it in lower } ||
                    MEDICATION_WORDS.any { it in lower } ||
                    TASK_WORDS.any { it in lower } ||
                    DATE_REGEX.containsMatchIn(sentence) ||
                    TIME_REGEX.containsMatchIn(sentence)

            if (isImportant) {
                points.add(sentence.trim())
            }
        }

        return points.take(5)
    }
}
