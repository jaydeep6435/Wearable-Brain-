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

    private val PERSON_STOP_WORDS = setOf(
        "yes", "no", "also", "maybe", "okay", "ok", "alright", "right",
        "great", "perfect", "sure", "good", "then", "done"
    )

    private fun extractPerson(sentence: String): String? {
        val raw = PERSON_REGEX.find(sentence)?.value?.trim() ?: return null
        val normalized = raw.lowercase()
        if (normalized in PERSON_STOP_WORDS) return null
        if (normalized.matches(Regex("^(speaker)\\s+\\d+$"))) return null
        return raw
    }

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
                val person = extractPerson(sentence)

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

    // ── Speaker Separation (rule-based) ─────────────────────

    data class SpeakerSegment(
        val speaker: String,
        val text: String
    )

    /**
     * Simple rule-based speaker separation.
     *
     * Detects perspective shifts (I/my vs you/your),
     * question patterns, and instruction patterns to
     * label sentences as different speakers.
     *
     * Returns structured speaker-labeled segments.
     */
    fun separateSpeakers(text: String): List<SpeakerSegment> {
        val sentences = text.split(Regex("[.!?]+"))
            .map { it.trim() }
            .filter { it.length > 3 }

        if (sentences.isEmpty()) return listOf(SpeakerSegment("SPEAKER_1", text))
        if (sentences.size == 1) return listOf(SpeakerSegment("SPEAKER_1", text))

        val segments = mutableListOf<SpeakerSegment>()
        var currentSpeaker = "SPEAKER_1"

        // Patterns that suggest a different speaker is talking
        val youPatterns = listOf("you should", "you need", "you have to", "your son",
            "your daughter", "your medicine", "your appointment", "don't forget",
            "remember to", "make sure")
        val instructionPatterns = listOf("take your", "call the", "we need to")

        for (sentence in sentences) {
            val lower = sentence.lowercase().trim()

            // Perspective-based speaker switch
            val isYouPerspective = youPatterns.any { lower.contains(it) } ||
                    instructionPatterns.any { lower.contains(it) }
            val isIPerspective = lower.startsWith("i ") || lower.startsWith("i'm ") ||
                    lower.startsWith("my ") || lower.contains(" i have ") ||
                    lower.contains(" i need ")
            val isQuestion = lower.endsWith("?") || lower.startsWith("when") ||
                    lower.startsWith("what") || lower.startsWith("where") ||
                    lower.startsWith("how") || lower.startsWith("did you")

            val detectedSpeaker = when {
                isIPerspective -> "SPEAKER_1"
                isYouPerspective || isQuestion -> "SPEAKER_2"
                else -> currentSpeaker  // keep same speaker if no signal
            }

            if (detectedSpeaker != currentSpeaker) {
                currentSpeaker = detectedSpeaker
            }

            segments.add(SpeakerSegment(currentSpeaker, sentence.trim()))
        }

        // Merge consecutive segments from same speaker
        val merged = mutableListOf<SpeakerSegment>()
        for (seg in segments) {
            if (merged.isNotEmpty() && merged.last().speaker == seg.speaker) {
                val last = merged.removeAt(merged.size - 1)
                merged.add(SpeakerSegment(last.speaker, "${last.text}. ${seg.text}"))
            } else {
                merged.add(seg)
            }
        }

        val speakerCount = merged.map { it.speaker }.distinct().size
        Log.i(TAG, "Speaker separation: ${sentences.size} sentences → ${merged.size} segments, $speakerCount speakers")

        return merged
    }
}
