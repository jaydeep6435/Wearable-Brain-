package com.example.memory_assistant

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log
import java.util.UUID

/**
 * MemoryDatabase — Android-native SQLite storage for the Memory Assistant.
 *
 * Schema mirrors the Python storage/db.py schema so data is compatible.
 * This provides REAL data persistence for the MethodChannel bridge,
 * replacing the stub responses that never saved anything.
 *
 * Tables: conversations, events, summaries
 */
class MemoryDatabase(private val appContext: Context) : SQLiteOpenHelper(
    appContext, "memory.db", null, DB_VERSION
) {
    companion object {
        const val TAG = "WBrain.DB"
        const val DB_VERSION = 1

        fun newId(): String = UUID.randomUUID().toString()
    }

    override fun onCreate(db: SQLiteDatabase) {
        Log.i(TAG, "Creating database tables...")

        db.execSQL("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                raw_text    TEXT,
                audio_path  TEXT,
                source      TEXT DEFAULT 'text'
            )
        """)

        db.execSQL("""
            CREATE TABLE IF NOT EXISTS events (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT,
                type            TEXT NOT NULL,
                description     TEXT,
                raw_date        TEXT,
                raw_time        TEXT,
                person          TEXT,
                importance      INTEGER DEFAULT 0,
                fingerprint     TEXT UNIQUE,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        db.execSQL("""
            CREATE TABLE IF NOT EXISTS summaries (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                summary         TEXT,
                key_points      TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        db.execSQL("CREATE INDEX IF NOT EXISTS idx_events_conv ON events(conversation_id)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_events_fp   ON events(fingerprint)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_summaries_conv ON summaries(conversation_id)")

        Log.i(TAG, "Database tables created ✓")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Future migrations go here
    }

    // ── Save ─────────────────────────────────────────────────

    fun saveConversation(text: String, source: String = "text"): String {
        val id = newId()
        val timestamp = java.text.SimpleDateFormat(
            "yyyy-MM-dd'T'HH:mm:ss", java.util.Locale.US
        ).format(java.util.Date())

        val values = ContentValues().apply {
            put("id", id)
            put("timestamp", timestamp)
            put("raw_text", text)
            put("source", source)
        }
        writableDatabase.insertWithOnConflict(
            "conversations", null, values, SQLiteDatabase.CONFLICT_REPLACE
        )
        Log.i(TAG, "✓ Saved conversation ${id.take(8)}... (${text.length} chars, source=$source)")
        return id
    }

    fun saveEvent(
        convId: String, type: String, description: String,
        date: String? = null, time: String? = null, person: String? = null
    ): String? {
        val effectiveDate = date?.trim().takeUnless { it.isNullOrBlank() }
            ?: extractDateHint(description)
        val effectiveTime = time?.trim().takeUnless { it.isNullOrBlank() }
            ?: extractTimeHint(description)
        val effectivePerson = person?.trim().takeUnless { it.isNullOrBlank() }
            ?: extractPersonHint(description)

        // Deduplicate by fingerprint
        val fp = "$type|${description.lowercase().trim()}|${effectiveDate ?: ""}|${effectiveTime ?: ""}|${effectivePerson ?: ""}"
            .hashCode().toString(16)

        val existing = readableDatabase.rawQuery(
            "SELECT id FROM events WHERE fingerprint = ?", arrayOf(fp)
        )
        if (existing.moveToFirst()) {
            existing.close()
            Log.d(TAG, "  Duplicate skipped: ${description.take(50)}")
            return null
        }
        existing.close()

        val id = newId()
        val values = ContentValues().apply {
            put("id", id)
            put("conversation_id", convId)
            put("type", type)
            put("description", description)
            put("raw_date", effectiveDate)
            put("raw_time", effectiveTime)
            put("person", effectivePerson)
            put("fingerprint", fp)
        }
        writableDatabase.insertWithOnConflict(
            "events", null, values, SQLiteDatabase.CONFLICT_REPLACE
        )
        Log.i(TAG, "  ✓ Saved event: [$type] ${description.take(60)}")
        return id
    }

    fun saveSummary(convId: String, summary: String, keyPoints: String = "") {
        val id = newId()
        val values = ContentValues().apply {
            put("id", id)
            put("conversation_id", convId)
            put("summary", summary)
            put("key_points", keyPoints)
        }
        writableDatabase.insertWithOnConflict(
            "summaries", null, values, SQLiteDatabase.CONFLICT_REPLACE
        )
        Log.i(TAG, "  ✓ Saved summary for conv ${convId.take(8)}")
    }

    // ── Legacy Query (used by queryMemory MethodChannel) ─────

    fun queryMemory(question: String): Map<String, Any> {
        Log.i(TAG, "⤷ queryMemory: '$question'")

        if (question.isBlank()) {
            return mapOf(
                "answer" to "Please ask a specific question about your memory.",
                "results" to emptyList<Map<String, Any>>(),
                "method" to "intent"
            )
        }

        // Use the intent-aware engine so answers stay concise and question-focused.
        val chat = chatWithMemory(question)
        val conciseAnswer = (chat["answer"] as? String)?.trim().orEmpty()

        // Provide structured evidence list for UI/debug without dumping full transcript text.
        val related = collectRelatedMemory(question, limit = 6)

        return mapOf(
            "answer" to conciseAnswer,
            "results" to related,
            "method" to "intent",
            "confidence" to (chat["confidence"] ?: "medium"),
            "mode" to (chat["mode"] ?: "local")
        )
    }

    private fun collectRelatedMemory(question: String, limit: Int = 6): List<Map<String, Any?>> {
        val stopWords = setOf(
            "what", "when", "where", "who", "how", "is", "are", "was", "were",
            "do", "did", "does", "the", "a", "an", "i", "my", "me", "have",
            "has", "had", "about", "for", "any", "tell", "show", "find",
            "can", "you", "please", "know", "memory", "remember"
        )
        val keywords = question.lowercase()
            .split(Regex("[\\s,;.!?]+"))
            .filter { it.length > 2 && it !in stopWords }

        if (keywords.isEmpty()) return emptyList()

        val rows = mutableListOf<Map<String, Any?>>()
        val seen = mutableSetOf<String>()

        for (kw in keywords) {
            // Event hits
            val ev = readableDatabase.rawQuery(
                """SELECT type, description, raw_date, raw_time, person
                   FROM events
                   WHERE description LIKE ? OR type LIKE ? OR person LIKE ?
                   ORDER BY rowid DESC LIMIT 10""",
                arrayOf("%$kw%", "%$kw%", "%$kw%")
            )
            while (ev.moveToNext()) {
                val desc = ev.getString(1) ?: ""
                val key = "e:${cleanDesc(desc).lowercase()}"
                if (desc.isNotBlank() && key !in seen) {
                    seen.add(key)
                    rows.add(
                        mapOf(
                            "source" to "event",
                            "type" to (ev.getString(0) ?: ""),
                            "description" to desc,
                            "raw_date" to ev.getString(2),
                            "raw_time" to ev.getString(3),
                            "person" to ev.getString(4)
                        )
                    )
                }
            }
            ev.close()

            // Summary hits
            val sm = readableDatabase.rawQuery(
                """SELECT summary
                   FROM summaries
                   WHERE summary LIKE ?
                   ORDER BY created_at DESC LIMIT 5""",
                arrayOf("%$kw%")
            )
            while (sm.moveToNext()) {
                val summary = sm.getString(0) ?: ""
                val key = "s:${summary.take(80).lowercase()}"
                if (summary.isNotBlank() && key !in seen) {
                    seen.add(key)
                    rows.add(
                        mapOf(
                            "source" to "summary",
                            "description" to cleanDesc(summary)
                        )
                    )
                }
            }
            sm.close()

            // Conversation raw text hits (snippet only)
            val cv = readableDatabase.rawQuery(
                """SELECT raw_text, timestamp
                   FROM conversations
                   WHERE raw_text LIKE ?
                   ORDER BY rowid DESC LIMIT 5""",
                arrayOf("%$kw%")
            )
            while (cv.moveToNext()) {
                val raw = cv.getString(0) ?: ""
                val key = "c:${raw.take(80).lowercase()}"
                if (raw.isNotBlank() && key !in seen) {
                    seen.add(key)
                    val firstSentence = raw
                        .split(Regex("[.!?\\n]+"))
                        .map { it.trim() }
                        .firstOrNull { it.isNotBlank() }
                        ?.take(160)
                        ?: raw.take(160)
                    rows.add(
                        mapOf(
                            "source" to "conversation",
                            "description" to firstSentence,
                            "timestamp" to cv.getString(1)
                        )
                    )
                }
            }
            cv.close()

            if (rows.size >= limit) break
        }

        return rows.take(limit)
    }

    // ── Stats ────────────────────────────────────────────────

    fun getStats(): Map<String, Any> {
        val convCount = countTable("conversations")
        val eventCount = countTable("events")
        val summaryCount = countTable("summaries")
        Log.i(TAG, "Stats: $convCount convs, $eventCount events, $summaryCount summaries")
        return mapOf(
            "total_conversations" to convCount,
            "total_events" to eventCount,
            "total_summaries" to summaryCount,
            "config" to mapOf("simplified_mode" to false, "low_resource_mode" to false)
        )
    }

    fun getMemoryCount(): Int {
        val count = countTable("conversations") + countTable("events")
        Log.i(TAG, "Memory count: $count")
        return count
    }

    fun getAllEvents(typeFilter: String? = null): List<Map<String, Any?>> {
        val results = mutableListOf<Map<String, Any?>>()
        val query = if (typeFilter != null) "SELECT * FROM events WHERE type = ? ORDER BY rowid DESC"
                    else "SELECT * FROM events ORDER BY rowid DESC"
        val args = if (typeFilter != null) arrayOf(typeFilter) else null

        val cursor = readableDatabase.rawQuery(query, args)
        while (cursor.moveToNext()) {
            results.add(mapOf(
                "id" to cursor.getString(cursor.getColumnIndexOrThrow("id")),
                "type" to cursor.getString(cursor.getColumnIndexOrThrow("type")),
                "description" to cursor.getString(cursor.getColumnIndexOrThrow("description")),
                "raw_date" to cursor.getString(cursor.getColumnIndexOrThrow("raw_date")),
                "raw_time" to cursor.getString(cursor.getColumnIndexOrThrow("raw_time")),
                "person" to cursor.getString(cursor.getColumnIndexOrThrow("person"))
            ))
        }
        cursor.close()
        return results
    }

    private fun countTable(table: String): Int {
        val cursor = readableDatabase.rawQuery("SELECT COUNT(*) FROM $table", null)
        cursor.moveToFirst()
        val count = cursor.getInt(0)
        cursor.close()
        return count
    }

    fun getUrgentItems(hours: Int = 48): List<Map<String, Any?>> {
        val results = mutableListOf<Map<String, Any?>>()
        val cursor = readableDatabase.rawQuery(
            """SELECT * FROM events
               WHERE type IN ('medication', 'meeting', 'appointment')
               AND importance >= 3
               ORDER BY importance DESC, rowid DESC
               LIMIT 10""", null
        )
        while (cursor.moveToNext()) {
            results.add(mapOf(
                "id" to cursor.getString(cursor.getColumnIndexOrThrow("id")),
                "type" to cursor.getString(cursor.getColumnIndexOrThrow("type")),
                "description" to cursor.getString(cursor.getColumnIndexOrThrow("description")),
                "raw_date" to cursor.getString(cursor.getColumnIndexOrThrow("raw_date")),
                "raw_time" to cursor.getString(cursor.getColumnIndexOrThrow("raw_time")),
                "person" to cursor.getString(cursor.getColumnIndexOrThrow("person")),
                "importance" to cursor.getInt(cursor.getColumnIndexOrThrow("importance"))
            ))
        }
        cursor.close()
        Log.i(TAG, "getUrgentItems($hours h) → ${results.size} results")
        return results
    }

    // ═══════════════════════════════════════════════════════════
    //  INTENT-AWARE CHAT WITH DETERMINISTIC REASONING
    // ═══════════════════════════════════════════════════════════

    data class StructuredResult(
        val answer: String,
        val items: List<Map<String, String>>,
        val confidence: String = "medium"
    )

    fun chatWithMemory(question: String): Map<String, Any> {
        Log.i(TAG, "━━━ chatWithMemory (Reasoning) ━━━━━━━━")
        Log.i(TAG, "  Q: '$question'")

        if (countTable("events") == 0 && countTable("conversations") == 0) {
            return mapOf(
                "answer" to "I don't have any memories stored yet. Try recording a conversation first using the Record tab!",
                "related_events" to emptyList<Any>(),
                "confidence" to "high", "mode" to "empty-db"
            )
        }

        // 1. Detect intent
        val intent = detectIntent(question)
        Log.i(TAG, "  Intent: $intent")

        // 2. Execute deterministic logic
        val result = executeIntent(intent, question)
        Log.i(TAG, "  Items: ${result.items.size}, Answer: ${result.answer.take(50)}...")

        // 3. Try LLM reformat (optional)
        try {
            val llm = formatWithLLM(question, result)
            if (llm != null) {
                Log.i(TAG, "  LLM formatted ✓")
                Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                return llm + mapOf("intent" to intent)
            }
        } catch (e: Exception) {
            Log.w(TAG, "  LLM skip: ${e.message}")
        }

        // 4. Use deterministic answer directly (always clean)
        Log.i(TAG, "  Local answer ✓")
        Log.i(TAG, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return mapOf(
            "answer" to result.answer,
            "related_events" to emptyList<Any>(),
            "confidence" to result.confidence,
            "mode" to "local", "intent" to intent
        )
    }

    // ── Intent Detection ────────────────────────────────────

    private fun detectIntent(q: String): String {
        val lower = q.lowercase().trim()

        // Count queries
        if (lower.contains("how many") || lower.contains("count") || lower.contains("number of")) {
            if (lower.contains("appointment") || lower.contains("meeting"))
                return "COUNT_APPOINTMENTS"
            return "COUNT_EVENTS"
        }

        // Events on a specific day
        val days = listOf("monday","tuesday","wednesday","thursday","friday","saturday","sunday","tomorrow","today","yesterday")
        val matched = days.firstOrNull { lower.contains(it) }
        if (matched != null && !lower.contains("summary") && !lower.contains("happened")) {
            return "EVENTS_ON_DATE"
        }

        // Summary
        val sumPat = listOf("summary","summarize","what happened","happened today","daily brief","recap","my day","overview","catch me up","update me","brief me")
        if (sumPat.any { lower.contains(it) }) return "DAILY_SUMMARY"

        // Appointment
        val apPat = listOf("appointment","meeting","doctor","visit","schedule","when is my","clinic","hospital","checkup")
        if (apPat.any { lower.contains(it) }) return "APPOINTMENT"

        // Specific time question (prefer direct answer over long lists)
        val timePat = listOf("what time", "when does", "when is it", "start time", "at what time")
        if (timePat.any { lower.contains(it) }) return "TIME_QUERY"

        // Medication
        val medPat = listOf("medicine","medication","pill","tablet","prescription","pharmacy","refill")
        if (medPat.any { lower.contains(it) }) return "MEDICATION"

        // Reminder
        val remPat = listOf("remind","reminder","task","todo","upcoming","don't forget")
        if (remPat.any { lower.contains(it) }) return "REMINDER"

        return "GENERAL"
    }

    // ── Execute Intent (deterministic logic) ────────────────

    private fun executeIntent(intent: String, question: String): StructuredResult = when (intent) {
        "COUNT_APPOINTMENTS" -> intentCountAppointments(question)
        "COUNT_EVENTS"       -> intentCountEvents()
        "EVENTS_ON_DATE"     -> intentEventsOnDate(question)
        "DAILY_SUMMARY"      -> intentDailySummary()
        "TIME_QUERY"         -> intentTimeQuery(question)
        "APPOINTMENT"        -> intentListAppointments()
        "MEDICATION"         -> intentListMedications()
        "REMINDER"           -> intentListReminders()
        else                 -> intentGeneralQuery(question)
    }

    // ── TIME_QUERY ────────────────────────────────────────

    private fun intentTimeQuery(question: String): StructuredResult {
        val q = question.lowercase()
        val stopWords = setOf(
            "what", "when", "time", "does", "is", "it", "start", "at", "the", "a", "an",
            "my", "me", "to", "for", "of", "on", "in", "about", "please"
        )
        val kws = q.split(Regex("[\\s,;.!?]+"))
            .filter { it.length > 2 && it !in stopWords }

        val items = mutableListOf<Map<String, String>>()

        fun collectRows(cursor: android.database.Cursor) {
            while (cursor.moveToNext()) {
                val desc = cursor.getString(0) ?: ""
                val date = cursor.getString(1) ?: ""
                val time = cursor.getString(2) ?: ""
                val person = cursor.getString(3) ?: ""
                if (time.isNotBlank()) {
                    items.add(
                        mapOf(
                            "description" to desc,
                            "date" to date,
                            "time" to time,
                            "person" to person,
                        )
                    )
                }
            }
        }

        for (kw in kws) {
            val cur = readableDatabase.rawQuery(
                """SELECT description, raw_date, raw_time, person
                   FROM events
                   WHERE raw_time IS NOT NULL AND trim(raw_time) != ''
                   AND (description LIKE ? OR type LIKE ? OR person LIKE ?)
                   ORDER BY rowid DESC LIMIT 5""",
                arrayOf("%$kw%", "%$kw%", "%$kw%")
            )
            collectRows(cur)
            cur.close()
            if (items.isNotEmpty()) break
        }

        if (items.isEmpty()) {
            val cur = readableDatabase.rawQuery(
                """SELECT description, raw_date, raw_time, person
                   FROM events
                   WHERE raw_time IS NOT NULL AND trim(raw_time) != ''
                   AND type IN ('meeting','appointment','task','note')
                   ORDER BY rowid DESC LIMIT 1""",
                null
            )
            collectRows(cur)
            cur.close()
        }

        val top = items.firstOrNull()
        val answer = if (top == null) {
            "I couldn't find a specific time in your saved memories yet."
        } else {
            val time = top["time"].orEmpty()
            val desc = cleanDesc(top["description"].orEmpty())
            if (desc.isNotBlank()) "It is at $time for: $desc." else "It is at $time."
        }

        return StructuredResult(answer, items, if (top != null) "high" else "low")
    }

    // ── COUNT_APPOINTMENTS ──────────────────────────────────

    private fun intentCountAppointments(q: String): StructuredResult {
        val items = fetchEvents("meeting")
        val count = items.size
        val answer = if (count == 0) {
            "You don't have any appointments scheduled right now."
        } else {
            val list = formatItemList(items.distinctBy { cleanDesc(it["description"]?:"").take(40) })
            "You have $count appointment${if (count != 1) "s" else ""}:\n\n$list"
        }
        return StructuredResult(answer, items, if (count > 0) "high" else "low")
    }

    private fun intentCountEvents(): StructuredResult {
        val e = countTable("events"); val c = countTable("conversations")
        return StructuredResult("You have $e events from $c conversations stored.", emptyList(), "high")
    }

    // ── EVENTS_ON_DATE ──────────────────────────────────────

    private fun intentEventsOnDate(question: String): StructuredResult {
        val q = question.lowercase()
        val days = listOf("monday","tuesday","wednesday","thursday","friday","saturday","sunday","tomorrow","today","yesterday")
        val target = days.firstOrNull { q.contains(it) } ?: return StructuredResult(
            "I'm not sure which day you mean. Could you say Monday, Sunday, or tomorrow?", emptyList(), "low"
        )

        val items = mutableListOf<Map<String, String>>()
        val cursor = readableDatabase.rawQuery(
            """SELECT description, raw_date, raw_time, person, type FROM events
               WHERE lower(raw_date) LIKE ? OR lower(description) LIKE ?
               ORDER BY rowid DESC LIMIT 10""",
            arrayOf("%$target%", "%$target%")
        )
        while (cursor.moveToNext()) {
            items.add(mapOf(
                "description" to (cursor.getString(0) ?: ""),
                "date" to (cursor.getString(1) ?: ""),
                "time" to (cursor.getString(2) ?: ""),
                "person" to (cursor.getString(3) ?: "")
            ))
        }
        cursor.close()

        val dayLabel = target.replaceFirstChar { it.uppercase() }
        val unique = items.distinctBy { cleanDesc(it["description"] ?: "").take(40) }

        val answer = if (unique.isEmpty()) {
            "I don't have anything scheduled for $dayLabel."
        } else {
            "Here's what I have for $dayLabel:\n\n${formatItemList(unique)}"
        }
        return StructuredResult(answer, items, if (items.isNotEmpty()) "high" else "low")
    }

    // ── DAILY_SUMMARY ───────────────────────────────────────

    private fun intentDailySummary(): StructuredResult {
        val appointments = mutableListOf<String>()
        val medications = mutableListOf<String>()
        val tasks = mutableListOf<String>()
        val people = mutableSetOf<String>()
        val items = mutableListOf<Map<String, String>>()

        val cur = readableDatabase.rawQuery(
            "SELECT type, description, raw_date, raw_time, person FROM events ORDER BY rowid DESC LIMIT 15", null
        )
        while (cur.moveToNext()) {
            val type = cur.getString(0) ?: ""
            val desc = cleanDesc(cur.getString(1) ?: "")
            val date = cur.getString(2) ?: ""
            val time = cur.getString(3) ?: ""
            val person = cur.getString(4) ?: ""
            if (person.isNotEmpty()) people.add(person)
            val ts = (if (time.isNotEmpty()) " at $time" else "") + (if (date.isNotEmpty()) " on $date" else "")
            when (type) {
                "meeting" -> appointments.add("$desc$ts")
                "medication" -> medications.add("$desc$ts")
                else -> tasks.add(desc)
            }
            items.add(mapOf("type" to type, "description" to desc))
        }
        cur.close()

        if (appointments.isEmpty() && medications.isEmpty() && tasks.isEmpty()) {
            return StructuredResult("I don't have enough details for a summary yet. Try recording a conversation first!", emptyList(), "low")
        }

        val sb = StringBuilder("Here's your summary:\n")
        if (appointments.isNotEmpty()) {
            sb.append("\n📋 Appointments:\n")
            appointments.distinctBy { it.take(40) }.take(5).forEach { sb.append("  • $it\n") }
        }
        if (medications.isNotEmpty()) {
            sb.append("\n💊 Medications:\n")
            medications.distinctBy { it.take(40) }.take(5).forEach { sb.append("  • $it\n") }
        }
        if (tasks.isNotEmpty()) {
            sb.append("\n📝 Other:\n")
            tasks.distinctBy { it.take(40) }.take(5).forEach { sb.append("  • $it\n") }
        }
        if (people.isNotEmpty()) sb.append("\n👥 People: ${people.joinToString(", ")}\n")
        sb.append("\nTake care! 💙")

        return StructuredResult(sb.toString(), items, "high")
    }

    // ── APPOINTMENT / MEDICATION / REMINDER ──────────────────

    private fun intentListAppointments(): StructuredResult {
        val items = fetchEvents("meeting")
        val unique = items.distinctBy { cleanDesc(it["description"]?:"").take(40) }
        val answer = if (unique.isEmpty()) "I don't have any appointments stored. Record a conversation mentioning your appointments and I'll remember them."
        else "Here are your appointments:\n\n${formatItemList(unique)}"
        return StructuredResult(answer, items, if (items.isNotEmpty()) "high" else "low")
    }

    private fun intentListMedications(): StructuredResult {
        val items = fetchEvents("medication")
        val unique = items.distinctBy { cleanDesc(it["description"]?:"").take(40) }
        val answer = if (unique.isEmpty()) "I don't have any medication information stored yet."
        else "Here are your medications:\n\n${formatItemList(unique)}"
        return StructuredResult(answer, items, if (items.isNotEmpty()) "high" else "low")
    }

    private fun intentListReminders(): StructuredResult {
        val items = mutableListOf<Map<String, String>>()
        val cur = readableDatabase.rawQuery(
            "SELECT description, raw_date, raw_time, person FROM events WHERE type IN ('task','meeting','medication') ORDER BY rowid DESC LIMIT 10", null
        )
        while (cur.moveToNext()) {
            items.add(mapOf("description" to (cur.getString(0)?:""), "date" to (cur.getString(1)?:""), "time" to (cur.getString(2)?:""), "person" to (cur.getString(3)?:"")))
        }
        cur.close()
        val unique = items.distinctBy { cleanDesc(it["description"]?:"").take(40) }
        val answer = if (unique.isEmpty()) "You don't have any upcoming reminders."
        else "Here's what you need to remember:\n\n${formatItemList(unique)}"
        return StructuredResult(answer, items, if (items.isNotEmpty()) "high" else "low")
    }

    // ── GENERAL QUERY ───────────────────────────────────────

    private fun intentGeneralQuery(question: String): StructuredResult {
        val items = mutableListOf<Map<String, String>>()
        val seen = mutableSetOf<String>()
        val stops = setOf("what","when","where","who","how","is","are","was","were","do","did","does","the","a","an","i","my","me","have","has","had","about","for","any","tell","show","find","can","you","please","know","give")
        val kws = question.lowercase().split(Regex("[\\s,;.!?]+")).filter { it.length > 2 && it !in stops }

        for (kw in kws) {
            val cur = readableDatabase.rawQuery(
                "SELECT description, raw_date, raw_time, person FROM events WHERE description LIKE ? OR person LIKE ? ORDER BY rowid DESC LIMIT 10",
                arrayOf("%$kw%", "%$kw%")
            )
            while (cur.moveToNext()) {
                val d = cur.getString(0) ?: ""
                val key = cleanDesc(d).take(40).lowercase()
                if (key !in seen && key.isNotEmpty()) {
                    seen.add(key)
                    items.add(mapOf("description" to d, "date" to (cur.getString(1)?:""), "time" to (cur.getString(2)?:""), "person" to (cur.getString(3)?:"")))
                }
            }
            cur.close()
            if (items.size >= 5) break
        }

        // Also check summaries
        try {
            val cur = readableDatabase.rawQuery("SELECT summary FROM summaries ORDER BY created_at DESC LIMIT 2", null)
            while (cur.moveToNext()) {
                val s = cur.getString(0) ?: ""
                if (s.isNotEmpty() && s.take(40).lowercase() !in seen) {
                    seen.add(s.take(40).lowercase())
                    items.add(mapOf("description" to s))
                }
            }
            cur.close()
        } catch (_: Exception) {}

        val unique = items.distinctBy { cleanDesc(it["description"]?:"").take(40) }
        val answer = if (unique.isEmpty()) {
            "I couldn't find anything specific about that yet."
        } else {
            val top = cleanDesc(unique.first()["description"].orEmpty())
            if (top.isBlank()) "I found related memory details." else top
        }
        return StructuredResult(answer, items, if (items.isNotEmpty()) "high" else "low")
    }

    // ═══════════════════════════════════════════════════════════
    //  HELPERS
    // ═══════════════════════════════════════════════════════════

    private fun fetchEvents(type: String): List<Map<String, String>> {
        val items = mutableListOf<Map<String, String>>()
        val cur = readableDatabase.rawQuery(
            "SELECT description, raw_date, raw_time, person FROM events WHERE type = ? ORDER BY rowid DESC LIMIT 10",
            arrayOf(type)
        )
        while (cur.moveToNext()) {
            items.add(mapOf(
                "description" to (cur.getString(0) ?: ""),
                "date" to (cur.getString(1) ?: ""),
                "time" to (cur.getString(2) ?: ""),
                "person" to (cur.getString(3) ?: "")
            ))
        }
        cur.close()
        return items
    }

    private fun formatItemList(items: List<Map<String, String>>): String {
        return items.take(5).mapIndexed { i, item ->
            val desc = cleanDesc(item["description"] ?: "")
            val time = item["time"]?.let { if (it.isNotEmpty()) " at $it" else "" } ?: ""
            val date = item["date"]?.let { if (it.isNotEmpty()) " on $it" else "" } ?: ""
            val person = item["person"]?.let { if (it.isNotEmpty()) " with $it" else "" } ?: ""
            "${i + 1}. $desc$time$date$person"
        }.joinToString("\n")
    }

    /** Aggressively clean ASR transcript artifacts */
    private fun cleanDesc(raw: String): String {
        var text = raw.trim()
        // Remove common ASR fillers
        val fillers = listOf(
            "hello my name is my name is", "my name is my name is",
            "hello my name is", "my name is"
        )
        for (f in fillers) {
            text = text.replace(Regex(f, RegexOption.IGNORE_CASE), "").trim()
        }
        // Remove repeated consecutive words
        val words = text.split(" ").filter { it.isNotBlank() }
        val cleaned = mutableListOf<String>()
        for (w in words) {
            if (cleaned.isEmpty() || cleaned.last().lowercase() != w.lowercase()) cleaned.add(w)
        }
        // Remove repeated consecutive phrases (2-3 word groups)
        text = cleaned.joinToString(" ")
        text = text.replace(Regex("(\\b\\w+(?:\\s+\\w+){0,2})\\s+\\1", RegexOption.IGNORE_CASE), "$1")
        return text.take(120).trim()
    }

    private fun extractTimeHint(text: String): String? {
        if (text.isBlank()) return null
        val m = Regex("\\b\\d{1,2}(?::\\d{2})?\\s?(?:AM|PM|am|pm)\\b").find(text)
        return m?.value?.trim()
    }

    private fun extractDateHint(text: String): String? {
        if (text.isBlank()) return null
        val rel = Regex("\\b(today|tomorrow|yesterday|this weekend|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\\b", RegexOption.IGNORE_CASE)
            .find(text)
            ?.value
            ?.trim()
        if (!rel.isNullOrBlank()) return rel

        val abs = Regex("\\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\\s+\\d{1,2}(?:,\\s*\\d{4})?\\b", RegexOption.IGNORE_CASE)
            .find(text)
            ?.value
            ?.trim()
        return abs
    }

    private fun extractPersonHint(text: String): String? {
        if (text.isBlank()) return null

        // Prefer explicit doctor-name patterns first.
        val dr = Regex("\\bDr\\.?\\s+[A-Z][a-z]+\\b").find(text)?.value?.trim()
        if (!dr.isNullOrBlank()) return dr

        // Safer generic person cue patterns.
        val cue = Regex("\\b(?:with|from|for|by)\\s+([A-Z][a-z]{2,}(?:\\s+[A-Z][a-z]{2,})?)\\b").find(text)
        return cue?.groupValues?.getOrNull(1)?.trim()
    }

    private fun detectAnswerStyle(question: String): String {
        val q = question.lowercase()
        return when {
            q.contains("json") -> "json"
            q.contains("bullet") || q.contains("points") || q.contains("list") -> "bullet"
            q.contains("numbered") || q.contains("steps") -> "numbered"
            q.startsWith("is ") || q.startsWith("are ") || q.startsWith("do ") || q.startsWith("does ") ||
                q.startsWith("did ") || q.startsWith("can ") || q.startsWith("was ") || q.startsWith("were ") ||
                q.startsWith("have ") || q.startsWith("has ") -> "yes_no"
            q.contains("short") || q.contains("brief") || q.contains("one line") -> "brief"
            else -> "direct"
        }
    }

    private fun styleInstruction(style: String): String {
        return when (style) {
            "json" -> "Return valid compact JSON only: {\"answer\":\"...\",\"confidence\":\"high|medium|low\"}."
            "bullet" -> "Return 2 to 5 bullet points using '- ' prefix."
            "numbered" -> "Return 2 to 5 numbered lines (1., 2., 3.)."
            "yes_no" -> "Start with 'Yes' or 'No' when possible, then one short clarification sentence."
            "brief" -> "Return one short sentence only."
            else -> "Return 1 to 3 short direct sentences."
        }
    }

    private fun applyStyleFallback(answer: String, style: String, confidence: String): String {
        val clean = answer
            .replace("\r", " ")
            .replace(Regex("\\s+"), " ")
            .trim()

        if (clean.isBlank()) return clean

        return when (style) {
            "json" -> {
                val safe = clean.replace("\"", "'")
                "{\"answer\":\"$safe\",\"confidence\":\"$confidence\"}"
            }
            "bullet" -> clean
                .split(Regex("(?<=[.!?])\\s+|\\n+"))
                .map { it.trim() }
                .filter { it.isNotBlank() }
                .take(5)
                .joinToString("\n") { "- $it" }
            "numbered" -> clean
                .split(Regex("(?<=[.!?])\\s+|\\n+"))
                .map { it.trim() }
                .filter { it.isNotBlank() }
                .take(5)
                .mapIndexed { idx, s -> "${idx + 1}. $s" }
                .joinToString("\n")
            "brief" -> clean
                .split(Regex("(?<=[.!?])\\s+"))
                .firstOrNull()
                ?.trim()
                ?: clean
            else -> clean
        }
    }

    private fun isLikelyRelevant(question: String, answer: String): Boolean {
        val stops = setOf(
            "what", "when", "where", "who", "how", "is", "are", "was", "were", "do", "does", "did",
            "a", "an", "the", "to", "for", "of", "in", "on", "my", "me", "you", "please", "about"
        )
        val qWords = question.lowercase().split(Regex("[\\s,;.!?]+"))
            .filter { it.length > 2 && it !in stops }
            .toSet()
        if (qWords.isEmpty()) return true

        val aWords = answer.lowercase().split(Regex("[\\s,;.!?]+"))
            .filter { it.length > 2 }
            .toSet()
        val overlap = qWords.count { it in aWords }
        return overlap >= 1 || answer.lowercase().contains("don't have") || answer.lowercase().contains("couldn't find")
    }

    private fun collectSpeakerEvidence(limitTurns: Int = 8): List<String> {
        val out = mutableListOf<String>()
        val c = readableDatabase.rawQuery(
            "SELECT raw_text, timestamp FROM conversations ORDER BY rowid DESC LIMIT 4",
            null
        )
        val linePattern = Regex("^([^:\\n]{2,30}):\\s*(.+)$")
        while (c.moveToNext()) {
            val raw = c.getString(0) ?: continue
            val ts = c.getString(1) ?: ""
            raw.split("\n").forEach { line ->
                val m = linePattern.find(line.trim()) ?: return@forEach
                val speaker = m.groupValues[1].trim()
                val text = cleanDesc(m.groupValues[2].trim())
                if (speaker.isNotBlank() && text.isNotBlank()) {
                    out.add("speaker=$speaker | text=$text${if (ts.isNotBlank()) " | time=$ts" else ""}")
                }
            }
            if (out.size >= limitTurns) break
        }
        c.close()
        return out.take(limitTurns)
    }

    // ═══════════════════════════════════════════════════════════
    //  ON-DEVICE FORMATTING (fully offline, no network)
    // ═══════════════════════════════════════════════════════════

    private fun formatWithLLM(question: String, result: StructuredResult): Map<String, Any>? {
        val style = detectAnswerStyle(question)
        val conf = result.confidence.ifBlank { "high" }
        val msg = applyStyleFallback(result.answer, style, conf)
        if (!isLikelyRelevant(question, msg)) return null
        return mapOf(
            "answer" to msg,
            "related_events" to emptyList<Any>(),
            "confidence" to conf,
            "mode" to "on_device",
        )
    }
}
