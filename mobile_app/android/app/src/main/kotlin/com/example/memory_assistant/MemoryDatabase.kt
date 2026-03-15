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
class MemoryDatabase(context: Context) : SQLiteOpenHelper(
    context, "memory.db", null, DB_VERSION
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
        // Deduplicate by fingerprint
        val fp = "$type|${description.lowercase().trim()}|${date ?: ""}|${time ?: ""}"
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
            put("raw_date", date)
            put("raw_time", time)
            put("person", person)
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
        val lower = question.lowercase()
        Log.i(TAG, "⤷ queryMemory: '$question'")

        val stopWords = setOf(
            "what", "when", "where", "who", "how", "is", "are", "was", "were",
            "do", "did", "does", "the", "a", "an", "i", "my", "me", "have",
            "has", "had", "about", "for", "any", "tell", "show", "find",
            "can", "you", "please", "know"
        )
        val keywords = lower.split(Regex("[\\s,;.!?]+"))
            .filter { it.length > 2 && it !in stopWords }

        if (keywords.isEmpty()) {
            return mapOf(
                "answer" to "Please ask a more specific question.",
                "results" to emptyList<Map<String, Any>>(),
                "method" to "keyword"
            )
        }

        val results = mutableListOf<Map<String, String?>>()
        val seen = mutableSetOf<String>()

        for (kw in keywords) {
            val cursor = readableDatabase.rawQuery(
                """SELECT * FROM events
                   WHERE description LIKE ? OR type LIKE ? OR person LIKE ?
                   ORDER BY rowid DESC LIMIT 20""",
                arrayOf("%$kw%", "%$kw%", "%$kw%")
            )
            while (cursor.moveToNext()) {
                val desc = cursor.getString(cursor.getColumnIndexOrThrow("description")) ?: ""
                if (desc.lowercase() !in seen) {
                    seen.add(desc.lowercase())
                    results.add(mapOf(
                        "type" to cursor.getString(cursor.getColumnIndexOrThrow("type")),
                        "description" to desc,
                        "raw_date" to cursor.getString(cursor.getColumnIndexOrThrow("raw_date")),
                        "raw_time" to cursor.getString(cursor.getColumnIndexOrThrow("raw_time")),
                        "person" to cursor.getString(cursor.getColumnIndexOrThrow("person"))
                    ))
                }
            }
            cursor.close()
        }

        val answer = if (results.isEmpty()) {
            "No memories found matching: ${keywords.joinToString(", ")}."
        } else {
            val parts = results.take(5).map { r ->
                val desc = r["description"] ?: "Unknown"
                val time = r["raw_time"]?.let { " at $it" } ?: ""
                val person = r["person"]?.let { " (with $it)" } ?: ""
                "• $desc$time$person"
            }
            "Found ${results.size} memory/memories:\n${parts.joinToString("\n")}"
        }

        Log.i(TAG, "  → ${results.size} results found")
        return mapOf("answer" to answer, "results" to results, "method" to "keyword")
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
        "APPOINTMENT"        -> intentListAppointments()
        "MEDICATION"         -> intentListMedications()
        "REMINDER"           -> intentListReminders()
        else                 -> intentGeneralQuery(question)
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
        val answer = if (unique.isEmpty()) "I couldn't find anything about that. Try asking about appointments, medications, or people by name."
        else "Here's what I found:\n\n${formatItemList(unique)}"
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

    // ═══════════════════════════════════════════════════════════
    //  LLM FORMATTING (optional — reformats pre-computed answer)
    // ═══════════════════════════════════════════════════════════

    private fun formatWithLLM(question: String, result: StructuredResult): Map<String, Any>? {
        val prompt = """You are a compassionate memory assistant for an Alzheimer patient.
Reformat the following answer to sound warm, clear, and human.

STRICT RULES:
- Speak calmly and reassuringly
- Keep ALL specific details (times, dates, names, counts)
- Do NOT add new information or invent anything
- Do NOT show raw transcript text
- Do NOT use labels like [meeting], [task], [conversation]
- Remove any duplicate phrases
- Present appointments as simple bullet points
- If counting, give the number first
- Keep response concise (under 100 words)
- Be warm and human, like a caring family member

COMPUTED ANSWER:
${result.answer}

PATIENT ASKED: $question

Return JSON only: {"message": "your warm answer", "tone": "calm", "type": "${result.confidence}"}
JSON:"""

        val url = java.net.URL("http://localhost:11434/api/generate")
        val conn = url.openConnection() as java.net.HttpURLConnection
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true
        conn.connectTimeout = 3000
        conn.readTimeout = 20000

        val body = org.json.JSONObject().apply {
            put("model", "phi3"); put("prompt", prompt); put("stream", false)
            put("options", org.json.JSONObject().apply { put("temperature", 0.3); put("num_predict", 250) })
        }
        conn.outputStream.bufferedWriter().use { it.write(body.toString()) }
        if (conn.responseCode != 200) return null

        val resp = conn.inputStream.bufferedReader().readText()
        val llmOut = org.json.JSONObject(resp).optString("response", "")
        if (llmOut.isBlank()) return null

        // Parse JSON (supports both "answer" and "message" keys)
        try {
            val p = org.json.JSONObject(llmOut)
            val a = p.optString("message", p.optString("answer", ""))
            if (a.isNotEmpty()) return mapOf("answer" to a, "related_events" to emptyList<Any>(), "confidence" to p.optString("type", p.optString("confidence","medium")), "mode" to "llm")
        } catch (_: Exception) {}

        val m = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL).find(llmOut)
        if (m != null) {
            try {
                val p = org.json.JSONObject(m.value)
                val a = p.optString("message", p.optString("answer", ""))
                if (a.isNotEmpty()) return mapOf("answer" to a, "related_events" to emptyList<Any>(), "confidence" to p.optString("type", p.optString("confidence","medium")), "mode" to "llm")
            } catch (_: Exception) {}
        }

        // Use cleaned raw text if substantial
        val cleaned = llmOut.trim().replace(Regex("\\[meeting]|\\[task]|\\[conversation]|Found \\d+ memory.*?:", RegexOption.IGNORE_CASE), "").trim()
        if (cleaned.length > 20) return mapOf("answer" to cleaned, "related_events" to emptyList<Any>(), "confidence" to "medium", "mode" to "llm")

        return null
    }
}
