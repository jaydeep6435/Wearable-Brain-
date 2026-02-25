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

    // ── Query ────────────────────────────────────────────────

    fun queryMemory(question: String): Map<String, Any> {
        val lower = question.lowercase()
        Log.i(TAG, "⤷ queryMemory: '$question'")

        // Remove stop words, keep meaningful keywords
        val stopWords = setOf(
            "what", "when", "where", "who", "how", "is", "are", "was", "were",
            "do", "did", "does", "the", "a", "an", "i", "my", "me", "have",
            "has", "had", "about", "for", "any", "tell", "show", "find",
            "can", "you", "please", "know"
        )
        val keywords = lower.split(Regex("[\\s,;.!?]+"))
            .filter { it.length > 2 && it !in stopWords }

        Log.d(TAG, "  Keywords: $keywords")

        if (keywords.isEmpty()) {
            Log.w(TAG, "  No keywords extracted")
            return mapOf(
                "answer" to "Please ask a more specific question.",
                "results" to emptyList<Map<String, Any>>(),
                "method" to "keyword"
            )
        }

        val results = mutableListOf<Map<String, String?>>()
        val seen = mutableSetOf<String>()

        // Search events
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

        // Also search conversation raw text
        for (kw in keywords) {
            val cursor = readableDatabase.rawQuery(
                """SELECT id, raw_text, timestamp FROM conversations 
                   WHERE raw_text LIKE ? ORDER BY timestamp DESC LIMIT 5""",
                arrayOf("%$kw%")
            )
            while (cursor.moveToNext()) {
                val text = cursor.getString(cursor.getColumnIndexOrThrow("raw_text")) ?: ""
                val snippet = text.take(200)
                if (snippet.lowercase() !in seen) {
                    seen.add(snippet.lowercase())
                    results.add(mapOf(
                        "type" to "conversation",
                        "description" to snippet,
                        "raw_date" to cursor.getString(cursor.getColumnIndexOrThrow("timestamp")),
                        "raw_time" to null,
                        "person" to null
                    ))
                }
            }
            cursor.close()
        }

        // Build answer
        val answer = if (results.isEmpty()) {
            "No memories found matching: ${keywords.joinToString(", ")}."
        } else {
            val parts = results.take(5).map { r ->
                val desc = r["description"] ?: "Unknown"
                val type = r["type"] ?: ""
                val time = r["raw_time"]?.let { " at $it" } ?: ""
                val person = r["person"]?.let { " (with $it)" } ?: ""
                "• [$type] $desc$time$person"
            }
            "Found ${results.size} memory/memories:\n${parts.joinToString("\n")}"
        }

        Log.i(TAG, "  → ${results.size} results found")

        return mapOf(
            "answer" to answer,
            "results" to results,
            "method" to "keyword"
        )
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
            "config" to mapOf(
                "simplified_mode" to false,
                "low_resource_mode" to false
            )
        )
    }

    fun getMemoryCount(): Int {
        val count = countTable("conversations") + countTable("events")
        Log.i(TAG, "Memory count: $count")
        return count
    }

    fun getAllEvents(typeFilter: String? = null): List<Map<String, Any?>> {
        val results = mutableListOf<Map<String, Any?>>()
        val query = if (typeFilter != null) {
            "SELECT * FROM events WHERE type = ? ORDER BY rowid DESC"
        } else {
            "SELECT * FROM events ORDER BY rowid DESC"
        }
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
}
