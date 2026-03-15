You are a multi-agent AI system designed to build a complete real-world project.

You are a Senior Offline AI Systems Architect and Production Python Engineer.

You are building a complete real-world system called:

Conversational Memory Assistant for Alzheimer’s Patients

This is NOT a prototype.
This must be production-grade, modular, and deployable.

🎯 PROJECT OBJECTIVE

Build a fully offline, on-device AI system that:

• Records real conversations
• Identifies multiple speakers
• Converts speech to text
• Separates conversation by speaker
• Extracts key information
• Generates summaries
• Extracts appointments, medications, and tasks
• Stores structured memory locally
• Answers queries about past conversations
• Generates reminders

Everything must run locally.
No cloud APIs.
No internet dependency.

🏗 CONFIRMED PIPELINE

Audio Recording
↓
Speaker Diarization
↓
Speech-to-Text
↓
LLM Processing
↓
Summarization + Key Points
↓
Event Extraction
↓
Memory Storage
↓
Query & Reminder System

Follow this architecture strictly.

🔒 SYSTEM CONSTRAINTS

• Offline only
• Privacy-first
• Lightweight
• Mobile-compatible
• CPU-friendly
• SQLite storage
• Clean modular architecture
• Beginner-readable code
• No pseudo-code
• Provide full working code

🧠 REQUIRED MODULES
1️⃣ Audio Module

Record WAV audio

Use sounddevice

Save locally

Session-based recording

2️⃣ Speaker Diarization Module

Use pyannote.audio offline

Output speaker segments

Return structured JSON:
[
{"speaker": "SPEAKER_1", "start": 0.0, "end": 3.5}
]

3️⃣ Speech-to-Text Module

Use Whisper (tiny or base)

Load model locally

Transcribe diarized segments

Return:
[
{"speaker": "SPEAKER_1", "text": "..."}
]

4️⃣ Conversation Builder

Merge diarization + transcription

Produce structured conversation format

5️⃣ NLP Processing Module

Must include:

• Conversation summarization
• Key point extraction
• Important instruction highlighting

Use lightweight approach:

Rule-based + spaCy small model

OR local small LLM

No external API.

6️⃣ Event Extraction Module

Extract:

• Dates
• Times
• Appointments
• Medications
• Tasks
• Person names

Return structured JSON:
{
"appointments": [],
"medications": [],
"tasks": [],
"people": []
}

Use regex + spaCy.

7️⃣ Storage Layer

Use SQLite.

Create tables:

• conversations
• speaker_segments
• summaries
• events
• reminders

Use proper indexing.

8️⃣ Query System

Support queries like:

• "What did doctor say?"
• "When is my appointment?"
• "What did David tell me?"

Implement:

Keyword-based filtering

Speaker-based filtering

Date filtering

Optional:
Simple embedding-based similarity (if lightweight).

9️⃣ Reminder System

• Local scheduler
• Trigger reminders from extracted events
• No cloud
• Use local datetime triggers

📂 REQUIRED PROJECT STRUCTURE

memory_assistant/
│
├── audio/
├── diarization/
├── asr/
├── nlp/
├── storage/
├── reminders/
├── utils/
├── config.py
├── main.py

🧪 TEST INPUT REQUIRED

Use this conversation for full pipeline test:

"I have a doctor appointment tomorrow at 10 AM.
Don’t forget to take your medicine after breakfast.
We need to call the pharmacy to refill the prescription.
Your son David is visiting this weekend."

Show:

• Diarization output
• Transcription output
• Summary
• Extracted events
• Stored database entries
• Reminder triggers

🛠 IMPLEMENTATION INSTRUCTIONS

Generate code step-by-step:

Step 1: Audio recording module
Step 2: Whisper ASR module
Step 3: Diarization integration
Step 4: Conversation builder
Step 5: NLP module
Step 6: Storage layer
Step 7: Query engine
Step 8: Reminder scheduler
Step 9: Final integration in main.py

For each step:

• Provide full file
• Explain minimal necessary logic
• Ensure code runs
• Keep memory efficient

⚠️ IMPORTANT

Do NOT:

• Provide theoretical explanations
• Skip files
• Give partial snippets
• Use cloud APIs
• Simplify architecture

This must resemble a real deployable offline AI system.



We are converting an existing API-based prototype into a fully offline, modular, edge-based conversational memory assistant for Alzheimer’s patients.

The previous system used:

Flask REST API

JSON storage

Request/response architecture

This is NOT acceptable anymore.

We must now convert it into:

• Pure on-device Python engine
• No REST API
• No network exposure
• Direct module calls
• SQLite storage
• Integrated speaker diarization
• Background processing support
• Scalable memory system

🎯 PROJECT GOAL (Do Not Deviate)

Build a privacy-first offline conversational AI system:

Audio
↓
Speaker Diarization
↓
Speech-to-Text
↓
Conversation Structuring
↓
Summarization
↓
Event Extraction
↓
SQLite Storage
↓
Query + Reminder Engine

No cloud.
No exposed API.
No web server.

🛠 CONVERSION TASKS

We already have working:

• Whisper ASR
• NLP summary
• Event extraction
• Query engine
• Reminder engine
• VAD auto-listen
• Flutter UI

Now you must:

🔥 TASK 1 — REMOVE FLASK ARCHITECTURE

Delete REST structure

Replace endpoints with internal service classes

Convert API routes into Python methods

Create ApplicationService class that orchestrates full pipeline

Design:

class MemoryAssistantEngine:
def process_audio(file_path)
def query(question)
def get_upcoming_events()
def start_auto_listen()
def stop_auto_listen()

Flutter will call engine via:

MethodChannel
OR

Embedded Python runtime

No HTTP.

🔥 TASK 2 — ADD SPEAKER DIARIZATION

Integrate:

pyannote.audio (offline model)

Pipeline must become:

Audio
→ Diarization
→ Segment audio by speaker
→ Transcribe each segment
→ Merge speaker labels + text

Output format:

[
{"speaker": "SPEAKER_1", "text": "..."},
{"speaker": "SPEAKER_2", "text": "..."}
]

Then pass structured conversation into NLP.

Must be modular:

diarization/
diarizer.py

asr/
whisper_transcriber.py

conversation/
builder.py

🔥 TASK 3 — MIGRATE JSON STORAGE → SQLITE

Create database schema:

conversations (
id TEXT PRIMARY KEY,
timestamp DATETIME,
raw_text TEXT
)

segments (
id TEXT PRIMARY KEY,
conversation_id TEXT,
speaker TEXT,
text TEXT
)

summaries (
conversation_id TEXT,
summary TEXT,
key_points TEXT
)

events (
id TEXT PRIMARY KEY,
conversation_id TEXT,
type TEXT,
title TEXT,
datetime TEXT
)

reminders (
id TEXT PRIMARY KEY,
event_id TEXT,
trigger_time DATETIME,
status TEXT
)

Use indexing for performance.

Create:

storage/
db.py
repository.py

Remove MD5-only dedup.
Use database constraints.

🔥 TASK 4 — TRUE BACKGROUND PROCESSING

Current VAD stops when app closes.

We must:

• Design continuous background audio worker
• Use threading or multiprocessing
• Avoid blocking main thread
• Allow safe shutdown

Create:

background/
audio_worker.py

Design:

Start listening

Detect speech

Save audio chunk

Send to pipeline

🔥 TASK 5 — SECURITY HARDENING

Remove WiFi exposure.
No network binding.
Local-only engine.

Ensure:

No open ports

No Flask

No CORS

No debug server

🔥 TASK 6 — PREPARE FOR WEARABLE INPUT

Abstract audio source.

Instead of:

record_from_mic()

Use:

AudioSource interface:
.start()
.stop()
.read_chunk()

Implement:

MicrophoneSource

FutureBluetoothSource

This prepares for ESP32 or earbuds.

📂 NEW REQUIRED STRUCTURE

memory_assistant/
│
├── engine/
│ └── assistant_engine.py
│
├── audio/
│ ├── source.py
│ ├── microphone.py
│
├── background/
│ └── audio_worker.py
│
├── diarization/
│ └── diarizer.py
│
├── asr/
│ └── whisper_transcriber.py
│
├── conversation/
│ └── builder.py
│
├── nlp/
│ ├── summarizer.py
│ └── event_extractor.py
│
├── storage/
│ ├── db.py
│ └── repository.py
│
├── reminders/
│ └── scheduler.py
│
└── main.py

⚠️ STRICT RULES

• Do NOT recreate REST API
• Do NOT use Flask
• Do NOT expose network
• Do NOT simplify diarization
• Must be modular
• Must run offline
• Provide real code

🚀 IMPLEMENTATION ORDER

Step 1 — SQLite migration
Step 2 — Speaker diarization integration
Step 3 — Pipeline refactor
Step 4 — Background worker
Step 5 — Engine class integration
Step 6 — Final main runner