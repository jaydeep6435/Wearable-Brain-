# Phi-2 GGUF Local Model Integration

This project now uses a local GGUF model workflow for the answer model card (no Ollama dependency).

## Selected model

- Model file: `phi-2.Q4_K_M.gguf`
- Storage path (Android app private):
  - `/data/user/0/com.example.memory_assistant/files/models/phi2/phi-2.Q4_K_M.gguf`

## What is implemented

- Local model download manager:
  - `android/app/src/main/kotlin/com/example/memory_assistant/Phi2ModelManager.kt`
- Existing UI buttons are wired to local file download flow:
  - `Download`, `Pause`, `Resume`, `Retry`
- Status fields (`ready`, `initializing`, `progress`, `downloaded_mb`, `total_mb`, `error`) are emitted through the same MethodChannel APIs.

## Pipeline connection point

- Native bridge:
  - `android/app/src/main/kotlin/com/example/memory_assistant/MainActivity.kt`
  - Methods: `checkLlmStatus`, `startLlmDownload`, `pauseLlmDownload`, `resumeLlmDownload`, `retryLlmDownload`
- Memory reasoning path:
  - `android/app/src/main/kotlin/com/example/memory_assistant/MemoryDatabase.kt`
  - `formatWithLLM(...)` is already fully offline and safe if the local model runtime is not attached yet.

## Runtime note

This change handles model asset download and local storage.
To execute GGUF token generation with `llama.cpp`, add an Android-native runtime bridge (JNI/native binary) and call it from `MemoryDatabase.kt`.

Reference command (desktop):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
./main -m phi-2.Q4_K_M.gguf -n 128
```

On Android, replace `./main` with a packaged Android native binary or JNI wrapper and point to the stored model path above.
