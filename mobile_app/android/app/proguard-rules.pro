# Sherpa JNI accesses Java fields by exact names (GetFieldID).
# Keep these classes and members stable in release builds.
-keep class com.k2fsa.sherpa.onnx.** { *; }
-keepclassmembers class com.k2fsa.sherpa.onnx.** { *; }
-keepnames class com.k2fsa.sherpa.onnx.**

# Keep app-side bridge classes used with JNI/model loading.
-keep class com.example.memory_assistant.SherpaTranscriber { *; }
