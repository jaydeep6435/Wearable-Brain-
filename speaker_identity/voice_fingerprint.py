"""
Voice Fingerprint Engine — Speaker Recognition via Embeddings
================================================================
Extracts voice embedding vectors from audio segments using pyannote's
embedding model. Compares embeddings via cosine similarity to identify
returning speakers across sessions.

Model: pyannote/embedding (speechbrain-based)
  - ~50 MB, runs on CPU, fully offline after first download
  - Produces 192-dim or 512-dim normalized embedding vectors

Fallback: If pyannote embedding model unavailable, all methods return
None/0.0 gracefully — the system continues without fingerprinting.

Usage:
    engine = VoiceFingerprintEngine()
    emb = engine.extract_embedding("segment.wav")
    score = engine.compare(emb_a, emb_b)  # cosine similarity
    name = engine.match_against_db(emb, stored_prints, threshold=0.75)
"""

import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Default similarity threshold for speaker matching
DEFAULT_THRESHOLD = 0.75

# Maximum number of stored embeddings per speaker (oldest rotated out)
MAX_EMBEDDINGS_PER_SPEAKER = 10

# Confidence bands for matching
CONFIDENCE_HIGH = 0.85      # Auto-assign, high certainty
CONFIDENCE_MEDIUM = 0.75    # Auto-assign, good certainty
CONFIDENCE_LOW = 0.60       # Flag for manual confirmation
# Below CONFIDENCE_LOW → no match


class VoiceFingerprintEngine:
    """
    Speaker embedding extraction and comparison.

    Uses a class-level singleton for the embedding model — loads once,
    shared across all instances. CPU-only, offline after first download.
    """

    # ── Class-level singleton ─────────────────────────────────
    _model = None
    _model_load_attempted = False
    _available = None  # None = not checked yet

    def __init__(self):
        """Initialize the engine. Model loads lazily on first use."""
        if VoiceFingerprintEngine._available is None:
            VoiceFingerprintEngine._available = self._check_available()

    # ── Availability ──────────────────────────────────────────

    @staticmethod
    def _check_available() -> bool:
        """Check if pyannote embedding model dependencies are installed."""
        try:
            from pyannote.audio import Model
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Whether the embedding model can be used."""
        return VoiceFingerprintEngine._available

    def _load_model(self):
        """Lazy-load the embedding model (singleton)."""
        if VoiceFingerprintEngine._model is not None:
            return
        if VoiceFingerprintEngine._model_load_attempted:
            return
        if not self.is_available:
            return

        VoiceFingerprintEngine._model_load_attempted = True

        try:
            from pyannote.audio import Model, Inference
            import torch

            print("[Fingerprint] Loading embedding model...")

            model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=os.environ.get("HF_TOKEN"),
            )
            model.to(torch.device("cpu"))
            model.eval()

            # Create inference pipeline
            VoiceFingerprintEngine._model = Inference(
                model,
                window="whole",
            )

            print("[Fingerprint] Embedding model loaded (CPU, singleton)")

        except Exception as e:
            print(f"[Fingerprint] Failed to load embedding model: {e}")
            print("[Fingerprint] Voice fingerprinting disabled — system continues normally")
            VoiceFingerprintEngine._available = False

    # ── Core API ──────────────────────────────────────────────

    def extract_embedding(
        self,
        audio_path: str,
        start: float = None,
        end: float = None,
    ) -> np.ndarray | None:
        """
        Extract a voice embedding vector from an audio file or segment.

        Args:
            audio_path: Path to audio file (wav, mp3, m4a).
            start: Optional start time in seconds (for segment extraction).
            end: Optional end time in seconds.

        Returns:
            Normalized embedding vector (np.ndarray, float32), or None if
            extraction fails or model is unavailable.
        """
        if not self.is_available:
            return None

        self._load_model()
        if VoiceFingerprintEngine._model is None:
            return None

        if not os.path.isfile(audio_path):
            return None

        try:
            # If start/end specified, extract segment
            if start is not None and end is not None:
                from pyannote.core import Segment
                crop = Segment(start, end)

                # Minimum segment length (0.5s)
                if end - start < 0.5:
                    return None

                embedding = VoiceFingerprintEngine._model.crop(
                    audio_path, crop
                )
            else:
                embedding = VoiceFingerprintEngine._model(audio_path)

            # Convert to numpy and normalize
            if hasattr(embedding, "numpy"):
                vec = embedding.numpy().flatten().astype(np.float32)
            elif isinstance(embedding, np.ndarray):
                vec = embedding.flatten().astype(np.float32)
            else:
                vec = np.array(embedding, dtype=np.float32).flatten()

            return self._normalize(vec)

        except Exception as e:
            print(f"[Fingerprint] Embedding extraction failed: {e}")
            return None

    @staticmethod
    def compare(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            emb_a: First embedding vector.
            emb_b: Second embedding vector.

        Returns:
            Cosine similarity score in [-1, 1]. Higher = more similar.
            Returns 0.0 if inputs are invalid.
        """
        if emb_a is None or emb_b is None:
            return 0.0
        if emb_a.size == 0 or emb_b.size == 0:
            return 0.0

        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(dot / (norm_a * norm_b))

    @staticmethod
    def compute_centroid(embeddings: list[np.ndarray]) -> np.ndarray | None:
        """
        Compute the centroid (mean vector) of multiple embeddings.

        The centroid is L2-normalized after averaging, providing a
        single representative vector for a speaker's voice model.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Normalized centroid vector, or None if input is empty.
        """
        if not embeddings:
            return None

        if len(embeddings) == 1:
            return VoiceFingerprintEngine._normalize(embeddings[0].copy())

        stacked = np.stack(embeddings, axis=0)
        centroid = np.mean(stacked, axis=0).astype(np.float32)
        return VoiceFingerprintEngine._normalize(centroid)

    @staticmethod
    def classify_confidence(score: float) -> tuple[str, bool]:
        """
        Classify a similarity score into confidence band.

        Args:
            score: Cosine similarity score.

        Returns:
            Tuple of (confidence_level, auto_assign).
            confidence_level: 'high', 'medium', 'low', or 'none'.
            auto_assign: Whether to automatically assign the label.
        """
        if score >= CONFIDENCE_HIGH:
            return "high", True
        elif score >= CONFIDENCE_MEDIUM:
            return "medium", True
        elif score >= CONFIDENCE_LOW:
            return "low", False   # Flag for manual confirmation
        else:
            return "none", False

    def match_against_db(
        self,
        embedding: np.ndarray,
        stored_voiceprints: list[dict],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict:
        """
        Compare an embedding against all stored voiceprints using
        centroid-based matching with confidence scoring.

        Strategy:
          1. Group stored embeddings by speaker
          2. Compute centroid per speaker
          3. Compare new embedding to each centroid
          4. Return best match with confidence level

        Args:
            embedding: Query embedding vector.
            stored_voiceprints: List of dicts with keys:
                speaker_name (str), embedding (bytes).
            threshold: Minimum similarity to consider a match
                       (default: CONFIDENCE_LOW for broader detection).

        Returns:
            MatchResult dict:
              matched_name (str|None), similarity_score (float),
              confidence ('high'|'medium'|'low'|'none'),
              auto_assign (bool)
        """
        empty_result = {
            "matched_name": None,
            "similarity_score": 0.0,
            "confidence": "none",
            "auto_assign": False,
        }

        if embedding is None or not stored_voiceprints:
            return empty_result

        # Group voiceprints by speaker
        speaker_embeddings: dict[str, list[np.ndarray]] = {}
        for vp in stored_voiceprints:
            name = vp.get("speaker_name", "")
            emb_bytes = vp.get("embedding")
            if not name or emb_bytes is None:
                continue

            stored_emb = self.bytes_to_embedding(emb_bytes)
            if stored_emb is not None:
                speaker_embeddings.setdefault(name, []).append(stored_emb)

        if not speaker_embeddings:
            return empty_result

        # Centroid-based matching
        best_name = None
        best_score = -1.0

        for name, embeddings in speaker_embeddings.items():
            centroid = self.compute_centroid(embeddings)
            if centroid is None:
                continue

            # Primary: compare against centroid
            centroid_score = self.compare(embedding, centroid)

            # Secondary: verify against top-3 nearest individual embeddings
            individual_scores = sorted(
                [self.compare(embedding, e) for e in embeddings],
                reverse=True,
            )
            top_k = individual_scores[:3]
            top_k_avg = sum(top_k) / len(top_k) if top_k else 0.0

            # Final score: weighted blend (70% centroid, 30% top-K)
            final_score = 0.7 * centroid_score + 0.3 * top_k_avg

            if final_score > best_score:
                best_score = final_score
                best_name = name

        # Classify confidence
        confidence, auto_assign = self.classify_confidence(best_score)

        # Only return a match if above the low threshold
        if best_score >= CONFIDENCE_LOW:
            return {
                "matched_name": best_name,
                "similarity_score": round(best_score, 4),
                "confidence": confidence,
                "auto_assign": auto_assign,
            }

        return {
            "matched_name": None,
            "similarity_score": round(best_score, 4),
            "confidence": "none",
            "auto_assign": False,
        }

    # ── Serialization ─────────────────────────────────────────

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Convert embedding vector to bytes for SQLite BLOB storage."""
        if embedding is None:
            return b""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(data: bytes) -> np.ndarray | None:
        """Convert bytes from SQLite BLOB back to embedding vector."""
        if not data:
            return None
        return np.frombuffer(data, dtype=np.float32).copy()

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return vec / norm


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    engine = VoiceFingerprintEngine()
    print(f"Embedding model available: {engine.is_available}")

    # Test serialization round-trip
    fake_emb = np.random.randn(192).astype(np.float32)
    fake_emb = VoiceFingerprintEngine._normalize(fake_emb)
    as_bytes = engine.embedding_to_bytes(fake_emb)
    restored = engine.bytes_to_embedding(as_bytes)
    assert np.allclose(fake_emb, restored), "Round-trip failed"
    print(f"Serialization test: {len(as_bytes)} bytes, {restored.shape}")

    # Test cosine similarity
    sim = engine.compare(fake_emb, fake_emb)
    print(f"Self-similarity: {sim:.4f} (should be ~1.0)")

    different = np.random.randn(192).astype(np.float32)
    different = VoiceFingerprintEngine._normalize(different)
    sim2 = engine.compare(fake_emb, different)
    print(f"Random similarity: {sim2:.4f} (should be ~0.0)")

    print("Quick test passed!")
