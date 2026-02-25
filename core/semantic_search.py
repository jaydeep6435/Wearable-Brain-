"""
Semantic Search — TF-IDF + Sentence Embedding for Memory Queries
=================================================================
Two search engines:

1. EmbeddingSearch (primary) — all-MiniLM-L6-v2 sentence embeddings
   - ~22 MB model, 384-dim vectors, CPU-friendly
   - Handles synonyms, paraphrases, concept-level similarity
   - Requires: sentence-transformers

2. SemanticSearch (fallback) — TF-IDF cosine similarity
   - Pure Python, zero dependencies
   - Keyword-level matching only

Usage:
    # Primary (embedding-based)
    emb_search = EmbeddingSearch()
    if emb_search.is_available:
        results = emb_search.search("doctor appointment", documents)

    # Fallback (TF-IDF)
    tfidf = SemanticSearch()
    tfidf.index(documents)
    results = tfidf.search("doctor appointment")
"""

import re
import math
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Model name for sentence embeddings
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


class EmbeddingSearch:
    """
    Sentence-embedding semantic search using all-MiniLM-L6-v2.

    Handles synonyms (doctor ≈ physician), paraphrases, and
    concept-level similarity — far superior to TF-IDF for
    natural language queries.

    Model: ~22 MB, 384-dim vectors, CPU-friendly, fully offline.
    Uses class-level singleton — loads once, shared across instances.
    """

    # ── Class-level singleton ─────────────────────────────────
    _model = None
    _model_load_attempted = False
    _available = None  # None = not checked yet

    def __init__(self):
        """Initialize. Model loads lazily on first encode() call."""
        if EmbeddingSearch._available is None:
            EmbeddingSearch._available = self._check_available()

    # ── Availability ──────────────────────────────────────────

    @staticmethod
    def _check_available() -> bool:
        """Check if sentence-transformers is installed."""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Whether the embedding model can be used."""
        return EmbeddingSearch._available

    def _load_model(self):
        """Lazy-load the sentence-transformer model (singleton)."""
        if EmbeddingSearch._model is not None:
            return
        if EmbeddingSearch._model_load_attempted:
            return
        if not self.is_available:
            return

        EmbeddingSearch._model_load_attempted = True

        try:
            from sentence_transformers import SentenceTransformer

            print(f"[SemanticSearch] Loading embedding model: {_EMBEDDING_MODEL}")
            EmbeddingSearch._model = SentenceTransformer(
                _EMBEDDING_MODEL,
                device="cpu",
            )
            print(f"[SemanticSearch] Model loaded ({_EMBEDDING_DIM}-dim, CPU)")

        except Exception as e:
            print(f"[SemanticSearch] Failed to load embedding model: {e}")
            print("[SemanticSearch] Falling back to TF-IDF search")
            EmbeddingSearch._available = False

    # ── Core API ──────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray | None:
        """
        Encode text into a normalized embedding vector.

        Args:
            text: Input text string.

        Returns:
            Normalized 384-dim float32 vector, or None if unavailable.
        """
        if not self.is_available:
            return None

        self._load_model()
        if EmbeddingSearch._model is None:
            return None

        try:
            embedding = EmbeddingSearch._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"[SemanticSearch] Encoding failed: {e}")
            return None

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Encode multiple texts in a single batch (more efficient).

        Args:
            texts: List of text strings.

        Returns:
            List of normalized embedding vectors.
        """
        if not self.is_available or not texts:
            return []

        self._load_model()
        if EmbeddingSearch._model is None:
            return []

        try:
            embeddings = EmbeddingSearch._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            print(f"[SemanticSearch] Batch encoding failed: {e}")
            return []

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Both vectors should be L2-normalized for optimal performance.

        Returns:
            Float in [-1, 1]. Higher = more similar.
        """
        if vec_a is None or vec_b is None:
            return 0.0
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0

        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def search(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        threshold: float = 0.2,
        precomputed_embeddings: dict | None = None,
    ) -> list[dict]:
        """
        Search documents using sentence embedding similarity.

        Args:
            query: Natural language search string.
            documents: List of dicts with at least a "text" field.
            top_k: Maximum results to return.
            threshold: Minimum similarity score to include.
            precomputed_embeddings: Optional dict mapping doc index
                or conversation_id to precomputed embedding vectors.

        Returns:
            List of {document, score, rank} dicts, sorted by score.
        """
        if not self.is_available or not documents:
            return []

        query_emb = self.encode(query)
        if query_emb is None:
            return []

        # Build document texts
        scores = []
        for i, doc in enumerate(documents):
            doc_emb = None

            # Try precomputed embedding first
            if precomputed_embeddings:
                doc_id = doc.get("conversation_id") or doc.get("id") or str(i)
                doc_emb = precomputed_embeddings.get(doc_id)
                if isinstance(doc_emb, bytes):
                    doc_emb = np.frombuffer(doc_emb, dtype=np.float32).copy()

            # Compute on the fly if not precomputed
            if doc_emb is None:
                text = self._build_text(doc)
                if not text.strip():
                    continue
                doc_emb = self.encode(text)

            if doc_emb is None:
                continue

            sim = self.cosine_similarity(query_emb, doc_emb)
            if sim >= threshold:
                scores.append((i, sim))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], 1):
            results.append({
                "document": documents[idx],
                "score": round(score, 4),
                "rank": rank,
            })

        return results

    @staticmethod
    def _build_text(doc: dict) -> str:
        """Build searchable text from document fields."""
        parts = []
        for key in ("text", "description", "summary", "type", "person", "speaker"):
            if doc.get(key):
                parts.append(str(doc[key]))
        if doc.get("key_points"):
            kp = doc["key_points"]
            if isinstance(kp, list):
                parts.extend(str(k) for k in kp)
            elif isinstance(kp, str):
                parts.append(kp)
        return " ".join(parts)

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Convert embedding to bytes for SQLite BLOB storage."""
        if embedding is None:
            return b""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(data: bytes) -> np.ndarray | None:
        """Convert bytes from SQLite BLOB back to embedding vector."""
        if not data:
            return None
        return np.frombuffer(data, dtype=np.float32).copy()


# ═══════════════════════════════════════════════════════════════
#  TF-IDF Fallback (preserved from original)
# ═══════════════════════════════════════════════════════════════


class SemanticSearch:
    """
    Lightweight TF-IDF semantic search engine.

    Uses pure Python (no sklearn required) — zero external dependencies.
    Computes TF-IDF vectors and cosine similarity for ranking.

    Designed for small-to-medium document sets (100–10,000 docs).
    """

    def __init__(self):
        """Initialize the search engine."""
        self._documents: list[dict] = []    # Original docs
        self._texts: list[str] = []          # Cleaned text per doc
        self._vocab: dict[str, int] = {}     # word → index
        self._idf: dict[str, float] = {}     # word → IDF score
        self._tfidf_matrix: list[dict] = []  # Sparse TF-IDF per doc
        self._indexed = False

    # ── Indexing ──────────────────────────────────────────────

    def index(self, documents: list[dict]) -> int:
        """
        Build the TF-IDF index from a list of documents.

        Args:
            documents: List of dicts, each with at least a "text" field.
                       May also have "id", "type", "description", etc.

        Returns:
            Number of documents indexed.
        """
        self._documents = documents
        self._texts = []
        self._vocab = {}
        self._idf = {}
        self._tfidf_matrix = []

        if not documents:
            self._indexed = False
            return 0

        # Step 1: Tokenize all documents
        doc_tokens: list[list[str]] = []
        for doc in documents:
            text = self._build_text(doc)
            self._texts.append(text)
            tokens = self._tokenize(text)
            doc_tokens.append(tokens)

        # Step 2: Build vocabulary
        all_words = set()
        for tokens in doc_tokens:
            all_words.update(tokens)

        self._vocab = {word: idx for idx, word in enumerate(sorted(all_words))}

        # Step 3: Compute IDF
        n_docs = len(doc_tokens)
        doc_freq: Counter = Counter()
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        self._idf = {}
        for word, df in doc_freq.items():
            # IDF = log(N / df) + 1 (smooth)
            self._idf[word] = math.log(n_docs / df) + 1.0

        # Step 4: Compute TF-IDF vectors (sparse)
        self._tfidf_matrix = []
        for tokens in doc_tokens:
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            tfidf_vec = {}
            for word, count in tf.items():
                tf_val = count / total
                idf_val = self._idf.get(word, 1.0)
                tfidf_vec[word] = tf_val * idf_val
            self._tfidf_matrix.append(tfidf_vec)

        self._indexed = True
        return len(documents)

    def search(
        self, query: str, top_k: int = 5, threshold: float = 0.05
    ) -> list[dict]:
        """
        Search for documents matching the query.

        Args:
            query: Natural language search string.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0–1) to include.

        Returns:
            List of dicts with "document", "score", and "rank" fields.
            Sorted by score descending.
        """
        if not self._indexed or not self._documents:
            return []

        # Vectorize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        total = len(query_tokens)
        query_vec = {}
        for word, count in query_tf.items():
            if word in self._idf:
                tf_val = count / total
                query_vec[word] = tf_val * self._idf[word]

        if not query_vec:
            return []

        # Compute cosine similarity with each document
        scores = []
        for i, doc_vec in enumerate(self._tfidf_matrix):
            sim = self._cosine_similarity(query_vec, doc_vec)
            if sim >= threshold:
                scores.append((i, sim))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-K
        results = []
        for rank, (idx, score) in enumerate(scores[:top_k], 1):
            results.append({
                "document": self._documents[idx],
                "score": round(score, 4),
                "rank": rank,
            })

        return results

    @property
    def is_indexed(self) -> bool:
        """Whether the index has been built."""
        return self._indexed

    @property
    def doc_count(self) -> int:
        """Number of indexed documents."""
        return len(self._documents) if self._indexed else 0

    @property
    def vocab_size(self) -> int:
        """Number of unique terms in the vocabulary."""
        return len(self._vocab)

    # ── Internal Helpers ──────────────────────────────────────

    @staticmethod
    def _build_text(doc: dict) -> str:
        """
        Build a searchable text representation from a document dict.

        Combines all relevant text fields for better matching.
        """
        parts = []

        # Primary text
        if doc.get("text"):
            parts.append(str(doc["text"]))

        # Description (events)
        if doc.get("description"):
            parts.append(str(doc["description"]))

        # Type
        if doc.get("type"):
            parts.append(str(doc["type"]))

        # Person
        if doc.get("person"):
            parts.append(str(doc["person"]))

        # Speaker
        if doc.get("speaker"):
            parts.append(str(doc["speaker"]))

        # Summary
        if doc.get("summary"):
            parts.append(str(doc["summary"]))

        # Key points
        if doc.get("key_points"):
            kp = doc["key_points"]
            if isinstance(kp, list):
                parts.extend(str(k) for k in kp)
            elif isinstance(kp, str):
                parts.append(kp)

        # Date/time context
        if doc.get("raw_date"):
            parts.append(str(doc["raw_date"]))
        if doc.get("raw_time"):
            parts.append(str(doc["raw_time"]))

        return " ".join(parts).lower()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text into meaningful words.

        Removes stop words and short tokens for better TF-IDF quality.
        """
        # Common English stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "because",
            "but", "or", "and", "if", "while", "about", "up", "down",
            "it", "its", "this", "that", "these", "those", "he", "she",
            "they", "them", "his", "her", "their", "what", "which", "who",
            "whom", "i", "me", "my", "we", "us", "our", "you", "your",
            "am", "any",
        }

        # Extract words (alphanumeric, 2+ chars)
        words = re.findall(r"[a-z0-9]{2,}", text.lower())
        return [w for w in words if w not in stop_words]

    @staticmethod
    def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
        """
        Compute cosine similarity between two sparse vectors.

        Args:
            vec_a: Dict mapping terms to weights.
            vec_b: Dict mapping terms to weights.

        Returns:
            Float between 0 and 1 (1 = identical).
        """
        # Dot product (only shared keys contribute)
        shared_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not shared_keys:
            return 0.0

        dot = sum(vec_a[k] * vec_b[k] for k in shared_keys)

        # Magnitudes
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("SemanticSearch — Quick Test")
    print("=" * 40)

    docs = [
        {"text": "Doctor appointment tomorrow at 10 AM", "type": "meeting"},
        {"text": "Take medicine after breakfast", "type": "medication"},
        {"text": "Call pharmacy to refill prescription", "type": "task"},
        {"text": "Son David visiting this weekend", "type": "meeting"},
        {"text": "Blood pressure was normal", "type": "note"},
        {"text": "Need to buy groceries from the store", "type": "task"},
    ]

    searcher = SemanticSearch()
    count = searcher.index(docs)
    print(f"Indexed {count} documents, vocab size: {searcher.vocab_size}")

    queries = [
        "when is my doctor visit",
        "what medication do I need",
        "pharmacy prescription",
        "who is coming to visit",
        "health checkup",
    ]

    for q in queries:
        results = searcher.search(q, top_k=3)
        print(f"\n  Q: {q}")
        for r in results:
            print(f"    [{r['score']:.3f}] {r['document']['text']}")

    print("\nQuick test passed!")
