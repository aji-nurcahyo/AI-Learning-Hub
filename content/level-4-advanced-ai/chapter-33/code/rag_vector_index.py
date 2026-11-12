"""Vector Indexing Engine for RAG Chunks.

Implements an in-memory FAISS-compatible vector index using pure NumPy.
Applies L2-normalization for exact Cosine Similarity via Inner Product (Dot Product).
Preserves chunk metadata and supports top-k nearest neighbor retrieval.
"""

import hashlib
import os
import re
from typing import Any, Dict, List, Tuple
import numpy as np

from document_ingestion import DocumentIngestionPipeline
from document_chunking import RecursiveChunker, TextChunk


class DeterministicSentenceEncoder:
    """Lightweight, deterministic sentence encoder running 100% offline on CPU.

    Uses subword n-gram character hashing projected onto a d-dimensional space
    with a fixed Gaussian projection matrix, followed by L2-normalization.
    """

    def __init__(self, embedding_dim: int = 64, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Random projection matrix from hash space (2048) to embedding_dim
        self.num_buckets = 2048
        self.projection = rng.normal(0.0, 1.0 / np.sqrt(embedding_dim), (self.num_buckets, embedding_dim))

    def _tokenize_to_features(self, text: str) -> np.ndarray:
        clean = text.lower().strip()
        words = re.findall(r"\w+", clean)
        features = np.zeros(self.num_buckets, dtype=np.float32)

        for w in words:
            # Word token hash
            h_word = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.num_buckets
            features[h_word] += 1.5

            # Character 3-grams
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    ngram = w[i : i + 3]
                    h_ng = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16) % self.num_buckets
                    features[h_ng] += 0.8

        return features

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            feat = self._tokenize_to_features(t)
            v = feat @ self.projection
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                v = v / norm
            else:
                v = np.zeros(self.embedding_dim, dtype=np.float32)
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)


class RAGVectorIndex:
    """FAISS-compatible in-memory vector index for RAG chunks."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.encoder = DeterministicSentenceEncoder(embedding_dim=embedding_dim)
        self.vectors: Optional[np.ndarray] = None
        self.chunks: List[TextChunk] = []

    def add_chunks(self, chunks: List[TextChunk]):
        """Encodes and indexes a list of text chunks."""
        texts = [c.content for c in chunks]
        embeddings = self.encoder.encode(texts)

        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """Retrieves top-k chunks with highest cosine similarity scores."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        q_vec = self.encoder.encode([query])[0]
        # Since vectors are L2-normalized, dot product is exact cosine similarity
        scores = self.vectors @ q_vec

        # Get top-k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def __len__(self) -> int:
        return len(self.chunks)


def main():
    print("=" * 70)
    print("CHAPTER 33: RAG VECTOR INDEXING & RETRIEVAL ENGINE")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)

    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40)
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))

    print(f"Ingested {len(docs)} documents -> Generated {len(all_chunks)} chunks.")

    # Build Vector Index
    index = RAGVectorIndex(embedding_dim=64)
    index.add_chunks(all_chunks)
    print(f"Indexed {len(index)} chunk vectors with dimension d=64.")

    # Test Query
    query = "Bagaimana arsitektur RAG mengatasi keterbatasan knowledge cutoff pada model bahasa?"
    print(f"\nTest Query: '{query}'")
    top_results = index.search(query, top_k=3)

    print("\nTop Retrieved Chunks:")
    for rank, (chunk, score) in enumerate(top_results, 1):
        print(f"[{rank}] Score: {score:.4f} | ID: {chunk.chunk_id} (Doc: {chunk.doc_id})")
        print(f"    Title: {chunk.metadata.get('title')}")
        print(f"    Excerpt: {chunk.content[:140]}...\n")

    print("[SUCCESS] RAG Vector Index operational with exact Cosine Similarity.")


if __name__ == "__main__":
    main()

