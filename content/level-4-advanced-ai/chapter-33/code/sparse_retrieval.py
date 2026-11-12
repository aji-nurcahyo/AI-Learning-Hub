"""Sparse Lexical Retrieval Engine for RAG.

Implements BM25 / TF-IDF keyword-matching retrieval over chunks.
Demonstrates exact term matching, inverted indexing, and score calculation.
"""

import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from document_ingestion import DocumentIngestionPipeline
from document_chunking import RecursiveChunker, TextChunk


class BM25Retriever:
    """Okapi BM25 Sparse Lexical Retriever implemented in pure Python."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[TextChunk] = []
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> [(chunk_idx, freq)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index_chunks(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self.doc_len = []
        self.doc_freqs.clear()
        self.inverted_index.clear()

        total_tokens = 0
        n_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            tokens = self.tokenize(chunk.content)
            dl = len(tokens)
            self.doc_len.append(dl)
            total_tokens += dl

            counts = Counter(tokens)
            for term, freq in counts.items():
                self.doc_freqs[term] += 1
                self.inverted_index[term].append((idx, freq))

        self.avg_dl = total_tokens / max(1, n_chunks)

        # Precompute IDF for all indexed terms
        self.idf.clear()
        for term, df in self.doc_freqs.items():
            # Standard Lucene/BM25 IDF formula
            idf_score = math.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.0, idf_score)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        q_tokens = self.tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        scores = defaultdict(float)

        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]

            for chunk_idx, tf in self.inverted_index[term]:
                dl = self.doc_len[chunk_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                scores[chunk_idx] += term_idf * (numerator / max(1e-9, denominator))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_results = [(self.chunks[idx], score) for idx, score in ranked[:top_k]]
        return top_results


def main():
    print("=" * 70)
    print("CHAPTER 33: SPARSE LEXICAL RETRIEVAL (BM25)")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)

    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40)
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    retriever = BM25Retriever()
    retriever.index_chunks(chunks)

    # Test exact acronym / keyword query
    query = "Non-Maximum Suppression IoU bounding box YOLO"
    print(f"Query: '{query}'")
    results = retriever.search(query, top_k=3)

    print("\nTop BM25 Retrieved Results:")
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"[{rank}] BM25 Score: {score:.4f} | ID: {chunk.chunk_id}")
        print(f"    Parent: {chunk.metadata.get('title')}")
        print(f"    Excerpt: {chunk.content[:140]}...\n")

    print("[SUCCESS] BM25 sparse retrieval operational.")


if __name__ == "__main__":
    main()

