"""Cross-Scoring Reranker Demonstration for RAG Pipelines.

Re-scores candidate chunks retrieved in the first stage using fine-grained
cross-matching features:
1. Exact query phrase match bonus.
2. Title relevance alignment.
3. Content term-frequency density.
Reorders candidates to present the highest-precision chunks to the context builder.
"""

import os
import re
from typing import Any, Dict, List, Tuple

from document_ingestion import DocumentIngestionPipeline
from document_chunking import RecursiveChunker, TextChunk
from rag_vector_index import RAGVectorIndex
from sparse_retrieval import BM25Retriever
from dense_retrieval import DenseRetriever
from hybrid_retrieval import HybridRetriever


class HeuristicReranker:
    """CPU-friendly cross-scoring reranker simulating neural cross-encoder mechanics."""

    def __init__(self, exact_phrase_weight: float = 0.40, title_weight: float = 0.25, density_weight: float = 0.35):
        self.exact_phrase_weight = exact_phrase_weight
        self.title_weight = title_weight
        self.density_weight = density_weight

    def score_pair(self, query: str, chunk: TextChunk) -> float:
        q_clean = query.lower()
        c_clean = chunk.content.lower()
        title_clean = chunk.metadata.get("title", "").lower()

        # 1. Exact sub-phrase matching (longest common token sequence)
        q_tokens = re.findall(r"\w+", q_clean)
        exact_score = 0.0
        # Check 2-gram and 3-gram matches
        if len(q_tokens) >= 2:
            n_grams = [" ".join(q_tokens[i : i + 2]) for i in range(len(q_tokens) - 1)]
            matched_ngrams = sum(1 for ng in n_grams if ng in c_clean)
            exact_score = matched_ngrams / len(n_grams)

        # 2. Title relevance alignment
        matched_title = sum(1 for t in q_tokens if t in title_clean)
        title_score = matched_title / max(1, len(q_tokens))

        # 3. Term frequency density in content
        matched_content = sum(1 for t in set(q_tokens) if t in c_clean)
        density_score = matched_content / max(1, len(set(q_tokens)))

        final_score = (
            self.exact_phrase_weight * exact_score
            + self.title_weight * title_score
            + self.density_weight * density_score
        )
        return round(final_score, 4)

    def rerank(self, query: str, candidates: List[TextChunk], top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        scored = []
        for chunk in candidates:
            score = self.score_pair(query, chunk)
            scored.append((chunk, score))

        # Sort descending
        reranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


def main():
    print("=" * 70)
    print("CHAPTER 33: CROSS-SCORING RERANKING DEMONSTRATION")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)

    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40)
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    sparse = BM25Retriever()
    sparse.index_chunks(chunks)

    vec_idx = RAGVectorIndex(embedding_dim=64)
    vec_idx.add_chunks(chunks)
    dense = DenseRetriever(vec_idx)

    hybrid = HybridRetriever(sparse, dense)

    query = "Bagaimana arsitektur YOLO mendeteksi objek secara real-time dengan grid?"
    print(f"Query: '{query}'")

    # Retrieve pool of 6 candidates
    raw_hybrid = hybrid.search(query, top_k=6)
    candidate_chunks = [c for c, _, _ in raw_hybrid]

    print("\n[Before Reranking] Initial Hybrid Candidates:")
    for idx, (c, score, _) in enumerate(raw_hybrid[:3], 1):
        print(f"  Rank {idx} | {c.chunk_id} | {c.metadata.get('title')}")

    reranker = HeuristicReranker()
    reranked_results = reranker.rerank(query, candidate_chunks, top_k=3)

    print("\n[After Reranking] Reranked High-Precision Chunks:")
    for idx, (c, score) in enumerate(reranked_results, 1):
        print(f"  Rank {idx} | Score: {score:.4f} | {c.chunk_id} | {c.metadata.get('title')}")
        print(f"         Excerpt: {c.content[:120]}...\n")

    print("[SUCCESS] Reranking demonstration completed.")


if __name__ == "__main__":
    main()

