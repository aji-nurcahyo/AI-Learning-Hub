"""Hybrid Retrieval Engine Combining Sparse and Dense Search.

Implements Reciprocal Rank Fusion (RRF) to merge ranked candidate lists
from BM25 (lexical precision) and Dense Vector search (semantic recall).
RRF Formula: Score(d) = sum( 1 / (k + rank_m(d)) ) for each retriever m.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from document_ingestion import DocumentIngestionPipeline
from document_chunking import RecursiveChunker, TextChunk
from rag_vector_index import RAGVectorIndex
from sparse_retrieval import BM25Retriever
from dense_retrieval import DenseRetriever


class HybridRetriever:
    """Fuses Sparse (BM25) and Dense (Cosine) retrievers via Reciprocal Rank Fusion."""

    def __init__(
        self,
        sparse_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ):
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3, candidate_pool: int = 10) -> List[Tuple[TextChunk, float, Dict[str, Any]]]:
        # 1. Retrieve candidates from both systems
        sparse_results = self.sparse.search(query, top_k=candidate_pool)
        dense_results = self.dense.search(query, top_k=candidate_pool)

        # 2. Map chunk_id to ranks and objects
        chunk_map: Dict[str, TextChunk] = {}
        sparse_ranks: Dict[str, int] = {}
        dense_ranks: Dict[str, int] = {}

        for rank, (chunk, score) in enumerate(sparse_results, 1):
            chunk_map[chunk.chunk_id] = chunk
            sparse_ranks[chunk.chunk_id] = rank

        for rank, (chunk, score) in enumerate(dense_results, 1):
            chunk_map[chunk.chunk_id] = chunk
            dense_ranks[chunk.chunk_id] = rank

        # 3. Compute RRF score
        rrf_scores = defaultdict(float)
        all_ids = set(sparse_ranks.keys()) | set(dense_ranks.keys())

        for cid in all_ids:
            score = 0.0
            if cid in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[cid])
            if cid in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[cid])
            rrf_scores[cid] = score

        # 4. Rank by combined RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for cid, score in ranked[:top_k]:
            chunk = chunk_map[cid]
            detail = {
                "rrf_score": round(score, 6),
                "sparse_rank": sparse_ranks.get(cid, None),
                "dense_rank": dense_ranks.get(cid, None),
            }
            final_results.append((chunk, score, detail))

        return final_results


def main():
    print("=" * 70)
    print("CHAPTER 33: HYBRID RETRIEVAL VIA RECIPROCAL RANK FUSION (RRF)")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)

    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40)
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    sparse_retriever = BM25Retriever()
    sparse_retriever.index_chunks(chunks)

    vector_index = RAGVectorIndex(embedding_dim=64)
    vector_index.add_chunks(chunks)
    dense_retriever = DenseRetriever(vector_index, similarity_threshold=0.20)

    hybrid = HybridRetriever(sparse_retriever, dense_retriever, rrf_k=60)

    query = "Perbedaan mekanisme adaptasi RAG dan LoRA pada pembaharuan data"
    print(f"Query: '{query}'")
    results = hybrid.search(query, top_k=3)

    print("\nTop Hybrid Retrieved Chunks (RRF Score):")
    for r, (c, score, detail) in enumerate(results, 1):
        print(f"[{r}] RRF Score: {score:.6f} | ID: {c.chunk_id}")
        print(f"    Sparse Rank: {detail['sparse_rank']} | Dense Rank: {detail['dense_rank']}")
        print(f"    Document: {c.metadata.get('title')}")
        print(f"    Excerpt: {c.content[:130]}...\n")

    print("[SUCCESS] Hybrid retrieval (Sparse + Dense RRF) verified.")


if __name__ == "__main__":
    main()
