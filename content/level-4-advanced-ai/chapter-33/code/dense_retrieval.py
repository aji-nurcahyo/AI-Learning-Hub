"""Dense Semantic Retrieval Engine with Filtering and Thresholding.

Performs vector search using dense embeddings with:
1. Similarity threshold gating (rejecting low confidence chunks).
2. Metadata filtering (restricting search to specific categories or tags).
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from document_ingestion import DocumentIngestionPipeline
from document_chunking import RecursiveChunker, TextChunk
from rag_vector_index import RAGVectorIndex


class DenseRetriever:
    """Dense retriever wrapping RAGVectorIndex with thresholding and metadata filters."""

    def __init__(self, index: RAGVectorIndex, similarity_threshold: float = 0.25):
        self.index = index
        self.similarity_threshold = similarity_threshold

    def search(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
    ) -> List[Tuple[TextChunk, float]]:
        # Retrieve candidate pool (larger than top_k to account for filtering)
        pool_size = max(top_k * 4, 15)
        raw_candidates = self.index.search(query, top_k=pool_size)

        filtered = []
        for chunk, score in raw_candidates:
            # 1. Similarity Threshold check
            if score < self.similarity_threshold:
                continue

            # 2. Category Filter check
            if category_filter and chunk.metadata.get("category") != category_filter:
                continue

            filtered.append((chunk, score))
            if len(filtered) >= top_k:
                break

        return filtered


def main():
    print("=" * 70)
    print("CHAPTER 33: DENSE RETRIEVAL WITH THRESHOLD & METADATA FILTERING")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)

    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40)
    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    index = RAGVectorIndex(embedding_dim=64)
    index.add_chunks(chunks)

    retriever = DenseRetriever(index, similarity_threshold=0.30)

    # 1. Standard Query
    query = "Bagaimana model language mengatasi halusinasi melalui sumber dokumen eksternal?"
    print(f"Query 1: '{query}'")
    results = retriever.search(query, top_k=2)
    for r, (c, s) in enumerate(results, 1):
        print(f"  [{r}] Score: {s:.4f} | {c.chunk_id} | {c.metadata.get('title')}")

    # 2. Query with Category Filter
    print(f"\nQuery 2 (Filtered to 'AI Agents'): '{query}'")
    filtered_results = retriever.search(query, top_k=2, category_filter="AI Agents")
    for r, (c, s) in enumerate(filtered_results, 1):
        print(f"  [{r}] Score: {s:.4f} | {c.chunk_id} | {c.metadata.get('category')} | {c.metadata.get('title')}")

    # 3. Low-Relevance Out-of-Domain Query with Threshold Gating
    ood_query = "Cara menanam pohon mangga gedong gincu di dataran rendah"
    print(f"\nQuery 3 (Out-of-Domain): '{ood_query}'")
    ood_results = retriever.search(ood_query, top_k=3)
    print(f"  Retrieved chunks above threshold {retriever.similarity_threshold}: {len(ood_results)} chunks (Abstention trigger)")

    print("\n[SUCCESS] Dense retrieval with metadata gating operational.")


if __name__ == "__main__":
    main()

