"""End-to-End Modular RAG Pipeline Architecture.

Connects all components into a production-grade educational pipeline:
Ingestion -> Chunking -> Hybrid Indexing -> Retrieval -> Reranking ->
Context Construction -> Grounded Answer Synthesis -> Citation Audit.
"""

import os
from typing import Any, Dict, List, Optional

from document_ingestion import DocumentIngestionPipeline
from document_chunking import RecursiveChunker, TextChunk
from rag_vector_index import RAGVectorIndex
from sparse_retrieval import BM25Retriever
from dense_retrieval import DenseRetriever
from hybrid_retrieval import HybridRetriever
from reranking_demo import HeuristicReranker
from context_builder import ContextBuilder
from grounded_answer_demo import GroundedAnswerGenerator
from citation_validation import CitationValidator


class CompleteRAGPipeline:
    """Production-style modular RAG pipeline."""

    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 40,
        embedding_dim: int = 64,
        max_context_chars: int = 1200,
    ):
        self.ingestion = DocumentIngestionPipeline()
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.sparse_retriever = BM25Retriever()
        self.vector_index = RAGVectorIndex(embedding_dim=embedding_dim)
        self.dense_retriever = DenseRetriever(self.vector_index)
        self.hybrid_retriever = HybridRetriever(self.sparse_retriever, self.dense_retriever)
        self.reranker = HeuristicReranker()
        self.context_builder = ContextBuilder(max_context_chars=max_context_chars)
        self.generator = GroundedAnswerGenerator()
        self.citation_validator = CitationValidator()

        self.indexed = False

    def load_and_index_corpus(self, json_path: str):
        """Ingests corpus documents, chunks them, and builds indexes."""
        docs = self.ingestion.ingest_from_json(json_path)
        all_chunks = []
        for d in docs:
            all_chunks.extend(self.chunker.chunk_document(d))

        self.sparse_retriever.index_chunks(all_chunks)
        self.vector_index.add_chunks(all_chunks)
        self.indexed = True

    def query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        """Executes full RAG workflow for a user query."""
        if not self.indexed:
            raise RuntimeError("Pipeline has not indexed any corpus yet. Call load_and_index_corpus() first.")

        # 1. Hybrid Retrieval (fetch candidate pool)
        raw_candidates = self.hybrid_retriever.search(user_query, top_k=top_k * 3)
        candidate_chunks = [c for c, _, _ in raw_candidates]

        # 2. Cross-Scoring Reranking
        reranked_tuples = self.reranker.rerank(user_query, candidate_chunks, top_k=top_k)

        # 3. Context Construction
        context_data = self.context_builder.build_context(user_query, reranked_tuples)

        # 4. Grounded Generation (deterministic offline baseline)
        answer_data = self.generator.generate_grounded_answer(user_query, reranked_tuples)

        # 5. Citation Validation
        audit_data = self.citation_validator.validate_answer(answer_data["answer"], reranked_tuples)

        return {
            "query": user_query,
            "status": answer_data["status"],
            "answer": answer_data["answer"],
            "confidence": answer_data["confidence"],
            "citations": answer_data["citations"],
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "rerank_score": score,
                    "title": c.metadata.get("title"),
                }
                for c, score in reranked_tuples
            ],
            "context_budget_utilized_pct": context_data["budget_utilized_pct"],
            "citation_audit": {
                "valid": audit_data["valid"],
                "precision": audit_data["citation_precision"],
                "hallucinated": audit_data["hallucinated_citations"],
            },
        }


def main():
    print("=" * 70)
    print("CHAPTER 33: COMPLETE END-TO-END MODULAR RAG PIPELINE")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = CompleteRAGPipeline()
    pipeline.load_and_index_corpus(corpus_path)

    # Test query
    q = "Bagaimana Multi-Head Self-Attention pada arsitektur Transformer memproyeksikan Query, Key, dan Value?"
    print(f"User Query: '{q}'\n")

    response = pipeline.query(q, top_k=2)

    print(f"Status: {response['status'].upper()} (Confidence: {response['confidence']})")
    print(f"Answer:\n{response['answer']}\n")
    print("Retrieved Source Attributions:")
    for c in response["retrieved_chunks"]:
        print(f"  - [{c['doc_id']}] {c['title']} (Rerank score: {c['rerank_score']})")

    print(f"\nCitation Audit: {'PASS' if response['citation_audit']['valid'] else 'FAIL'}")
    print(f"Citation Precision: {response['citation_audit']['precision']}")
    print(f"Context Budget Utilized: {response['context_budget_utilized_pct']}%")

    print("\n[SUCCESS] End-to-end RAG pipeline executed successfully.")


if __name__ == "__main__":
    main()

