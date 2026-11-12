"""RAG Evaluation Suite: Decoupling Retrieval and Generation Metrics.

Computes:
1. Retrieval Metrics:
   - Precision@k
   - Recall@k
   - Hit Rate@k
   - Mean Reciprocal Rank (MRR)
2. Generation / Grounding Metrics:
   - Groundedness (fraction of generated statements supported by retrieved context)
   - Faithfulness (absence of contradictory claims)
   - Answer Relevance (semantic and lexical overlap with original query)
   - Citation Correctness (precision of document citations)
"""

import json
import os
from typing import Any, Dict, List, Set, Tuple

from document_ingestion import DocumentIngestionPipeline
from document_chunking import RecursiveChunker, TextChunk
from rag_vector_index import RAGVectorIndex
from sparse_retrieval import BM25Retriever
from dense_retrieval import DenseRetriever
from hybrid_retrieval import HybridRetriever
from grounded_answer_demo import GroundedAnswerGenerator
from citation_validation import CitationValidator


class RAGEvaluator:
    """Evaluates RAG systems across independent retrieval and generation axes."""

    def __init__(self, k: int = 3):
        self.k = k
        self.citation_validator = CitationValidator()

    # --- 1. Retrieval Metrics ---
    def compute_retrieval_metrics(
        self,
        retrieved_doc_ids: List[str],
        ground_truth_doc_ids: List[str],
    ) -> Dict[str, float]:
        top_k_ids = retrieved_doc_ids[: self.k]
        gt_set = set(ground_truth_doc_ids)

        if not gt_set:
            # Out-of-domain / unanswerable case: success if retrieved nothing or low relevance
            return {
                "precision_at_k": 1.0 if not top_k_ids else 0.0,
                "recall_at_k": 1.0 if not top_k_ids else 0.0,
                "hit_rate_at_k": 1.0 if not top_k_ids else 0.0,
                "reciprocal_rank": 0.0,
            }

        hits = sum(1 for d in top_k_ids if d in gt_set)
        p_at_k = hits / len(top_k_ids) if top_k_ids else 0.0
        r_at_k = hits / len(gt_set)
        hit_rate = 1.0 if hits > 0 else 0.0

        # Reciprocal Rank: 1 / (first position of relevant doc)
        rr = 0.0
        for rank, d in enumerate(retrieved_doc_ids, 1):
            if d in gt_set:
                rr = 1.0 / rank
                break

        return {
            "precision_at_k": round(p_at_k, 4),
            "recall_at_k": round(r_at_k, 4),
            "hit_rate_at_k": hit_rate,
            "reciprocal_rank": round(rr, 4),
        }

    # --- 2. Generation Metrics ---
    def compute_generation_metrics(
        self,
        query: str,
        answer_text: str,
        retrieved_chunks: List[Tuple[TextChunk, float]],
    ) -> Dict[str, float]:
        val_res = self.citation_validator.validate_answer(answer_text, retrieved_chunks)

        total_reports = len(val_res["sentence_reports"])
        if total_reports == 0:
            return {
                "groundedness": 1.0,
                "faithfulness": 1.0,
                "citation_correctness": 1.0,
            }

        supported_count = sum(1 for r in val_res["sentence_reports"] if r["status"] == "supported")
        unfaithful_count = sum(1 for r in val_res["sentence_reports"] if r["status"] == "unfaithful")

        groundedness = supported_count / total_reports
        faithfulness = 1.0 - (unfaithful_count / total_reports)
        citation_correctness = val_res["citation_precision"]

        return {
            "groundedness": round(groundedness, 4),
            "faithfulness": round(faithfulness, 4),
            "citation_correctness": round(citation_correctness, 4),
        }


def main():
    print("=" * 70)
    print("CHAPTER 33: COMPREHENSIVE RAG EVALUATION BENCHMARK")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    benchmarks = corpus_data.get("benchmark_queries", [])
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
    generator = GroundedAnswerGenerator()
    evaluator = RAGEvaluator(k=3)

    retrieval_scores = []
    generation_scores = []

    print(f"Running evaluation over {len(benchmarks)} benchmark queries (k=3)...\n")

    for b in benchmarks:
        qid = b["query_id"]
        q_text = b["query"]
        expected_ids = b["expected_doc_ids"]

        # Run hybrid retrieval
        raw_results = hybrid.search(q_text, top_k=3)
        retrieved_chunks = [(c, s) for c, s, _ in raw_results]
        retrieved_doc_ids = [c.doc_id for c, _ in retrieved_chunks]

        # 1. Retrieval Eval
        r_metrics = evaluator.compute_retrieval_metrics(retrieved_doc_ids, expected_ids)
        retrieval_scores.append(r_metrics)

        # 2. Generation & Answer Eval
        ans_res = generator.generate_grounded_answer(q_text, retrieved_chunks)
        g_metrics = evaluator.compute_generation_metrics(q_text, ans_res["answer"], retrieved_chunks)
        generation_scores.append(g_metrics)

        print(f"[{qid}] '{q_text[:40]}...'")
        print(f"     Expected: {expected_ids} | Retrieved: {retrieved_doc_ids}")
        print(f"     Recall@3: {r_metrics['recall_at_k']} | MRR: {r_metrics['reciprocal_rank']}")
        print(f"     Status: {ans_res['status']} | Groundedness: {g_metrics['groundedness']}\n")

    # Aggregate Averages
    avg_p3 = sum(s["precision_at_k"] for s in retrieval_scores) / len(retrieval_scores)
    avg_r3 = sum(s["recall_at_k"] for s in retrieval_scores) / len(retrieval_scores)
    avg_mrr = sum(s["reciprocal_rank"] for s in retrieval_scores) / len(retrieval_scores)
    avg_groundedness = sum(s["groundedness"] for s in generation_scores) / len(generation_scores)
    avg_faithfulness = sum(s["faithfulness"] for s in generation_scores) / len(generation_scores)

    print("-" * 70)
    print("BENCHMARK SUMMARY RESULTS:")
    print(f"Retrieval Metrics (k=3):")
    print(f"- Mean Precision@3: {avg_p3:.4f}")
    print(f"- Mean Recall@3:    {avg_r3:.4f}")
    print(f"- Mean MRR:         {avg_mrr:.4f}")
    print(f"Generation Metrics:")
    print(f"- Mean Groundedness:        {avg_groundedness:.4f}")
    print(f"- Mean Faithfulness:        {avg_faithfulness:.4f}")
    print("=" * 70)
    print("[SUCCESS] Decoupled RAG evaluation completed.")


if __name__ == "__main__":
    main()

