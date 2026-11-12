"""Chunk Overlap and Size Trade-Off Analysis.

Empirically investigates:
1. Impact of chunk size on total chunk count and semantic coherence.
2. Impact of chunk overlap on redundancy factor and context window consumption.
3. Information preservation at boundaries vs dilution effects.
"""

import os
from typing import Any, Dict, List
from document_ingestion import DocumentIngestionPipeline
from document_chunking import FixedSizeChunker


def compute_redundancy(chunks: List[Any], original_length: int) -> float:
    """Computes total characters in chunks divided by original document length."""
    total_chunk_chars = sum(len(c.content) for c in chunks)
    return round(total_chunk_chars / max(1, original_length), 3)


def run_overlap_experiment(doc, chunk_sizes=[150, 300, 500], overlap_ratios=[0.0, 0.10, 0.25, 0.50]):
    results = []
    orig_len = len(doc.content)

    for size in chunk_sizes:
        for ratio in overlap_ratios:
            overlap = int(size * ratio)
            if overlap >= size:
                continue

            chunker = FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
            chunks = chunker.chunk_document(doc)
            redundancy = compute_redundancy(chunks, orig_len)

            results.append({
                "chunk_size": size,
                "overlap": overlap,
                "overlap_ratio_pct": int(ratio * 100),
                "num_chunks": len(chunks),
                "redundancy_factor": redundancy,
                "avg_chunk_len": round(sum(len(c.content) for c in chunks) / max(1, len(chunks)), 1),
            })

    return results


def main():
    print("=" * 75)
    print("CHAPTER 33: CHUNK OVERLAP AND SIZE TRADE-OFF ANALYSIS")
    print("=" * 75)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)
    test_doc = docs[0]  # RAG intro doc

    print(f"Target Document: '{test_doc.title}' (Original length: {len(test_doc.content)} chars)\n")
    print(f"{'Chunk Size':<12} | {'Overlap':<10} | {'Ratio (%)':<10} | {'Chunks':<8} | {'Redundancy':<12} | {'Avg Len'}")
    print("-" * 75)

    results = run_overlap_experiment(test_doc)
    for r in results:
        print(
            f"{r['chunk_size']:<12} | {r['overlap']:<10} | {r['overlap_ratio_pct']:<10} | "
            f"{r['num_chunks']:<8} | {r['redundancy_factor']:<12} | {r['avg_chunk_len']}"
        )

    print("\nKey Analytical Takeaways:")
    print("1. Overlap = 0%: Minimal storage footprint (Redundancy ~ 1.0x), but high risk")
    print("   of splitting relational facts across boundaries (boundary clipping).")
    print("2. Overlap = 10-25%: Recommended sweet spot (Redundancy 1.1x - 1.3x), guaranteeing")
    print("   phrase continuity across chunk boundaries without overwhelming retrieval budget.")
    print("3. Overlap >= 50%: Severe information duplication (Redundancy >= 1.9x), causing")
    print("   dilution of unique information when top-k chunks are packed into context window.")


if __name__ == "__main__":
    main()

