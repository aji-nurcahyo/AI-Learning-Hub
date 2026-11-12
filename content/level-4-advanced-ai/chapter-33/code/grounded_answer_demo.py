"""Deterministic Grounded Answer Generator Baseline.

Provides an offline, deterministic grounded answer baseline for RAG:
1. Performs extractive sentence selection based on query-token overlap.
2. Formulates synthesized answers strictly from extracted context.
3. Automatically maps factual assertions to document citations [Doc: <id>].
4. Enforces answer abstention when context is insufficient or relevance is low.

NOTE: This is a deterministic rule-based baseline designed for education,
correctness verification, and reproducible offline evaluation.
It does NOT perform neural generative text generation (LLM inference).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from document_chunking import TextChunk


class GroundedAnswerGenerator:
    """Deterministic extractor and answer synthesizer for offline RAG demonstration."""

    def __init__(self, min_token_overlap: int = 2, max_sentences: int = 3):
        self.min_token_overlap = min_token_overlap
        self.max_sentences = max_sentences

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        # Split on period, exclamation, question mark followed by space or newline
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) > 10]

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[TextChunk, float]],
    ) -> Dict[str, Any]:
        """Synthesizes a grounded answer or abstains if insufficient evidence."""
        # 1. Check for empty or non-relevance abstention
        if not retrieved_chunks:
            return {
                "answer": "Maaf, sistem tidak menemukan bukti faktual yang relevan dalam korpus dokumen untuk menjawab pertanyaan ini.",
                "status": "abstained",
                "citations": [],
                "extracted_sentences": [],
                "confidence": 0.0,
            }

        q_tokens = set(re.findall(r"\w+", query.lower()))
        # Remove common indonesian stopwords for matching
        stopwords = {"yang", "dan", "di", "dari", "ke", "untuk", "pada", "adalah", "ini", "itu", "atau", "dengan", "apa", "bagaimana", "mengapa"}
        q_keywords = q_tokens - stopwords

        candidate_sentences = []

        for chunk, score in retrieved_chunks:
            sentences = self._split_into_sentences(chunk.content)
            for s in sentences:
                s_tokens = set(re.findall(r"\w+", s.lower()))
                overlap = len(q_keywords & s_tokens)
                if overlap >= self.min_token_overlap:
                    candidate_sentences.append({
                        "sentence": s,
                        "overlap": overlap,
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "chunk_score": score,
                    })

        # 2. Check if any relevant sentence was found
        if not candidate_sentences:
            return {
                "answer": "Bukti tidak mencukupi: Tidak ditemukan informasi yang secara eksplisit menjawab pertanyaan dalam dokumen yang ditelusuri.",
                "status": "abstained",
                "citations": [],
                "extracted_sentences": [],
                "confidence": 0.0,
            }

        # 3. Sort sentences by token overlap and chunk relevance
        candidate_sentences.sort(key=lambda x: (x["overlap"], x["chunk_score"]), reverse=True)

        selected = candidate_sentences[: self.max_sentences]
        citations = []
        answer_parts = []

        for item in selected:
            sent = item["sentence"]
            doc_id = item["doc_id"]
            if doc_id not in citations:
                citations.append(doc_id)
            # Append with citation tag
            answer_parts.append(f"{sent} [Doc: {doc_id}]")

        answer_text = " ".join(answer_parts)

        return {
            "answer": answer_text,
            "status": "grounded",
            "citations": citations,
            "extracted_sentences": [item["sentence"] for item in selected],
            "confidence": round(sum(item["chunk_score"] for item in selected) / len(selected), 4),
        }


def main():
    print("=" * 70)
    print("CHAPTER 33: DETERMINISTIC GROUNDED ANSWER GENERATOR (BASELINE)")
    print("=" * 70)

    from document_chunking import TextChunk

    c1 = TextChunk(
        "doc_rag_01_c001",
        "doc_rag_01",
        0,
        "Retrieval-Augmented Generation (RAG) menggabungkan model bahasa parametrik dengan basis retrieval eksternal. RAG mengatasi keterbatasan knowledge cutoff dan mengurangi halusinasi faktual.",
        0,
        180,
        {"title": "Pengantar RAG"},
    )

    generator = GroundedAnswerGenerator(min_token_overlap=1)

    # Case A: Answerable Query
    q_valid = "Bagaimana RAG mengatasi keterbatasan knowledge cutoff dan halusinasi?"
    res_valid = generator.generate_grounded_answer(q_valid, [(c1, 0.92)])
    print(f"Query 1: '{q_valid}'")
    print(f"Status: {res_valid['status'].upper()} (Confidence: {res_valid['confidence']})")
    print(f"Answer: {res_valid['answer']}")
    print(f"Citations: {res_valid['citations']}\n")

    # Case B: Unanswerable Out-of-Domain Query (Abstention)
    q_invalid = "Berapa massa jenis unsur kimia plutonium pada suhu kamar?"
    res_invalid = generator.generate_grounded_answer(q_invalid, [])
    print(f"Query 2: '{q_invalid}'")
    print(f"Status: {res_invalid['status'].upper()}")
    print(f"Answer: {res_invalid['answer']}")
    print(f"Citations: {res_invalid['citations']}\n")

    print("[SUCCESS] Grounded answer generator baseline operational.")


if __name__ == "__main__":
    main()

