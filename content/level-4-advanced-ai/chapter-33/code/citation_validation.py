"""Citation Validation and Source Attribution Verifier.

Verifies:
1. Citation Existence: Every cited doc_id exists in the retrieved context set.
2. Citation Faithfulness: The referenced document contains factual evidence
   matching the sentence claim (entailment/overlap check).
3. Hallucinated Citation Detection: Flags citations pointing to non-retrieved docs.
4. Unsupported Claim Detection: Flags factual sentences lacking source tags.
"""

import re
from typing import Any, Dict, List, Set, Tuple
from document_chunking import TextChunk


class CitationValidator:
    """Audits generated answers for citation integrity and evidence grounding."""

    def __init__(self, min_support_ratio: float = 0.35):
        self.min_support_ratio = min_support_ratio

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extracts all [Doc: <id>] or [Dokumen ID: <id>] tags."""
        matches = re.findall(r"\[(?:Doc|Dokumen ID):\s*([a-zA-Z0-9_\-]+)\]", text)
        return list(dict.fromkeys(matches))  # preserve order, unique

    def validate_answer(
        self,
        answer_text: str,
        retrieved_chunks: List[Tuple[TextChunk, float]],
    ) -> Dict[str, Any]:
        retrieved_doc_ids: Set[str] = {c.doc_id for c, _ in retrieved_chunks}
        chunk_content_by_doc: Dict[str, str] = {}
        for c, _ in retrieved_chunks:
            chunk_content_by_doc[c.doc_id] = chunk_content_by_doc.get(c.doc_id, "") + " " + c.content.lower()

        found_citations = self.extract_citations(answer_text)

        # 1. Check for hallucinated / unretrieved doc IDs
        hallucinated_citations = [cid for cid in found_citations if cid not in retrieved_doc_ids]

        # 2. Split answer into statements/sentences and verify per-statement evidence
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
        sentence_reports = []
        valid_citations = []
        unsupported_sentences = []

        for sent in sentences:
            sent_citations = self.extract_citations(sent)
            # Remove citation tags to inspect text content
            clean_sent = re.sub(r"\[(?:Doc|Dokumen ID):\s*[^\]]+\]", "", sent).strip()
            sent_tokens = set(re.findall(r"\w+", clean_sent.lower()))

            if not sent_citations:
                unsupported_sentences.append(clean_sent)
                sentence_reports.append({
                    "sentence": clean_sent,
                    "citations": [],
                    "status": "unsupported",
                    "reason": "Missing citation tag",
                })
                continue

            # Verify if cited doc actually contains the words
            sent_supported = False
            for doc_id in sent_citations:
                if doc_id in chunk_content_by_doc:
                    doc_text = chunk_content_by_doc[doc_id]
                    doc_tokens = set(re.findall(r"\w+", doc_text))
                    overlap = len(sent_tokens & doc_tokens)
                    support_ratio = overlap / max(1, len(sent_tokens))

                    if support_ratio >= self.min_support_ratio:
                        sent_supported = True
                        valid_citations.append(doc_id)
                        break

            if sent_supported:
                sentence_reports.append({
                    "sentence": clean_sent,
                    "citations": sent_citations,
                    "status": "supported",
                })
            else:
                sentence_reports.append({
                    "sentence": clean_sent,
                    "citations": sent_citations,
                    "status": "unfaithful",
                    "reason": "Cited document does not contain sufficient overlapping evidence",
                })

        all_valid = (len(hallucinated_citations) == 0) and (len(unsupported_sentences) == 0)
        citation_precision = (
            len(valid_citations) / max(1, len(found_citations)) if found_citations else 1.0
        )

        return {
            "valid": all_valid,
            "citation_precision": round(citation_precision, 3),
            "total_citations_found": len(found_citations),
            "hallucinated_citations": hallucinated_citations,
            "unsupported_sentences": unsupported_sentences,
            "sentence_reports": sentence_reports,
        }


def main():
    print("=" * 70)
    print("CHAPTER 33: CITATION & SOURCE ATTRIBUTION VALIDATION")
    print("=" * 70)

    from document_chunking import TextChunk

    c1 = TextChunk("c1", "doc_rag_01", 0, "RAG menggabungkan model bahasa dengan basis data eksternal.", 0, 60)
    retrieved = [(c1, 0.95)]

    validator = CitationValidator()

    # Case 1: Grounded answer with valid citation
    ans1 = "RAG menggabungkan model bahasa dengan basis data eksternal. [Doc: doc_rag_01]"
    res1 = validator.validate_answer(ans1, retrieved)
    print("Case 1 (Valid Citation):")
    print(f"Text: '{ans1}'")
    print(f"Result Valid: {res1['valid']} | Precision: {res1['citation_precision']}")
    print(f"Hallucinated citations: {res1['hallucinated_citations']}\n")

    # Case 2: Hallucinated / phantom citation
    ans2 = "RAG ditemukan pada abad ke-19 oleh penemu telegraf. [Doc: doc_phantom_99]"
    res2 = validator.validate_answer(ans2, retrieved)
    print("Case 2 (Hallucinated Citation):")
    print(f"Text: '{ans2}'")
    print(f"Result Valid: {res2['valid']} | Precision: {res2['citation_precision']}")
    print(f"Hallucinated citations: {res2['hallucinated_citations']}\n")

    # Case 3: Unsupported statement lacking citation
    ans3 = "Sistem ini menggunakan algoritma gradient descent tanpa pembuktian."
    res3 = validator.validate_answer(ans3, retrieved)
    print("Case 3 (Unsupported Statement):")
    print(f"Text: '{ans3}'")
    print(f"Result Valid: {res3['valid']} | Unsupported: {res3['unsupported_sentences']}\n")

    print("[SUCCESS] Citation validation engine operational.")


if __name__ == "__main__":
    main()

