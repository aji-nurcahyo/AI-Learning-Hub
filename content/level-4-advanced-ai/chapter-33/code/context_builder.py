"""Context Construction Engine for RAG Prompts.

Builds structured, grounded prompt contexts under strict budget constraints:
1. Context budget enforcement (character/token limit).
2. De-duplication and redundancy pruning.
3. Explicit document ID mapping and source citation tags.
4. Clear system delimiters to isolate untrusted context data from instructions.
"""

from typing import Any, Dict, List, Optional, Tuple
from document_chunking import TextChunk


class ContextBuilder:
    """Assembles retrieved chunks into a bounded, formatted context window."""

    def __init__(self, max_context_chars: int = 1500, include_metadata: bool = True):
        self.max_context_chars = max_context_chars
        self.include_metadata = include_metadata

    def build_context(
        self,
        query: str,
        retrieved_chunks: List[Tuple[TextChunk, float]],
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Builds a grounded prompt context string with source citation headers."""
        if system_instruction is None:
            system_instruction = (
                "Anda adalah asisten AI akademis. Jawablah pertanyaan HANYA berdasarkan konteks "
                "yang diberikan di bawah ini. Wajib sertakan sitasi nomor dokumen [Doc: <ID>] untuk "
                "setiap fakta atau klaim yang Anda sebutkan. Jika informasi dalam konteks tidak mencukupi, "
                "nyatakan secara jujur bahwa bukti tidak ditemukan dan tolak berspekulasi (abstain)."
            )

        context_blocks = []
        current_len = 0
        used_chunks = []
        seen_texts = set()

        for chunk, score in retrieved_chunks:
            # Simple exact duplication prune
            snippet_key = chunk.content[:80].strip()
            if snippet_key in seen_texts:
                continue
            seen_texts.add(snippet_key)

            # Format individual chunk with citation header
            header = f"[Dokumen ID: {chunk.doc_id} | Chunk: {chunk.chunk_id} | Skor: {score:.3f}]"
            block = f"{header}\n{chunk.content}\n"

            if current_len + len(block) > self.max_context_chars:
                # Budget exceeded, stop packing
                break

            context_blocks.append(block)
            current_len += len(block)
            used_chunks.append(chunk)

        formatted_context = "\n".join(context_blocks)

        prompt = (
            f"<SYSTEM_INSTRUCTION>\n{system_instruction}\n</SYSTEM_INSTRUCTION>\n\n"
            f"<RETRIEVED_CONTEXT>\n{formatted_context if formatted_context else '[TIDAK ADA KONTEKS RELEVAN]'}\n</RETRIEVED_CONTEXT>\n\n"
            f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
            f"<GROUNDED_ANSWER>\n"
        )

        return {
            "prompt": prompt,
            "used_chunk_count": len(used_chunks),
            "used_chunk_ids": [c.chunk_id for c in used_chunks],
            "context_char_length": current_len,
            "budget_limit": self.max_context_chars,
            "budget_utilized_pct": round((current_len / self.max_context_chars) * 100, 1),
        }


def main():
    print("=" * 70)
    print("CHAPTER 33: CONTEXT CONSTRUCTION & PROMPT PACKING")
    print("=" * 70)

    # Mock sample chunks for demonstration
    from document_chunking import TextChunk

    c1 = TextChunk(
        "doc_rag_01_c001",
        "doc_rag_01",
        0,
        "Retrieval-Augmented Generation (RAG) menggabungkan model parametrik dengan basis data eksternal.",
        0,
        96,
        {"title": "Pengantar RAG"},
    )
    c2 = TextChunk(
        "doc_ft_01_c001",
        "doc_ft_01",
        0,
        "Fine-tuning memperbarui bobot model secara permanen, berbeda dengan RAG yang hanya memperkaya prompt runtime.",
        0,
        111,
        {"title": "Perbedaan RAG dan FT"},
    )

    builder = ContextBuilder(max_context_chars=600)
    res = builder.build_context(
        query="Apa keunggulan RAG dibanding Fine-Tuning untuk data dinamis?",
        retrieved_chunks=[(c1, 0.88), (c2, 0.82)],
    )

    print("Constructed Grounded Prompt:\n")
    print(res["prompt"])
    print("-" * 70)
    print(f"Context Statistics:")
    print(f"- Used Chunks: {res['used_chunk_count']} ({res['used_chunk_ids']})")
    print(f"- Context Size: {res['context_char_length']} / {res['budget_limit']} chars ({res['budget_utilized_pct']}%)")
    print("\n[SUCCESS] Context construction engine operational.")


if __name__ == "__main__":
    main()

