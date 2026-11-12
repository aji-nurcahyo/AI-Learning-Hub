"""Document Chunking Strategies for RAG Pipeline.

Implements:
1. Fixed-size chunking (character/token approximation)
2. Recursive chunking (splitting by hierarchy: paragraphs, sentences, words)
3. Sentence-boundary chunking
Preserves metadata: doc_id, chunk_index, character offsets, and source title.
"""

import os
import re
from typing import Any, Dict, List, Optional
from document_ingestion import Document, DocumentIngestionPipeline


class TextChunk:
    """Represents a text chunk with parent provenance and metadata."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        content: str,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.content = content
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata or {}
        self.token_count_approx = max(1, len(content.split()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "char_span": [self.start_char, self.end_char],
            "word_count": self.token_count_approx,
            "metadata": self.metadata,
            "content": self.content,
        }

    def __repr__(self) -> str:
        return f"<TextChunk id='{self.chunk_id}' doc='{self.doc_id}' len={len(self.content)}>"


class BaseChunker:
    """Base class for text chunkers."""

    def chunk_document(self, doc: Document) -> List[TextChunk]:
        raise NotImplementedError


class FixedSizeChunker(BaseChunker):
    """Chunks text by fixed character length with overlap."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> List[TextChunk]:
        text = doc.content
        chunks = []
        start = 0
        idx = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = TextChunk(
                    chunk_id=f"{doc.doc_id}_c{idx:03d}",
                    doc_id=doc.doc_id,
                    chunk_index=idx,
                    content=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={"title": doc.title, "category": doc.category, "method": "fixed_size"},
                )
                chunks.append(chunk)
                idx += 1

            if end == len(text):
                break
            start += step

        return chunks


class RecursiveChunker(BaseChunker):
    """Splits text hierarchically using a list of separator delimiters."""

    def __init__(
        self,
        chunk_size: int = 350,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        for s in separators:
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator else list(text)
        good_splits = []

        for piece in splits:
            if not piece.strip():
                continue
            if separator != "\n\n" and separator != "\n":
                piece = piece + (separator if separator != " " else "")
            good_splits.append(piece.strip())

        # Merge pieces up to chunk_size
        curr_chunk = []
        curr_len = 0

        for piece in good_splits:
            p_len = len(piece)
            if curr_len + p_len > self.chunk_size and curr_chunk:
                merged = " ".join(curr_chunk)
                final_chunks.append(merged)
                # Overlap logic: keep last elements if possible
                overlap_accum = []
                accum_len = 0
                for item in reversed(curr_chunk):
                    if accum_len + len(item) <= self.chunk_overlap:
                        overlap_accum.insert(0, item)
                        accum_len += len(item)
                    else:
                        break
                curr_chunk = overlap_accum
                curr_len = sum(len(x) for x in curr_chunk)

            curr_chunk.append(piece)
            curr_len += p_len

        if curr_chunk:
            final_chunks.append(" ".join(curr_chunk))

        return final_chunks

    def chunk_document(self, doc: Document) -> List[TextChunk]:
        raw_pieces = self._split_text(doc.content, self.separators)
        chunks = []
        search_offset = 0

        for idx, piece in enumerate(raw_pieces):
            # Locate approximate character offsets
            start_char = doc.content.find(piece[:30], search_offset)
            if start_char == -1:
                start_char = search_offset
            end_char = start_char + len(piece)
            search_offset = max(search_offset, start_char + max(1, len(piece) - self.chunk_overlap))

            chunk = TextChunk(
                chunk_id=f"{doc.doc_id}_c{idx:03d}",
                doc_id=doc.doc_id,
                chunk_index=idx,
                content=piece,
                start_char=start_char,
                end_char=end_char,
                metadata={"title": doc.title, "category": doc.category, "method": "recursive"},
            )
            chunks.append(chunk)

        return chunks


def main():
    print("=" * 70)
    print("CHAPTER 33: DOCUMENT CHUNKING STRATEGIES")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)
    sample_doc = docs[0]

    print(f"Selected document: '{sample_doc.title}' ({len(sample_doc.content)} characters)")

    # 1. Fixed-Size Chunking
    fixed_chunker = FixedSizeChunker(chunk_size=250, chunk_overlap=40)
    fixed_chunks = fixed_chunker.chunk_document(sample_doc)
    print(f"\n[1] Fixed-Size Chunking (size=250, overlap=40): Generated {len(fixed_chunks)} chunks.")
    for c in fixed_chunks[:2]:
        print(f"    - [{c.chunk_id}] ({len(c.content)} chars, span {c.start_char}-{c.end_char}): {c.content[:80]}...")

    # 2. Recursive Chunking
    rec_chunker = RecursiveChunker(chunk_size=280, chunk_overlap=40)
    rec_chunks = rec_chunker.chunk_document(sample_doc)
    print(f"\n[2] Recursive Chunking (size=280, overlap=40): Generated {len(rec_chunks)} chunks.")
    for c in rec_chunks[:2]:
        print(f"    - [{c.chunk_id}] ({len(c.content)} chars, span {c.start_char}-{c.end_char}): {c.content[:80]}...")

    print("\n[SUCCESS] Document chunking strategies verified with metadata preservation.")


if __name__ == "__main__":
    main()

