"""Document Ingestion Module for RAG Pipeline.

Demonstrates loading structured/unstructured documents, metadata extraction,
unique document ID assignment, source provenance tracking, deduplication,
and content sanitization.
"""

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional


class Document:
    """Represents an ingested document with source provenance and metadata."""

    def __init__(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "General",
        author: str = "Unknown",
        date: str = "2026-01-01",
        source_url: str = "",
        tags: Optional[List[str]] = None,
        acl_roles: Optional[List[str]] = None,
    ):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.category = category
        self.author = author
        self.date = date
        self.source_url = source_url
        self.tags = tags or []
        self.acl_roles = acl_roles or ["student", "instructor", "admin"]
        self.content_hash = self._compute_hash(content)

    @staticmethod
    def _compute_hash(text: str) -> str:
        clean = " ".join(text.strip().lower().split())
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "author": self.author,
            "date": self.date,
            "source_url": self.source_url,
            "tags": self.tags,
            "acl_roles": self.acl_roles,
            "content_length": len(self.content),
            "content_hash": self.content_hash,
            "content": self.content,
        }

    def __repr__(self) -> str:
        return f"<Document doc_id='{self.doc_id}' title='{self.title[:30]}' len={len(self.content)}>"


class DocumentIngestionPipeline:
    """Ingests raw documents from JSON/text, sanitizes text, and deduplicates."""

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.seen_hashes: Dict[str, str] = {}  # hash -> doc_id

    def sanitize_text(self, text: str) -> str:
        """Removes control characters, standardizes whitespace, and normalizes line breaks."""
        # Replace non-printable ASCII control characters except \n and \t
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize multiple spaces and tabs to single spaces within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in sanitized.splitlines()]
        # Remove consecutive blank lines
        clean_text = "\n".join(line for line in lines if line)
        return clean_text

    def ingest_from_json(self, json_path: str) -> List[Document]:
        """Loads and ingests a JSON corpus."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Corpus not found at: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_docs = data.get("documents", [])
        ingested = []

        for item in raw_docs:
            clean_content = self.sanitize_text(item.get("content", ""))
            doc_hash = Document._compute_hash(clean_content)

            # Check for exact duplicate content
            if doc_hash in self.seen_hashes:
                existing_id = self.seen_hashes[doc_hash]
                print(f"[INGEST DUP] Skipping duplicate content: {item.get('id')} matches {existing_id}")
                continue

            doc = Document(
                doc_id=item.get("id", f"doc_{len(self.documents)+1:03d}"),
                title=item.get("title", "Untitled Document"),
                content=clean_content,
                category=item.get("category", "General"),
                author=item.get("author", "Academic Staff"),
                date=item.get("date", "2026-10-01"),
                source_url=item.get("source_url", ""),
                tags=item.get("tags", []),
                acl_roles=item.get("acl_roles", ["student", "instructor", "admin"]),
            )

            self.documents[doc.doc_id] = doc
            self.seen_hashes[doc_hash] = doc.doc_id
            ingested.append(doc)

        return ingested

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "categories": sorted(list({d.category for d in self.documents.values()})),
            "average_length_chars": round(
                sum(len(d.content) for d in self.documents.values()) / max(1, len(self.documents)), 2
            ),
        }


def main():
    print("=" * 70)
    print("CHAPTER 33: DOCUMENT INGESTION & PROVENANCE PIPELINE")
    print("=" * 70)

    # Locate corpus path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    ingested = pipeline.ingest_from_json(corpus_path)

    print(f"Loaded {len(ingested)} documents from: {corpus_path}")
    print("\nCorpus Summary:")
    summary = pipeline.summary()
    print(f"- Total documents ingested: {summary['total_documents']}")
    print(f"- Categories: {', '.join(summary['categories'])}")
    print(f"- Average length: {summary['average_length_chars']} characters")

    # Display sample document with metadata
    sample = ingested[0]
    print("\nSample Ingested Document:")
    print(f"- ID: {sample.doc_id}")
    print(f"- Title: {sample.title}")
    print(f"- Category: {sample.category}")
    print(f"- Source URL: {sample.source_url}")
    print(f"- Hash (SHA-256): {sample.content_hash[:16]}...")
    print(f"- Content preview: {sample.content[:160]}...")

    print("\n[SUCCESS] Document ingestion completed with provenance tracking.")


if __name__ == "__main__":
    main()

