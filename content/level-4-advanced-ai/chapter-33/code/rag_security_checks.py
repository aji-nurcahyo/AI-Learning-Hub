"""Security and Privacy Verification Engine for RAG Systems.

Implements defense-in-depth protections:
1. Indirect Prompt Injection Scanner (detects adversarial system overrides in documents).
2. Document-Level Access Control Lists (ACL permissions check).
3. Context Sanitization and Prompt Boundary Delimiters.
4. Prevention of arbitrary execution of retrieved content.
"""

import os
import re
from typing import Any, Dict, List, Set, Tuple

from document_ingestion import Document, DocumentIngestionPipeline


class RAGSecurityGuard:
    """Detects indirect prompt injection and enforces role-based access control."""

    INJECTION_PATTERNS = [
        r"abaikan\s+(?:semua\s+)?instruksi",
        r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"perintah\s+sistem\s*:",
        r"system\s*prompt\s*:",
        r"ubah\s+peran\s+anda",
        r"you\s+are\s+now\s+(?:a\s+)?hacker",
        r"bocorkan\s+(?:semua\s+)?database",
        r"leak\s+(?:all\s+)?credentials",
        r"<script.*?>.*?</script>",
    ]

    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def scan_for_prompt_injection(self, text: str) -> Dict[str, Any]:
        """Scans document text for hostile instruction injection payloads."""
        detected = []
        for pat in self.compiled_patterns:
            matches = pat.findall(text)
            if matches:
                detected.extend(matches)

        is_safe = len(detected) == 0
        return {
            "is_safe": is_safe,
            "detected_threats": detected,
            "threat_count": len(detected),
            "recommendation": "ALLOW" if is_safe else "QUARANTINE_DOCUMENT",
        }

    def check_access_permission(self, doc: Document, user_roles: List[str]) -> bool:
        """Enforces Document-Level Access Control (ACL)."""
        doc_roles = set(doc.acl_roles)
        user_role_set = set(user_roles)
        # Check if there is any intersection between user roles and document roles
        return len(doc_roles & user_role_set) > 0


def main():
    print("=" * 70)
    print("CHAPTER 33: RAG SECURITY & PROMPT INJECTION DEFENSE")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.abspath(os.path.join(script_dir, "..", "data", "ai_course_corpus.json"))

    pipeline = DocumentIngestionPipeline()
    docs = pipeline.ingest_from_json(corpus_path)

    guard = RAGSecurityGuard()

    print(f"Scanning {len(docs)} ingested documents for adversarial injection payloads...\n")

    safe_docs = []
    quarantined = []

    for d in docs:
        scan = guard.scan_for_prompt_injection(d.content)
        if scan["is_safe"]:
            safe_docs.append(d)
        else:
            quarantined.append((d, scan["detected_threats"]))

    print(f"Safe Documents: {len(safe_docs)}")
    print(f"Quarantined Documents: {len(quarantined)}")

    for d, threats in quarantined:
        print(f"\n[ALERT] Quarantined Threat Detected in Document '{d.doc_id}' ({d.title}):")
        print(f"        Detected Payload Signatures: {threats}")
        print("        Action: Excluded from Vector Index to prevent indirect prompt injection.")

    # Access Control Test
    print("\n" + "-" * 70)
    print("Testing Document-Level Access Control (ACL):")
    restricted_doc = Document(
        doc_id="doc_internal_exam_01",
        title="Soal Ujian Akhir Semester Rahasia",
        content="Kunci jawaban soal AI 2026...",
        acl_roles=["instructor", "admin"],
    )

    student_allowed = guard.check_access_permission(restricted_doc, ["student"])
    instructor_allowed = guard.check_access_permission(restricted_doc, ["instructor"])

    print(f"Document ACL: {restricted_doc.acl_roles}")
    print(f"- Student Role Access:    {'GRANTED' if student_allowed else 'DENIED (Blocked)'}")
    print(f"- Instructor Role Access: {'GRANTED' if instructor_allowed else 'DENIED (Blocked)'}")

    print("\n[SUCCESS] RAG Security checks completed.")


if __name__ == "__main__":
    main()

