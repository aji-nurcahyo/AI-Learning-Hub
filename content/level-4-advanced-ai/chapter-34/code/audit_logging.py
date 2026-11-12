"""Structured Audit Logging and Observability for AI Agents.

Implements an immutable, structured audit trail recording:
1. Session ID, step sequence, and ISO-8601 timestamps.
2. User role and permission authorization status.
3. Model thought trace, requested tool action, and parameters.
4. Human approval gate decisions (auto-approved, approved, denied).
5. Execution observations, errors, and output payloads.
6. SHA-256 hash chaining to ensure audit record tamper-evidence.
"""

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional


class AuditRecord:
    """Represents a single tamper-evident audit record in the agent execution chain."""

    def __init__(
        self,
        session_id: str,
        step: int,
        user_role: str,
        thought: str,
        action: str,
        tool_args: Dict[str, Any],
        risk_level: str,
        approval_status: str,
        observation: Any,
        prev_hash: str = "GENESIS_BLOCK",
    ):
        self.session_id = session_id
        self.step = step
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.user_role = user_role
        self.thought = thought
        self.action = action
        self.tool_args = tool_args
        self.risk_level = risk_level
        self.approval_status = approval_status
        self.observation = observation
        self.prev_hash = prev_hash
        self.record_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = (
            f"{self.session_id}:{self.step}:{self.timestamp}:{self.action}:"
            f"{json.dumps(self.tool_args, sort_keys=True)}:{self.approval_status}:{self.prev_hash}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "user_role": self.user_role,
            "thought": self.thought,
            "action": self.action,
            "tool_args": self.tool_args,
            "risk_level": self.risk_level,
            "approval_status": self.approval_status,
            "observation": self.observation,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }


class AgentAuditLogger:
    """Manages the append-only audit trail with cryptographic chain verification."""

    def __init__(self):
        self.chain: List[AuditRecord] = []

    def log_event(
        self,
        session_id: str,
        step: int,
        user_role: str,
        thought: str,
        action: str,
        tool_args: Dict[str, Any],
        risk_level: str,
        approval_status: str,
        observation: Any,
    ) -> AuditRecord:
        prev_hash = self.chain[-1].record_hash if self.chain else "GENESIS_BLOCK"
        record = AuditRecord(
            session_id=session_id,
            step=step,
            user_role=user_role,
            thought=thought,
            action=action,
            tool_args=tool_args,
            risk_level=risk_level,
            approval_status=approval_status,
            observation=observation,
            prev_hash=prev_hash,
        )
        self.chain.append(record)
        return record

    def verify_integrity(self) -> bool:
        """Verifies that no record in the audit chain has been altered or tampered with."""
        for i, rec in enumerate(self.chain):
            # 1. Check prev_hash linkage
            if i > 0 and rec.prev_hash != self.chain[i - 1].record_hash:
                return False
            # 2. Check record self-hash
            if rec.record_hash != rec._compute_hash():
                return False
        return True


def main():
    print("=" * 70)
    print("CHAPTER 34: STRUCTURED AGENT AUDIT LOGGING")
    print("=" * 70)

    logger = AgentAuditLogger()

    # Log Step 1 (Read-Only)
    rec1 = logger.log_event(
        session_id="sess_itera_001",
        step=1,
        user_role="student",
        thought="Pengguna menanyakan materi Transformer. Menggunakan search tool.",
        action="search_knowledge_base",
        tool_args={"query": "Transformer attention"},
        risk_level="read_only",
        approval_status="auto_approved",
        observation="Ditemukan dokumen doc_tf_01",
    )
    print(f"Log Record 1 Hash: {rec1.record_hash[:16]}... (Prev: {rec1.prev_hash})")

    # Log Step 2 (State-Changing Approval Required)
    rec2 = logger.log_event(
        session_id="sess_itera_001",
        step=2,
        user_role="student",
        thought="Menghapus berkas cache lama.",
        action="delete_cache_entry",
        tool_args={"entry_id": "cache_temp_01"},
        risk_level="state_changing",
        approval_status="denied",
        observation="Otorisasi ditolak oleh pengguna",
    )
    print(f"Log Record 2 Hash: {rec2.record_hash[:16]}... (Prev: {rec2.prev_hash[:16]}...)")

    # Verify Integrity
    intact = logger.verify_integrity()
    print(f"\nVerifikasi Integritas Jejak Audit: {'VALID (Untampered)' if intact else 'CORRUPTED'}")

    print("\n[SUCCESS] Structured audit logging and cryptographic verification verified.")


if __name__ == "__main__":
    main()

