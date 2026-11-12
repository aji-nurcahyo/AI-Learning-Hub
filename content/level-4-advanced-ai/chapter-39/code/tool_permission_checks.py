"""Tool Permission Checks, Allowlisting & Human-in-the-Loop Approval Gate."""

import hashlib
import json
from typing import Dict, Any, Tuple


ALLOWED_TOOLS = {
    "read_sensor_summary": {"risk": "READ_ONLY", "roles": {"viewer", "analyst", "admin"}, "requires_approval": False},
    "export_report": {"risk": "READ_ONLY", "roles": {"analyst", "admin"}, "requires_approval": False},
    "update_threshold": {"risk": "STATE_CHANGING", "roles": {"admin"}, "requires_approval": True},
}


def authorize_tool_call(
    role: str,
    tool_name: str,
    args: Dict[str, Any],
    human_approved: bool = False,
) -> Tuple[bool, str, str]:
    if tool_name not in ALLOWED_TOOLS:
        reason = f"DENIED: Unregistered tool '{tool_name}' blocked by allowlist."
        entry_hash = hashlib.sha256(f"{role}|{tool_name}|{reason}".encode()).hexdigest()[:16]
        return False, reason, entry_hash

    policy = ALLOWED_TOOLS[tool_name]
    if role not in policy["roles"]:
        reason = f"DENIED: Role '{role}' lacks permission for '{tool_name}'."
        entry_hash = hashlib.sha256(f"{role}|{tool_name}|{reason}".encode()).hexdigest()[:16]
        return False, reason, entry_hash

    if policy["requires_approval"] and not human_approved:
        reason = f"BLOCKED: State-changing tool '{tool_name}' requires explicit Human Approval Gate."
        entry_hash = hashlib.sha256(f"{role}|{tool_name}|{reason}".encode()).hexdigest()[:16]
        return False, reason, entry_hash

    reason = f"APPROVED: Executed '{tool_name}' with args={json.dumps(args)}"
    entry_hash = hashlib.sha256(f"{role}|{tool_name}|{reason}".encode()).hexdigest()[:16]
    return True, reason, entry_hash


def main():
    print("=" * 78)
    print("CHAPTER 39: TOOL PERMISSIONS, ALLOWLIST & HUMAN APPROVAL GATE")
    print("=" * 78)

    scenarios = [
        ("viewer", "read_sensor_summary", {"device_id": "MTR-01"}, False, True),
        ("viewer", "export_report", {"fmt": "json"}, False, False),
        ("analyst", "delete_all_models", {}, False, False),
        ("admin", "update_threshold", {"new_limit": 80.0}, False, False),
        ("admin", "update_threshold", {"new_limit": 80.0}, True, True),
    ]

    for role, tool, args, approved, expected in scenarios:
        ok, msg, audit_id = authorize_tool_call(role, tool, args, human_approved=approved)
        print(f"- Role={role:7s} | Tool={tool:20s} | Approval={str(approved):5s} -> [audit:{audit_id}] {msg}")
        assert ok is expected

    print("\n[SUCCESS] Least-privilege tool permissions and Human Approval Gate verified.")


if __name__ == "__main__":
    main()

