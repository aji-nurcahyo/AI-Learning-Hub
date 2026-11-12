"""Human Approval Gate and Supervised Execution.

Implements human-in-the-loop oversight:
1. Classifies tools into READ_ONLY (auto-approved) vs STATE_CHANGING (requires human approval).
2. Suspends execution when a high-risk tool call is requested.
3. Supports interactive or programmatic callback for authorization (ALLOW / DENY).
4. Handles approval denial gracefully: the agent receives an authorization denial observation.
"""

from typing import Any, Callable, Dict, Optional
from tool_registry import ToolDefinition, ToolRegistry, ToolRiskLevel, build_default_registry


class ApprovalStatus:
    APPROVED = "approved"
    DENIED = "denied"
    AUTO_APPROVED = "auto_approved"


class HumanApprovalGate:
    """Gates tool execution based on human approval for state-changing actions."""

    def __init__(self, approval_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None):
        # approval_callback takes (tool_name, arguments) -> returns True (allow) or False (deny)
        self.approval_callback = approval_callback

    def check_and_request_approval(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates whether tool call can proceed or requires human confirmation."""
        # 1. Read-Only Tools are automatically permitted under least-privilege policy
        if tool.risk_level == ToolRiskLevel.READ_ONLY:
            return {
                "status": ApprovalStatus.AUTO_APPROVED,
                "permitted": True,
                "reason": "Tool tergolong READ_ONLY (aman, tanpa efek samping permanen).",
            }

        # 2. State-Changing Tools require explicit authorization
        print(f"\n[GERBANG PERSETUJUAN MANUSIA] Permintaan eksekusi terdeteksi:")
        print(f"  Tool Berisiko   : '{tool.name}' (STATE_CHANGING)")
        print(f"  Argumen Eksekusi: {arguments}")
        print("  Status          : Eksekusi DITANGGUHKAN menunggu keputusan manusia.")

        # If a callback is provided, invoke it; otherwise default to safe DENIAL
        is_approved = False
        if self.approval_callback:
            is_approved = self.approval_callback(tool.name, arguments)

        if is_approved:
            print("  Keputusan       : DISETUJUI (APPROVED) oleh supervisor manusia.\n")
            return {
                "status": ApprovalStatus.APPROVED,
                "permitted": True,
                "reason": "Eksekusi disetujui secara eksplisit oleh supervisor manusia.",
            }
        else:
            print("  Keputusan       : DITOLAK (DENIED) oleh supervisor manusia demi keamanan.\n")
            return {
                "status": ApprovalStatus.DENIED,
                "permitted": False,
                "reason": "Otorisasi ditolak oleh pengguna manusia. Operasi dibatalkan.",
            }


def main():
    print("=" * 70)
    print("CHAPTER 34: HUMAN APPROVAL GATE DEMONSTRATION")
    print("=" * 70)

    reg = build_default_registry()
    gate = HumanApprovalGate()

    # Case 1: Read-Only Tool (Calculator)
    tool_calc = reg.get_tool("calculator")
    res1 = gate.check_and_request_approval(tool_calc, {"expression": "25 * 4"})
    print(f"Case 1 (Read-Only): Permitted={res1['permitted']} | Status={res1['status']}")

    # Case 2: State-Changing Tool with Denial (Default Fail-Safe)
    tool_del = reg.get_tool("delete_cache_entry")
    res2 = gate.check_and_request_approval(tool_del, {"entry_id": "cache_99"})
    print(f"Case 2 (State-Changing, Default): Permitted={res2['permitted']} | Status={res2['status']}")

    # Case 3: State-Changing Tool with Explicit Approval Callback
    approving_gate = HumanApprovalGate(approval_callback=lambda name, args: True)
    res3 = approving_gate.check_and_request_approval(tool_del, {"entry_id": "cache_99"})
    print(f"Case 3 (State-Changing, Approved): Permitted={res3['permitted']} | Status={res3['status']}")

    print("\n[SUCCESS] Human approval gate behavior verified.")


if __name__ == "__main__":
    main()

