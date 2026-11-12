"""Reference Implementation: Supervised AI Learning Assistant with Read-Only Tools.

Core Project Deliverable for Chapter 34:
1. Question input processing.
2. Tool registry with strict schema validation.
3. Read-only search tool (retrieving course materials).
4. Local calculator tool (arithmetic evaluation).
5. State-changing cache deletion tool (requiring human approval).
6. Deterministic routing policy (educational mock policy).
7. Working memory and step budget limits (max 5 steps).
8. Structured cryptographic audit logging.
9. Permission and human approval gating with denial handling.
10. Safe, graceful termination.
"""

from typing import Any, Callable, Dict, List, Optional
from tool_registry import ToolRegistry, build_default_registry
from tool_calling_demo import ToolExecutionEngine
from agent_state import AgentStateMachine, AgentStatus
from human_approval_gate import HumanApprovalGate, ApprovalStatus
from permission_manager import PermissionManager, UserRole
from audit_logging import AgentAuditLogger


class SupervisedAILearningAssistant:
    """Supervised AI Agent for academic student and instructor assistance."""

    def __init__(
        self,
        user_role: UserRole = UserRole.STUDENT,
        max_steps: int = 5,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.user_role = user_role
        self.max_steps = max_steps
        self.registry = build_default_registry()
        self.engine = ToolExecutionEngine(self.registry)
        self.permission_manager = PermissionManager(user_role)
        self.approval_gate = HumanApprovalGate(approval_callback)
        self.audit_logger = AgentAuditLogger()

    def _policy_decide(self, goal: str, step: int, last_obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministic educational routing policy."""
        goal_lower = goal.lower()

        # Step 1: Tool Selection
        if step == 1:
            if "hitung" in goal_lower or "*" in goal_lower or "+" in goal_lower or "/" in goal_lower:
                return {
                    "thought": "Tugas membutuhkan kalkulasi numerik. Memilih tool 'calculator'.",
                    "action": "calculator",
                    "args": {"expression": "25 * 4 + 10"},
                }
            elif "cari" in goal_lower or "materi" in goal_lower or "rag" in goal_lower:
                return {
                    "thought": "Tugas membutuhkan materi akademik. Memilih tool 'search_knowledge_base'.",
                    "action": "search_knowledge_base",
                    "args": {"query": goal},
                }
            elif "hapus" in goal_lower or "cache" in goal_lower:
                return {
                    "thought": "Tugas meminta penghapusan cache. Memilih tool state-changing 'delete_cache_entry'.",
                    "action": "delete_cache_entry",
                    "args": {"entry_id": "cache_temp_session"},
                }
            else:
                return {
                    "thought": "Tugas bersifat konseptual umum dan dapat dijawab langsung tanpa alat.",
                    "action": "finish",
                    "args": {"answer": "Halo! Saya adalah Asisten AI Akademik ITERA siap membantu belajar AI."},
                }

        # Step 2: Observation Evaluation
        if last_obs and last_obs.get("status") == "error":
            return {
                "thought": f"Terjadi kesalahan pada eksekusi alat ({last_obs.get('error')}). Menghentikan eksekusi secara aman.",
                "action": "finish",
                "args": {"answer": f"Operasi tidak dapat diselesaikan: {last_obs.get('error')}"},
            }

        data = last_obs.get("data") if last_obs else None
        return {
            "thought": f"Pengamatan berhasil diperoleh: {str(data)[:60]}... Informasi memadai untuk jawaban final.",
            "action": "finish",
            "args": {"answer": f"Jawaban tuntas berdasarkan pengamatan: {data}"},
        }

    def execute_task(self, session_id: str, goal: str) -> Dict[str, Any]:
        """Runs the supervised agent workflow under full governance controls."""
        state = AgentStateMachine(session_id=session_id, goal=goal, max_steps=self.max_steps)
        state.transition_to(AgentStatus.PLANNING, "Task initialized")

        last_obs = None
        final_answer = None

        while not state.is_terminated():
            state.increment_step()
            step_idx = state.current_step

            # 1. Thought & Action Decision
            decision = self._policy_decide(goal, step_idx, last_obs)
            thought = decision["thought"]
            action = decision["action"]
            args = decision["args"]

            # 2. Check for Completion
            if action == "finish":
                final_answer = args.get("answer")
                state.transition_to(AgentStatus.TERMINATED_SUCCESS, "Goal achieved")
                self.audit_logger.log_event(
                    session_id=session_id,
                    step=step_idx,
                    user_role=self.user_role.value,
                    thought=thought,
                    action="finish",
                    tool_args=args,
                    risk_level="none",
                    approval_status="auto_approved",
                    observation="Task finished",
                )
                break

            # 3. RBAC Permission Check
            perm = self.permission_manager.check_permission(action)
            if not perm["authorized"]:
                last_obs = {"status": "error", "error": perm["message"], "data": None}
                state.transition_to(AgentStatus.TERMINATED_FAILED, "Permission denied")
                self.audit_logger.log_event(
                    session_id=session_id,
                    step=step_idx,
                    user_role=self.user_role.value,
                    thought=thought,
                    action=action,
                    tool_args=args,
                    risk_level="unauthorized",
                    approval_status="denied",
                    observation=perm["message"],
                )
                final_answer = f"Aksi Ditolak: {perm['message']}"
                break

            # 4. Human Approval Gating
            tool_def = self.registry.get_tool(action)
            approval = self.approval_gate.check_and_request_approval(tool_def, args)

            if not approval["permitted"]:
                last_obs = {"status": "error", "error": approval["reason"], "data": None}
                self.audit_logger.log_event(
                    session_id=session_id,
                    step=step_idx,
                    user_role=self.user_role.value,
                    thought=thought,
                    action=action,
                    tool_args=args,
                    risk_level=tool_def.risk_level.value,
                    approval_status="denied",
                    observation=approval["reason"],
                )
                # Next step handles denial
                continue

            # 5. Tool Execution
            state.transition_to(AgentStatus.EXECUTING_TOOL, f"Invoking {action}")
            obs = self.engine.execute_tool_call(action, args)
            last_obs = obs

            self.audit_logger.log_event(
                session_id=session_id,
                step=step_idx,
                user_role=self.user_role.value,
                thought=thought,
                action=action,
                tool_args=args,
                risk_level=tool_def.risk_level.value,
                approval_status=approval["status"],
                observation=obs["data"] if obs["status"] == "success" else obs["error"],
            )

            # Return to planning
            if not state.is_terminated():
                state.transition_to(AgentStatus.PLANNING, "Processing observation")

        return {
            "session_id": session_id,
            "goal": goal,
            "user_role": self.user_role.value,
            "status": state.status.value,
            "total_steps": state.current_step,
            "final_answer": final_answer,
            "audit_chain_length": len(self.audit_logger.chain),
            "audit_intact": self.audit_logger.verify_integrity(),
        }


def main():
    print("=" * 70)
    print("CHAPTER 34: SUPERVISED AI LEARNING ASSISTANT (REFERENCE PROJECT)")
    print("=" * 70)

    # Scenario 1: Student invoking safe read-only search
    assistant = SupervisedAILearningAssistant(user_role=UserRole.STUDENT)
    res1 = assistant.execute_task("sess_01", "Cari materi pengantar RAG dalam sistem")
    print("\nSkenario 1 (Mahasiswa - Pencarian Materi):")
    print(f"Status: {res1['status'].upper()} | Langkah: {res1['total_steps']}")
    print(f"Jawaban:\n{res1['final_answer']}")
    print(f"Integritas Audit: {res1['audit_intact']}\n")

    # Scenario 2: Student attempting unauthorized state-changing tool (Cache Deletion)
    print("-" * 70)
    print("Skenario 2 (Mahasiswa - Upaya Penghapusan Cache Tanpa Izin):")
    res2 = assistant.execute_task("sess_02", "Hapus file cache sistem sekarang")
    print(f"Status: {res2['status'].upper()} | Langkah: {res2['total_steps']}")
    print(f"Jawaban:\n{res2['final_answer']}\n")

    # Scenario 3: Admin attempting state-changing tool with human approval
    print("-" * 70)
    print("Skenario 3 (Admin - Penghapusan Cache dengan Persetujuan Manusia):")
    admin_assistant = SupervisedAILearningAssistant(
        user_role=UserRole.ADMIN,
        approval_callback=lambda name, args: True,  # simulated human clicks APPROVE
    )
    res3 = admin_assistant.execute_task("sess_03", "Hapus file cache sistem sekarang")
    print(f"Status: {res3['status'].upper()} | Langkah: {res3['total_steps']}")
    print(f"Jawaban:\n{res3['final_answer']}")

    print("\n[SUCCESS] Supervised AI Learning Assistant operational with complete controls.")


if __name__ == "__main__":
    main()

