"""Agent State and Working Memory Management.

Manages agent lifecycle transitions:
INITIALIZED -> PLANNING -> AWAITING_APPROVAL -> EXECUTING_TOOL -> TERMINATED
Maintains working memory, step history, and execution budgets.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional


class AgentStatus(Enum):
    INITIALIZED = "initialized"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING_TOOL = "executing_tool"
    TERMINATED_SUCCESS = "terminated_success"
    TERMINATED_FAILED = "terminated_failed"
    TERMINATED_MAX_STEPS = "terminated_max_steps"


class AgentStateMachine:
    """Finite State Machine enforcing legal agent execution transitions."""

    # Allowed transitions: CurrentStatus -> Set of allowed NextStatus
    TRANSITIONS = {
        AgentStatus.INITIALIZED: {AgentStatus.PLANNING, AgentStatus.TERMINATED_FAILED},
        AgentStatus.PLANNING: {
            AgentStatus.EXECUTING_TOOL,
            AgentStatus.AWAITING_APPROVAL,
            AgentStatus.TERMINATED_SUCCESS,
            AgentStatus.TERMINATED_FAILED,
            AgentStatus.TERMINATED_MAX_STEPS,
        },
        AgentStatus.AWAITING_APPROVAL: {
            AgentStatus.EXECUTING_TOOL,
            AgentStatus.PLANNING,
            AgentStatus.TERMINATED_FAILED,
        },
        AgentStatus.EXECUTING_TOOL: {
            AgentStatus.PLANNING,
            AgentStatus.TERMINATED_FAILED,
        },
        AgentStatus.TERMINATED_SUCCESS: set(),
        AgentStatus.TERMINATED_FAILED: set(),
        AgentStatus.TERMINATED_MAX_STEPS: set(),
    }

    def __init__(self, session_id: str, goal: str, max_steps: int = 5):
        self.session_id = session_id
        self.goal = goal
        self.max_steps = max_steps
        self.current_step = 0
        self.status = AgentStatus.INITIALIZED
        self.history: List[Dict[str, Any]] = []
        self.context_memory: Dict[str, Any] = {}
        self.created_at = time.time()

    def transition_to(self, new_status: AgentStatus, reason: str = ""):
        allowed = self.TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Ilegal state transition: cannot transition from '{self.status.value}' to '{new_status.value}'"
            )
        self.status = new_status
        self.history.append({
            "step": self.current_step,
            "status": new_status.value,
            "reason": reason,
            "timestamp": time.time(),
        })

    def increment_step(self):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.transition_to(AgentStatus.TERMINATED_MAX_STEPS, "Exceeded maximum allowed steps budget")

    def is_terminated(self) -> bool:
        return self.status in {
            AgentStatus.TERMINATED_SUCCESS,
            AgentStatus.TERMINATED_FAILED,
            AgentStatus.TERMINATED_MAX_STEPS,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "status": self.status.value,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "history_length": len(self.history),
            "is_terminated": self.is_terminated(),
        }


def main():
    print("=" * 70)
    print("CHAPTER 34: AGENT STATE MACHINE & WORKING MEMORY")
    print("=" * 70)

    sm = AgentStateMachine(session_id="session_ai_001", goal="Cari definisi Transformer dan hitung 10 * 5", max_steps=4)
    print(f"State Awal: {sm.status.value.upper()}")

    # Transition 1: Start Planning
    sm.transition_to(AgentStatus.PLANNING, "Agent receives user goal")
    print(f"State Sekarang: {sm.status.value.upper()}")

    # Transition 2: Executing Tool
    sm.transition_to(AgentStatus.EXECUTING_TOOL, "Invoking search tool")
    print(f"State Sekarang: {sm.status.value.upper()}")

    # Transition 3: Back to Planning
    sm.transition_to(AgentStatus.PLANNING, "Processing observation")
    sm.increment_step()
    print(f"State Sekarang: {sm.status.value.upper()} (Step: {sm.current_step})")

    # Transition 4: Terminate Success
    sm.transition_to(AgentStatus.TERMINATED_SUCCESS, "Goal achieved")
    print(f"State Akhir: {sm.status.value.upper()} (Is Terminated: {sm.is_terminated()})")

    # Verify Illegal Transition Prevention
    try:
        sm.transition_to(AgentStatus.EXECUTING_TOOL, "Illegal attempt after termination")
    except ValueError as e:
        print(f"\n[Protected] Mencegah transisi ilegal: {e}")

    print("\n[SUCCESS] Agent state machine verified.")


if __name__ == "__main__":
    main()

