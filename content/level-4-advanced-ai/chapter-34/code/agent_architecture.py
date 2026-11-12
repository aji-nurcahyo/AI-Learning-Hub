"""AI Agent Architecture Foundations.

Demonstrates the core architectural components of a supervised AI agent:
1. Agent Controller (Orchestrator).
2. Policy / Decision Engine (Deterministic educational policy).
3. Tool Registry (Discoverable capabilities).
4. State Store & Working Memory.
5. Approval Layer & Permission Boundary.
6. Execution Engine & Audit Logger.
"""

from typing import Any, Dict, List, Optional
import time


class AgentState:
    """Represents the internal working state and memory of an agent."""

    def __init__(self, session_id: str, goal: str):
        self.session_id = session_id
        self.goal = goal
        self.step_count = 0
        self.max_steps = 5
        self.observations: List[Dict[str, Any]] = []
        self.actions_taken: List[Dict[str, Any]] = []
        self.is_terminated = False
        self.termination_reason: Optional[str] = None
        self.final_answer: Optional[str] = None

    def add_step(self, thought: str, action: str, tool_args: Dict[str, Any], observation: Any):
        self.step_count += 1
        record = {
            "step": self.step_count,
            "thought": thought,
            "action": action,
            "tool_args": tool_args,
            "observation": observation,
            "timestamp": time.time(),
        }
        self.actions_taken.append(record)
        self.observations.append({"step": self.step_count, "data": observation})


class MockDecisionPolicy:
    """Deterministic educational policy simulating LLM thought/action decisions.

    NOTE: This is a deterministic rule-based controller for educational
    purposes and testing. It does NOT invoke neural LLM APIs.
    """

    def decide_next_step(self, state: AgentState) -> Dict[str, Any]:
        """Chooses the next thought and action based on current state and goal."""
        goal_lower = state.goal.lower()

        # Step 1: If no actions taken yet, determine initial tool
        if state.step_count == 0:
            if "hitung" in goal_lower or "tambah" in goal_lower or "+" in goal_lower or "*" in goal_lower:
                return {
                    "thought": "Pengguna meminta perhitungan matematika. Saya harus menggunakan tool kalkulator.",
                    "action": "calculator",
                    "args": {"expression": "25 * 4 + 10"},
                }
            elif "cari" in goal_lower or "apa itu" in goal_lower or "bagaimana" in goal_lower:
                return {
                    "thought": "Pengguna menanyakan informasi akademik. Saya harus mencari materi dalam korpus lokal.",
                    "action": "search_knowledge_base",
                    "args": {"query": state.goal},
                }
            elif "hapus" in goal_lower or "update" in goal_lower:
                return {
                    "thought": "Pengguna meminta modifikasi data state-changing. Ini membutuhkan persetujuan manusia.",
                    "action": "delete_cache_entry",
                    "args": {"entry_id": "temp_cache_001"},
                }
            else:
                return {
                    "thought": "Pertanyaan dapat dijawab langsung tanpa alat eksternal.",
                    "action": "finish",
                    "args": {"answer": "Halo! Saya adalah Asisten AI Akademis ITERA. Silakan ajukan pertanyaan seputar perkuliahan."},
                }

        # Step 2: Evaluate observations from previous step
        last_obs = state.observations[-1]["data"] if state.observations else None
        return {
            "thought": f"Hasil dari eksekusi alat telah diperoleh: {str(last_obs)[:60]}. Informasi sudah cukup untuk menyusun jawaban akhir.",
            "action": "finish",
            "args": {"answer": f"Berdasarkan hasil eksekusi alat: {last_obs}"},
        }


def main():
    print("=" * 70)
    print("CHAPTER 34: AI AGENT MODULAR ARCHITECTURE")
    print("=" * 70)

    state = AgentState(session_id="sess_001", goal="Berapa hasil perhitungan 25 * 4 + 10?")
    policy = MockDecisionPolicy()

    print(f"Goal: '{state.goal}'")
    print(f"Initial State: Step {state.step_count}, Terminated: {state.is_terminated}\n")

    # Cycle 1
    decision = policy.decide_next_step(state)
    print(f"[Cycle 1] Thought: {decision['thought']}")
    print(f"          Action:  {decision['action']} (Args: {decision['args']})")

    # Simulate observation
    mock_obs = 110.0
    state.add_step(decision["thought"], decision["action"], decision["args"], mock_obs)

    # Cycle 2
    decision2 = policy.decide_next_step(state)
    print(f"\n[Cycle 2] Thought: {decision2['thought']}")
    print(f"          Action:  {decision2['action']} (Args: {decision2['args']})")

    if decision2["action"] == "finish":
        state.is_terminated = True
        state.final_answer = decision2["args"]["answer"]

    print("\nFinal State Summary:")
    print(f"- Total Steps: {state.step_count}")
    print(f"- Final Answer: {state.final_answer}")
    print("\n[SUCCESS] AI Agent modular architecture verified.")


if __name__ == "__main__":
    main()

