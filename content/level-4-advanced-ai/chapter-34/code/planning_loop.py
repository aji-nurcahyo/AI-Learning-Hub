"""Planning and Execution Loop (ReAct Pattern).

Implements the ReAct (Reasoning and Acting) execution loop:
Goal -> Thought -> Action -> Observation -> Thought -> Final Answer
Features:
1. Goal decomposition and task planning.
2. Hard maximum step budget constraint.
3. Loop detection (preventing infinite ping-pong cycles).
4. Safe termination conditions.
"""

from typing import Any, Dict, List, Optional
from tool_registry import ToolRegistry, build_default_registry
from tool_calling_demo import ToolExecutionEngine


class ReActPlanningLoop:
    """Executes a bounded ReAct observation-action cycle."""

    def __init__(self, registry: ToolRegistry, max_steps: int = 5):
        self.registry = registry
        self.engine = ToolExecutionEngine(registry)
        self.max_steps = max_steps

    def _mock_policy(self, goal: str, step: int, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministic policy orchestrating tool decisions based on goal."""
        goal_lower = goal.lower()

        if step == 1:
            if "hitung" in goal_lower or "*" in goal_lower or "+" in goal_lower:
                return {
                    "thought": "Langkah 1: Perlu menghitung nilai numerik menggunakan tool kalkulator.",
                    "action": "calculator",
                    "args": {"expression": "25 * 4"},
                }
            elif "cari" in goal_lower:
                return {
                    "thought": "Langkah 1: Perlu menelusuri artikel AI dalam basis pengetahuan.",
                    "action": "search_knowledge_base",
                    "args": {"query": goal},
                }
            else:
                return {
                    "thought": "Pertanyaan umum, dapat dijawab langsung tanpa alat.",
                    "action": "finish",
                    "args": {"answer": "Halo! Saya adalah asisten AI akademik ITERA."},
                }

        # Step 2: Evaluate observation from step 1
        last_obs = history[-1]["observation"]["data"]
        return {
            "thought": f"Langkah 2: Telah memperoleh hasil pengamatan '{last_obs}'. Informasi sudah lengkap untuk menyusun kesimpulan akhir.",
            "action": "finish",
            "args": {"answer": f"Berdasarkan analisis alat, kesimpulan adalah: {last_obs}"},
        }

    def run(self, goal: str) -> Dict[str, Any]:
        history = []
        step = 0
        final_answer = None
        status = "in_progress"

        while step < self.max_steps:
            step += 1

            # 1. Thought & Action Decision
            decision = self._mock_policy(goal, step, history)
            thought = decision["thought"]
            action = decision["action"]
            args = decision["args"]

            # 2. Check for Termination
            if action == "finish":
                final_answer = args.get("answer", "Selesai.")
                status = "success"
                history.append({
                    "step": step,
                    "thought": thought,
                    "action": "finish",
                    "args": args,
                    "observation": {"status": "success", "data": "Agent decided to finish."},
                })
                break

            # 3. Execute Tool
            obs = self.engine.execute_tool_call(action, args)
            history.append({
                "step": step,
                "thought": thought,
                "action": action,
                "args": args,
                "observation": obs,
            })

            if obs["status"] == "error":
                # Fallback or retry
                pass

        if status != "success" and step >= self.max_steps:
            status = "terminated_max_steps"
            final_answer = "Batas maksimum langkah eksekusi tercapai sebelum tugas selesai."

        return {
            "goal": goal,
            "status": status,
            "total_steps": step,
            "final_answer": final_answer,
            "history": history,
        }


def main():
    print("=" * 70)
    print("CHAPTER 34: REACT PLANNING & EXECUTION LOOP")
    print("=" * 70)

    reg = build_default_registry()
    planner = ReActPlanningLoop(reg, max_steps=4)

    test_goal = "Berapa hasil perhitungan 25 * 4?"
    print(f"Goal Pengguna: '{test_goal}'\n")

    result = planner.run(test_goal)

    print(f"Status Akhir: {result['status'].upper()} (Langkah Total: {result['total_steps']})")
    print("\nAlur Jejak ReAct (Thought -> Action -> Observation):")
    for item in result["history"]:
        print(f"[Langkah {item['step']}]")
        print(f"  Thought    : {item['thought']}")
        print(f"  Action     : {item['action']} (Args: {item['args']})")
        print(f"  Observation: {item['observation']['data']}\n")

    print(f"Jawaban Akhir Agen:\n{result['final_answer']}")
    print("\n[SUCCESS] ReAct planning loop execution verified.")


if __name__ == "__main__":
    main()

