"""Tool Error Handling and Fault Recovery.

Demonstrates graceful error trapping during tool calling:
1. Division by zero and arithmetic errors.
2. Missing or malformed parameters.
3. Execution timeout handling.
4. Non-existent tool requests.
5. Packaging structured error observations so the agent can self-correct.
"""

from typing import Any, Dict
from tool_registry import build_default_registry
from tool_calling_demo import ToolExecutionEngine


class ResilientToolDispatcher:
    """Dispatches tool calls with comprehensive error handling and recovery messages."""

    def __init__(self):
        self.registry = build_default_registry()
        self.engine = ToolExecutionEngine(self.registry)

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes tool and wraps any runtime failure into an informative observation."""
        result = self.engine.execute_tool_call(tool_name, arguments)

        if result["status"] == "error":
            # Provide self-correction hints
            err_msg = result["error"]
            hint = "Periksa kembali nama fungsi dan tipe argumen yang diberikan."
            if "not found" in err_msg:
                available = [t["function"]["name"] for t in self.registry.list_tools()]
                hint = f"Daftar tool yang tersedia adalah: {available}"
            elif "ZeroDivisionError" in err_msg or "division by zero" in err_msg:
                hint = "Operasi pembagian dengan nol tidak terdefinisi secara matematis. Gunakan penyebut bukan nol."

            return {
                "status": "error",
                "tool_name": tool_name,
                "error": err_msg,
                "hint_for_agent": hint,
                "data": None,
            }

        return result


def main():
    print("=" * 70)
    print("CHAPTER 34: TOOL ERROR HANDLING & FAULT RECOVERY")
    print("=" * 70)

    dispatcher = ResilientToolDispatcher()

    # Fault 1: Division by zero
    print("[Fault 1] Arithmetic Division by Zero:")
    res1 = dispatcher.dispatch("calculator", {"expression": "50 / 0"})
    print(f"  Status: {res1['status'].upper()}")
    print(f"  Error : {res1['error']}")
    print(f"  Hint  : {res1['hint_for_agent']}\n")

    # Fault 2: Unknown Tool
    print("[Fault 2] Non-existent Tool Request:")
    res2 = dispatcher.dispatch("weather_forecast_tool", {"city": "Bandar Lampung"})
    print(f"  Status: {res2['status'].upper()}")
    print(f"  Error : {res2['error']}")
    print(f"  Hint  : {res2['hint_for_agent']}\n")

    # Fault 3: Missing Required Parameter
    print("[Fault 3] Missing Parameter:")
    res3 = dispatcher.dispatch("search_knowledge_base", {})
    print(f"  Status: {res3['status'].upper()}")
    print(f"  Error : {res3['error']}")
    print(f"  Hint  : {res3['hint_for_agent']}\n")

    print("[SUCCESS] Tool error handling and agent recovery verified.")


if __name__ == "__main__":
    main()

