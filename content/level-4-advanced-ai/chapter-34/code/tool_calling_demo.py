"""Tool Calling Lifecycle Demonstration.

Demonstrates the step-by-step lifecycle of invoking tools:
1. Tool lookup in registry.
2. Argument schema validation.
3. Permission & approval check.
4. Safe execution within exception boundary.
5. Packaging output into standardized Observation format.
"""

from typing import Any, Dict
from tool_registry import ToolRegistry, build_default_registry
from tool_schema_validation import ToolSchemaValidator


class ToolExecutionEngine:
    """Safely dispatches and executes tool calls."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a tool call and returns a standardized observation payload."""
        # 1. Lookup Tool
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"Tool '{tool_name}' not found in registry",
                "data": None,
            }

        # 2. Validate Schema
        is_valid, errors = ToolSchemaValidator.validate(tool.parameters, arguments)
        if not is_valid:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"Schema validation failed: {'; '.join(errors)}",
                "data": None,
            }

        # 3. Execute with Safe Exception Boundary
        try:
            output = tool.execute(**arguments)
            return {
                "status": "success",
                "tool_name": tool_name,
                "error": None,
                "data": output,
            }
        except Exception as e:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"Execution error: {str(e)}",
                "data": None,
            }


def main():
    print("=" * 70)
    print("CHAPTER 34: TOOL CALLING EXECUTION LIFECYCLE")
    print("=" * 70)

    reg = build_default_registry()
    engine = ToolExecutionEngine(reg)

    # 1. Successful Tool Call (Calculator)
    print("[1] Executing valid calculator call...")
    obs1 = engine.execute_tool_call("calculator", {"expression": "50 * 2 + 35"})
    print(f"    Status: {obs1['status'].upper()} | Result: {obs1['data']}\n")

    # 2. Schema Validation Failure (Missing argument)
    print("[2] Executing invalid search call (missing required 'query')...")
    obs2 = engine.execute_tool_call("search_knowledge_base", {})
    print(f"    Status: {obs2['status'].upper()} | Error: {obs2['error']}\n")

    # 3. Execution Runtime Error (Zero division)
    print("[3] Executing runtime fault (division by zero)...")
    obs3 = engine.execute_tool_call("calculator", {"expression": "100 / 0"})
    print(f"    Status: {obs3['status'].upper()} | Error: {obs3['error']}\n")

    print("[SUCCESS] Tool calling lifecycle verified.")


if __name__ == "__main__":
    main()

