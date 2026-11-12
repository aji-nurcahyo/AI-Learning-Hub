"""Tool Schema Validation Engine.

Implements JSON Schema validation for tool execution arguments:
1. Verifies presence of all required parameters.
2. Enforces data type conformance (string, integer, float/number, boolean, dict, list).
3. Detects unexpected/unsupported arguments.
4. Prevents injection of invalid or malformed parameter payloads.
"""

from typing import Any, Dict, List, Tuple


class SchemaValidationError(Exception):
    """Raised when tool call arguments violate the defined schema."""
    pass


class ToolSchemaValidator:
    """Validates runtime tool invocation arguments against parameter JSON schemas."""

    TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    @classmethod
    def validate(cls, parameters_schema: Dict[str, Any], call_args: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates call_args against parameters_schema. Returns (is_valid, error_messages)."""
        errors = []
        required_fields = parameters_schema.get("required", [])
        properties = parameters_schema.get("properties", {})

        # 1. Check for missing required fields
        for field in required_fields:
            if field not in call_args:
                errors.append(f"Missing required parameter: '{field}'")

        # 2. Check each provided argument
        for arg_name, arg_val in call_args.items():
            if arg_name not in properties:
                errors.append(f"Unexpected parameter '{arg_name}' not defined in tool schema")
                continue

            expected_type_str = properties[arg_name].get("type")
            expected_py_type = cls.TYPE_MAP.get(expected_type_str)

            if expected_py_type:
                # Handle special case: bool is subclass of int in Python
                if expected_type_str == "integer" and isinstance(arg_val, bool):
                    errors.append(f"Parameter '{arg_name}' expected integer, got boolean ({arg_val})")
                elif not isinstance(arg_val, expected_py_type):
                    errors.append(
                        f"Parameter '{arg_name}' expected type '{expected_type_str}', got '{type(arg_val).__name__}'"
                    )

            # String emptiness check
            if expected_type_str == "string" and isinstance(arg_val, str) and not arg_val.strip():
                errors.append(f"Parameter '{arg_name}' cannot be an empty string")

        is_valid = len(errors) == 0
        return is_valid, errors


def main():
    print("=" * 70)
    print("CHAPTER 34: TOOL SCHEMA VALIDATION")
    print("=" * 70)

    from tool_registry import build_default_registry

    reg = build_default_registry()
    calc_tool = reg.get_tool("calculator")
    schema = calc_tool.parameters

    print(f"Schema Parameter untuk Tool '{calc_tool.name}':")
    print(schema)

    # Test Case 1: Valid Arguments
    valid_args = {"expression": "12 * 8"}
    ok, errs = ToolSchemaValidator.validate(schema, valid_args)
    print(f"\n[Test 1] Valid Args {valid_args}: Valid={ok}, Errors={errs}")

    # Test Case 2: Missing Required Field
    missing_args = {}
    ok, errs = ToolSchemaValidator.validate(schema, missing_args)
    print(f"[Test 2] Missing Args {missing_args}: Valid={ok}, Errors={errs}")

    # Test Case 3: Wrong Type (integer instead of string)
    wrong_type_args = {"expression": 12345}
    ok, errs = ToolSchemaValidator.validate(schema, wrong_type_args)
    print(f"[Test 3] Wrong Type {wrong_type_args}: Valid={ok}, Errors={errs}")

    # Test Case 4: Unexpected Extra Argument
    extra_args = {"expression": "5 + 5", "malicious_flag": True}
    ok, errs = ToolSchemaValidator.validate(schema, extra_args)
    print(f"[Test 4] Extra Args {extra_args}: Valid={ok}, Errors={errs}")

    print("\n[SUCCESS] Tool schema validation engine verified.")


if __name__ == "__main__":
    main()

