"""Defensive Input Schema & Boundary Validation for ML Features and Text Payloads."""

import math
from typing import Dict, Any, Tuple


def validate_tabular_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    required = ("sensor_temp_c", "vibration_rms", "device_id")
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object."
    for key in required:
        if key not in payload:
            return False, f"Missing required field: '{key}'"

    # Reject booleans masquerading as ints/floats in Python (isinstance(True, int) == True!)
    for num_key, (lo, hi) in [("sensor_temp_c", (-20.0, 150.0)), ("vibration_rms", (0.0, 50.0))]:
        val = payload[num_key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return False, f"Invalid numeric type for '{num_key}'"
        if math.isnan(val) or math.isinf(val):
            return False, f"Non-finite value (NaN/Inf) rejected for '{num_key}'"
        if not (lo <= float(val) <= hi):
            return False, f"Out-of-bounds '{num_key}': {val} not in [{lo}, {hi}]"

    dev = payload["device_id"]
    if not isinstance(dev, str) or not (3 <= len(dev) <= 32) or not dev.replace("-", "").isalnum():
        return False, "Invalid device_id format (allowlist alphanumeric/hyphen only)"

    return True, "VALID"


def main():
    print("=" * 75)
    print("CHAPTER 39: DEFENSIVE INPUT SCHEMA & BOUNDARY VALIDATION")
    print("=" * 75)

    test_cases = [
        ("Valid Sensor Reading", {"sensor_temp_c": 48.5, "vibration_rms": 1.2, "device_id": "MTR-01"}),
        ("Missing Field", {"sensor_temp_c": 48.5, "device_id": "MTR-01"}),
        ("Boolean Injection", {"sensor_temp_c": True, "vibration_rms": 1.2, "device_id": "MTR-01"}),
        ("NaN Poison Input", {"sensor_temp_c": float("nan"), "vibration_rms": 1.2, "device_id": "MTR-01"}),
        ("Out-of-Bounds Spike", {"sensor_temp_c": 9999.0, "vibration_rms": 1.2, "device_id": "MTR-01"}),
        ("Malformed ID / Traversal", {"sensor_temp_c": 45.0, "vibration_rms": 1.0, "device_id": "../etc/passwd"}),
    ]

    for label, payload in test_cases:
        ok, reason = validate_tabular_payload(payload)
        status = "ALLOW" if ok else "BLOCK"
        print(f"- {label:25s} -> [{status}] ({reason})")
        if label == "Valid Sensor Reading":
            assert ok
        else:
            assert not ok

    print("\n[SUCCESS] Defensive validation blocked 5/5 malformed payloads cleanly.")


if __name__ == "__main__":
    main()

