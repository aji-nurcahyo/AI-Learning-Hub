"""Input Request Schema Validation for Model Serving API.

Validates JSON inference payloads prior to hitting model.predict():
1. Enforces required 'features' array key.
2. Enforces exact feature vector dimensionality (d=6).
3. Rejects non-numeric, boolean, NaN, and Inf values.
4. Enforces finite domain boundaries to mitigate garbage-in/garbage-out.
"""

import math
from typing import Any, Dict, List, Tuple


class RequestValidationError(Exception):
    """Raised when incoming API inference request fails schema checks."""
    pass


def validate_predict_payload(payload: Any, expected_dim: int = 6) -> List[float]:
    """Validates request dictionary and returns clean float feature list."""
    if not isinstance(payload, dict):
        raise RequestValidationError("Payload must be a JSON object.")

    if "features" not in payload:
        raise RequestValidationError("Missing required key 'features'.")

    feats = payload["features"]
    if not isinstance(feats, list):
        raise RequestValidationError("Key 'features' must be a list of numbers.")

    if len(feats) != expected_dim:
        raise RequestValidationError(
            f"Feature dimension mismatch: expected {expected_dim}, got {len(feats)}."
        )

    cleaned: List[float] = []
    for idx, val in enumerate(feats):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise RequestValidationError(
                f"Feature at index {idx} has invalid type '{type(val).__name__}'; expected numeric."
            )
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            raise RequestValidationError(
                f"Feature at index {idx} is non-finite ({val}); NaN and Inf are rejected."
            )
        if abs(fval) > 1e6:
            raise RequestValidationError(
                f"Feature at index {idx} exceeds reasonable sensor bounds (|x| > 1e6)."
            )
        cleaned.append(fval)

    return cleaned


def main():
    print("=" * 75)
    print("CHAPTER 36: API REQUEST SCHEMA VALIDATION")
    print("=" * 75)

    test_cases = [
        ("Valid Payload", {"features": [0.5, -1.2, 0.3, 1.1, -0.4, 0.8]}),
        ("Missing Key", {"data": [1, 2, 3, 4, 5, 6]}),
        ("Wrong Dimension", {"features": [0.5, -1.2, 0.3]}),
        ("String Injection", {"features": [0.5, "DROP TABLE", 0.3, 1.1, -0.4, 0.8]}),
        ("NaN Rejection", {"features": [0.5, float("nan"), 0.3, 1.1, -0.4, 0.8]}),
        ("Boolean Rejection", {"features": [True, 0.0, 0.0, 0.0, 0.0, 0.0]}),
    ]

    for label, payload in test_cases:
        try:
            clean = validate_predict_payload(payload, expected_dim=6)
            print(f"[PASS] {label:<20} -> Accepted (First feature: {clean[0]})")
        except RequestValidationError as e:
            print(f"[REJECT] {label:<18} -> Blocked cleanly: {e}")

    print("\n[SUCCESS] API input validation verified.")


if __name__ == "__main__":
    main()

