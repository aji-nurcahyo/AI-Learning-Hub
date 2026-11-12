"""Local Validated Inference Service with Human Escalation Fallback (Offline CPU)."""

import math
from typing import Dict, Any, Tuple
import numpy as np

from config import CapstoneConfig
from evaluation_and_error_analysis import load_verified_bundle


def validate_sensor_request(payload: Dict[str, Any], cfg: CapstoneConfig = CapstoneConfig()) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Payload must be a dictionary."
    bounds = cfg.get_bounds()
    for feat in cfg.feature_names:
        if feat not in payload:
            return False, f"Missing sensor feature '{feat}'."
        val = payload[feat]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return False, f"Invalid type for '{feat}' (booleans/strings rejected)."
        if math.isnan(val) or math.isinf(val):
            return False, f"Non-finite value (NaN/Inf) rejected for '{feat}'."
        lo, hi = bounds[feat]
        if not (lo <= float(val) <= hi):
            return False, f"Out-of-bounds reading for '{feat}': {val} outside [{lo}, {hi}]."
    return True, "VALID"


def predict_sensor_reading(payload: Dict[str, Any], bundle: dict = None, cfg: CapstoneConfig = CapstoneConfig()) -> dict:
    ok, msg = validate_sensor_request(payload, cfg)
    if not ok:
        return {"status": "REJECTED_INVALID_INPUT", "error": msg, "human_escalation": True}

    if bundle is None:
        bundle = load_verified_bundle(cfg)

    vec = np.array([[float(payload[f]) for f in cfg.feature_names]], dtype=np.float64)
    scaled = bundle["scaler"].transform(vec)
    probs = bundle["model"].predict_proba(scaled)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    escalate = confidence < cfg.confidence_escalation_threshold

    return {
        "status": "OK",
        "model_name": bundle["model_name"],
        "predicted_class": pred_class,
        "label": "WARNING_FAULT" if pred_class == 1 else "NORMAL",
        "fault_probability": round(float(probs[1]), 4),
        "confidence": round(confidence, 4),
        "human_escalation": escalate,
        "recommended_action": "Manual engineer inspection required" if (pred_class == 1 or escalate) else "Continue nominal telemetry logging",
    }


def main():
    print("=" * 78)
    print("CHAPTER 40 CAPSTONE: VALIDATED LOCAL INFERENCE & FALLBACK")
    print("=" * 78)
    bundle = load_verified_bundle()

    samples = [
        ("Nominal Motor", {"temperature_c": 54.0, "vibration_rms": 1.6, "current_amp": 13.5, "acoustic_db": 61.0, "rpm_deviation": 5.0, "voltage_ripple_v": 1.1}),
        ("Overheating & High Vib", {"temperature_c": 78.0, "vibration_rms": 4.2, "current_amp": 21.0, "acoustic_db": 79.0, "rpm_deviation": -65.0, "voltage_ripple_v": 2.4}),
        ("Corrupted NaN Telemetry", {"temperature_c": float("nan"), "vibration_rms": 1.6, "current_amp": 13.5, "acoustic_db": 61.0, "rpm_deviation": 5.0, "voltage_ripple_v": 1.1}),
    ]
    for title, payload in samples:
        out = predict_sensor_reading(payload, bundle=bundle)
        print(f"- {title:24s} -> {out}")


if __name__ == "__main__":
    main()
