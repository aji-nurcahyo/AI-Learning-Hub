"""Reproducible Unit & Correctness Test Suite for Chapter 40 Capstone Project."""

import unittest
import numpy as np

from config import CapstoneConfig
from preprocessing import prepare_splits
from local_inference import validate_sensor_request, predict_sensor_reading
from monitoring_and_drift import compute_psi
from responsible_ai_assessment import build_capstone_risk_register


class TestCapstoneReferenceSystem(unittest.TestCase):
    def test_zero_leakage_and_deterministic_splits(self):
        cfg = CapstoneConfig()
        d1 = prepare_splits(cfg)
        d2 = prepare_splits(cfg)
        self.assertTrue(np.array_equal(d1.X_train, d2.X_train))
        self.assertEqual(len(d1.y_train) + len(d1.y_val) + len(d1.y_test), cfg.n_samples)
        self.assertAlmostEqual(float(d1.X_train.mean()), 0.0, places=6)

    def test_input_validation_blocks_malformed_payloads(self):
        bad_nan = {"temperature_c": float("nan"), "vibration_rms": 1.0, "current_amp": 10.0, "acoustic_db": 60.0, "rpm_deviation": 0.0, "voltage_ripple_v": 1.0}
        bad_bool = {"temperature_c": True, "vibration_rms": 1.0, "current_amp": 10.0, "acoustic_db": 60.0, "rpm_deviation": 0.0, "voltage_ripple_v": 1.0}
        ok1, _ = validate_sensor_request(bad_nan)
        ok2, _ = validate_sensor_request(bad_bool)
        self.assertFalse(ok1)
        self.assertFalse(ok2)
        res = predict_sensor_reading(bad_nan)
        self.assertEqual(res["status"], "REJECTED_INVALID_INPUT")
        self.assertTrue(res["human_escalation"])

    def test_psi_drift_detection_sensitivity(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(55.0, 5.0, size=500)
        same = rng.normal(55.0, 5.0, size=500)
        shifted = rng.normal(67.0, 5.0, size=500)
        self.assertLess(compute_psi(ref, same), 0.15)
        self.assertGreater(compute_psi(ref, shifted), 0.25)

    def test_risk_register_completeness(self):
        reg = build_capstone_risk_register()
        self.assertGreaterEqual(len(reg), 4)
        for item in reg:
            self.assertLessEqual(item["residual_score"], item["likelihood"] * item["severity"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

