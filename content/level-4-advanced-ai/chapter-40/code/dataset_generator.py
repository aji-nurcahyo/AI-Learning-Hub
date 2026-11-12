"""Synthetic Sensor Telemetry Generator & Data Dictionary Creator."""

import json
import os
from typing import Tuple
import numpy as np

from config import CapstoneConfig


DATA_DICTIONARY = {
    "dataset_name": "Synthetic Industrial Motor Telemetry v1 (Chapter 40 Capstone)",
    "provenance": "100% synthetically generated via deterministic Gaussian-linear physical proxies (seed=42); contains zero proprietary or personal data.",
    "license": "CC-BY-4.0",
    "target": {"name": "fault_warning", "classes": {"0": "Normal Operating State", "1": "Warning / Anomaly"}},
    "features": [
        {"name": "temperature_c", "unit": "Celsius", "nominal_mean": 55.0, "valid_range": [0.0, 140.0]},
        {"name": "vibration_rms", "unit": "mm/s", "nominal_mean": 1.8, "valid_range": [0.0, 25.0]},
        {"name": "current_amp", "unit": "Amperes", "nominal_mean": 14.0, "valid_range": [0.0, 80.0]},
        {"name": "acoustic_db", "unit": "dB", "nominal_mean": 62.0, "valid_range": [20.0, 130.0]},
        {"name": "rpm_deviation", "unit": "RPM", "nominal_mean": 0.0, "valid_range": [-500.0, 500.0]},
        {"name": "voltage_ripple_v", "unit": "Volts", "nominal_mean": 1.2, "valid_range": [0.0, 30.0]},
    ],
}


def generate_sensor_telemetry(
    n_samples: int = 1500,
    seed: int = 42,
    temp_shift: float = 0.0,
    vib_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates realistic synthetic sensor telemetry for predictive monitoring."""
    rng = np.random.default_rng(seed)

    # Latent wear/stress factor drives correlated telemetry
    stress = rng.normal(0.0, 1.0, size=n_samples)
    temp = 55.0 + 7.5 * stress + rng.normal(0.0, 3.0, size=n_samples) + temp_shift
    vib = np.clip(1.8 + 0.85 * np.maximum(stress, -1.0) + rng.normal(0.0, 0.45, size=n_samples) + vib_shift, 0.2, 20.0)
    curr = np.clip(14.0 + 2.2 * stress + rng.normal(0.0, 1.1, size=n_samples), 4.0, 60.0)
    acou = np.clip(62.0 + 5.0 * stress + 2.5 * (vib - 1.8) + rng.normal(0.0, 2.5, size=n_samples), 35.0, 120.0)
    rpm_dev = -18.0 * stress + rng.normal(0.0, 14.0, size=n_samples)
    v_rip = np.clip(1.2 + 0.4 * np.abs(stress) + rng.normal(0.0, 0.25, size=n_samples), 0.1, 15.0)

    # Non-linear fault score + controlled sensor noise
    fault_score = (
        0.55 * ((temp - 58.0) / 8.0)
        + 0.70 * ((vib - 2.1) / 0.8)
        + 0.35 * ((acou - 65.0) / 6.0)
        + 0.25 * np.maximum(v_rip - 1.6, 0.0)
        + rng.normal(0.0, 0.35, size=n_samples)
    )
    y = (fault_score > 0.45).astype(int)
    X = np.column_stack([temp, vib, curr, acou, rpm_dev, v_rip]).astype(np.float64)
    return X, y


def save_data_dictionary(cfg: CapstoneConfig) -> str:
    os.makedirs(cfg.data_dir, exist_ok=True)
    out_path = os.path.join(cfg.data_dir, "sensor_data_dictionary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(DATA_DICTIONARY, f, indent=2)
    return out_path


def main():
    cfg = CapstoneConfig()
    X, y = generate_sensor_telemetry(cfg.n_samples, cfg.seed)
    dict_path = save_data_dictionary(cfg)
    print("=" * 75)
    print("CHAPTER 40 CAPSTONE: DATASET & DATA DICTIONARY")
    print("=" * 75)
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Fault Positive Rate: {100.0 * y.mean():.2f}%")
    print(f"Saved Data Dictionary to: {dict_path}")


if __name__ == "__main__":
    main()

