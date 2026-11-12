"""Central Configuration for Chapter 40 Capstone Reference Implementation."""

from dataclasses import dataclass
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class CapstoneConfig:
    project_name: str = "Intelligent Sensor Monitoring (CPU Reference)"
    seed: int = 42
    n_samples: int = 1500
    val_size: float = 0.20
    test_size: float = 0.20
    confidence_escalation_threshold: float = 0.65
    psi_alert_threshold: float = 0.25
    data_dir: str = os.path.join(BASE_DIR, "data")
    exp_dir: str = os.path.join(BASE_DIR, "experiments")
    feature_names: tuple = (
        "temperature_c",
        "vibration_rms",
        "current_amp",
        "acoustic_db",
        "rpm_deviation",
        "voltage_ripple_v",
    )
    feature_bounds: dict = None  # Populated via method

    def get_bounds(self) -> dict:
        return {
            "temperature_c": (0.0, 140.0),
            "vibration_rms": (0.0, 25.0),
            "current_amp": (0.0, 80.0),
            "acoustic_db": (20.0, 130.0),
            "rpm_deviation": (-500.0, 500.0),
            "voltage_ripple_v": (0.0, 30.0),
        }

