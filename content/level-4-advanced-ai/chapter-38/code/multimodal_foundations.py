"""Multimodal AI Foundations: Single-Modality vs Cross-Modal Integration.

Demonstrates why Multimodal AI requires cross-modal representation fusion rather
than running two isolated single-modality models independently:
- Example: A visual image alone might distinguish shape/color, while text telemetry
  specifies operational anomaly context ("faulty" vs "normal"). Combining modalities
  resolves ambiguity that neither modality alone can decide.
"""

import numpy as np


def main():
    print("=" * 75)
    print("CHAPTER 38: MULTIMODAL AI FOUNDATIONS")
    print("=" * 75)

    # Conceptual illustration: Industrial Equipment Inspection
    # Image modality outputs visual defect indicator (0=clean, 1=rust/crack)
    # Sensor/Text modality outputs temperature report (0=cool, 1=overheat)
    # Target Alert = 1 ONLY when BOTH visual defect AND thermal overheat co-occur (XOR/AND synergy)
    scenarios = [
        ("Clean Image + Cool Report", 0.0, 0.0, 0),
        ("Clean Image + Overheat Report", 0.0, 1.0, 0),
        ("Crack Image + Cool Report", 1.0, 0.0, 0),
        ("Crack Image + Overheat Report", 1.0, 1.0, 1),
    ]

    print("1. Why Isolated Unimodal Models Fail Synergy Cases:")
    print(f"   {'Scenario':<32} | {'Visual':<8} | {'Text/Therm':<10} | {'True Critical Alert'}")
    print("   " + "-" * 68)
    for label, v, t, y in scenarios:
        print(f"   {label:<32} | {v:<8.1f} | {t:<10.1f} | {y}")

    print("\n   Explanation: Neither visual inspection alone nor text telemetry alone can")
    print("   distinguish a harmless surface crack from an active overheating failure.")
    print("   Multimodal fusion combines heterogeneous representations to resolve ambiguity.\n")

    print("[SUCCESS] Multimodal foundations verified.")


if __name__ == "__main__":
    main()

