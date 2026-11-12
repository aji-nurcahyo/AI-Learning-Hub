"""Simulated Edge Hardware Constraint Assessment.

Evaluates conceptual deployment feasibility against representative embedded hardware:
1. ARM Cortex-M7 MCU (TinyML): 256 KB SRAM (Tensor Arena), 1 MB Flash.
2. ESP32-S3 Microcontroller: 512 KB Internal SRAM (+ optional PSRAM), 4 MB Flash.
3. Raspberry Pi 4 SBC (Embedded Linux): 4 GB RAM, 32 GB SD Card.

IMPORTANT: This script evaluates simulated weight + activation tensor budgets in Python.
It does NOT perform physical hardware-in-the-loop flashing or live MCU execution.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EdgeTargetDevice:
    name: str
    sram_kb: float
    flash_kb: float
    runtime_overhead_kb: float
    notes: str


def assess_device_fit(device: EdgeTargetDevice, raw_weights_kb: float, peak_activation_kb: float) -> str:
    total_ram_needed = peak_activation_kb + device.runtime_overhead_kb
    # Flash must store weights + runtime binary (~200 KB for TFLite Micro)
    total_flash_needed = raw_weights_kb + (200.0 if device.flash_kb < 10000 else 50000.0)

    ram_ok = total_ram_needed <= device.sram_kb
    flash_ok = total_flash_needed <= device.flash_kb

    if ram_ok and flash_ok:
        return f"SIMULATED FIT (RAM {total_ram_needed:.1f}/{device.sram_kb:.0f} KB, Flash {total_flash_needed:.1f}/{device.flash_kb:.0f} KB)"
    reasons = []
    if not ram_ok:
        reasons.append(f"RAM overflow ({total_ram_needed:.1f} > {device.sram_kb:.0f} KB)")
    if not flash_ok:
        reasons.append(f"Flash overflow ({total_flash_needed:.1f} > {device.flash_kb:.0f} KB)")
    return "DOES NOT FIT: " + "; ".join(reasons)


def main():
    print("=" * 75)
    print("CHAPTER 37: SIMULATED EDGE HARDWARE CONSTRAINT ASSESSMENT")
    print("=" * 75)

    devices: List[EdgeTargetDevice] = [
        EdgeTargetDevice("ARM Cortex-M7 (TinyML)", sram_kb=256.0, flash_kb=1024.0, runtime_overhead_kb=120.0, notes="TFLite Micro runtime ~120KB RAM"),
        EdgeTargetDevice("ESP32-S3 (No PSRAM)", sram_kb=512.0, flash_kb=4096.0, runtime_overhead_kb=220.0, notes="FreeRTOS + WiFi stack uses ~220KB"),
        EdgeTargetDevice("Raspberry Pi 4 (SBC)", sram_kb=4000000.0, flash_kb=32000000.0, runtime_overhead_kb=250000.0, notes="Full Linux + PyTorch runtime"),
    ]

    # Compare 3 conceptual deployment profiles (raw parameter memory + peak activation arena)
    models = [
        ("FP32 Teacher (45.9K params)", 179.5, 4.0),       # 179.5 KB weights + 4 KB arena
        ("INT8 Quantized Teacher", 45.9, 1.5),             # ~4x smaller raw weights
        ("Distilled Student (1.1K FP32)", 4.5, 0.5),       # 4.5 KB weights + 0.5 KB arena
        ("Heavy Edge Vision CNN (ResNet18)", 44000.0, 2048.0),  # 44 MB weights + 2 MB arena
    ]

    for m_name, w_kb, act_kb in models:
        print(f"\nModel Profile: {m_name} (Raw Weights={w_kb} KB, Tensor Arena={act_kb} KB)")
        for dev in devices:
            status = assess_device_fit(dev, w_kb, act_kb)
            print(f"  - {dev.name:<26}: {status}")

    print("\nDisclaimer: Physical deployment on ESP32/ARM Cortex-M7 requires converting weights")
    print("to C arrays (e.g., TFLite Micro / ONNX Runtime) and testing on physical target silicon.")
    print("[SUCCESS] Edge hardware constraint assessment completed.")


if __name__ == "__main__":
    main()

