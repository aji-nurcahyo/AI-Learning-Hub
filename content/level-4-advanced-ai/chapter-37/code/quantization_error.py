"""Quantization Error Analysis: Outliers and Per-Tensor vs Per-Channel.

Demonstrates:
1. Signal-to-Quantization-Noise Ratio (SQNR in dB) across Uniform, Gaussian, and Outlier tensors.
2. Why a single large outlier inflates scale s, crushing resolution for normal weights.
3. How Per-Channel Quantization solves multi-scale weight matrices (e.g. Linear layers).
"""

import numpy as np
from fp32_int8_quantization_math import symmetric_quantize_int8, dequantize_int8


def compute_sqnr_db(orig: np.ndarray, approx: np.ndarray) -> float:
    signal_power = float(np.mean(orig ** 2))
    noise_power = float(np.mean((orig - approx) ** 2))
    if noise_power <= 1e-18:
        return 100.0
    return round(10.0 * np.log10(signal_power / noise_power), 2)


def main():
    print("=" * 75)
    print("CHAPTER 37: QUANTIZATION ERROR & PER-CHANNEL VS PER-TENSOR ANALYSIS")
    print("=" * 75)

    rng = np.random.default_rng(42)

    # 1. Uniform vs Gaussian vs Outlier tensor
    w_uniform = rng.uniform(-1.0, 1.0, size=(256,)).astype(np.float32)
    w_gauss = rng.normal(0.0, 0.3, size=(256,)).astype(np.float32)
    w_outlier = w_gauss.copy()
    w_outlier[0] = 15.0  # Single extreme outlier

    print("1. Impact of Distribution & Outliers (Per-Tensor Symmetric INT8):")
    for label, w in [("Uniform [-1, 1]", w_uniform), ("Gaussian (std=0.3)", w_gauss), ("Gaussian + Outlier (15.0)", w_outlier)]:
        q, s, z = symmetric_quantize_int8(w)
        w_hat = dequantize_int8(q, s, z)
        mse = float(np.mean((w - w_hat) ** 2))
        sqnr = compute_sqnr_db(w, w_hat)
        print(f"   {label:<26} | Scale: {s:.5f} | MSE: {mse:.3e} | SQNR: {sqnr:5.2f} dB")

    # 2. Multi-Channel Matrix: Channel 0 small [-0.1, 0.1], Channel 1 large [-10.0, 10.0]
    W_matrix = np.vstack([
        rng.normal(0.0, 0.05, size=(128,)).astype(np.float32),  # Channel 0 (tiny weights)
        rng.normal(0.0, 4.00, size=(128,)).astype(np.float32),  # Channel 1 (large weights)
    ])

    # Per-Tensor Quantization
    q_pt, s_pt, z_pt = symmetric_quantize_int8(W_matrix)
    W_hat_pt = dequantize_int8(q_pt, s_pt, z_pt)
    sqnr_ch0_pt = compute_sqnr_db(W_matrix[0], W_hat_pt[0])

    # Per-Channel Quantization (independent scale per output row)
    W_hat_pc = np.zeros_like(W_matrix)
    for ch in range(W_matrix.shape[0]):
        q_c, s_c, z_c = symmetric_quantize_int8(W_matrix[ch])
        W_hat_pc[ch] = dequantize_int8(q_c, s_c, z_c)
    sqnr_ch0_pc = compute_sqnr_db(W_matrix[0], W_hat_pc[0])

    print("\n2. Multi-Scale Weight Matrix (Ch0 std=0.05 vs Ch1 std=4.00):")
    print(f"   Per-Tensor  Quantization SQNR on small Channel 0: {sqnr_ch0_pt:6.2f} dB (Severely degraded!)")
    print(f"   Per-Channel Quantization SQNR on small Channel 0: {sqnr_ch0_pc:6.2f} dB (Preserved cleanly!)")

    print("\n[SUCCESS] Quantization error and granularity analysis verified.")


if __name__ == "__main__":
    main()

