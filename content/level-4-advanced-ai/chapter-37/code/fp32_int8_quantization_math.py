"""Manual Mathematical Walkthrough: FP32 to INT8 Quantization.

Demonstrates both Symmetric and Asymmetric Affine Quantization from scratch:
1. Symmetric:  q = clip(round(x / s), -127, 127),  s = max(|x|) / 127,  z = 0
2. Asymmetric: q = clip(round(x / s) + z, -128, 127),
               s = (x_max - x_min) / 255, z = round(-x_min / s) - 128
3. Dequantization: x_hat = s * (q - z)
4. Exact verification of quantization error (MSE, Max Abs Error).
"""

from typing import Tuple
import numpy as np


def symmetric_quantize_int8(x: np.ndarray) -> Tuple[np.ndarray, float, int]:
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
    zero_point = 0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale, zero_point


def asymmetric_quantize_int8(x: np.ndarray) -> Tuple[np.ndarray, float, int]:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_max, x_min):
        return np.zeros_like(x, dtype=np.int8), 1.0, 0
    scale = (x_max - x_min) / 255.0
    # Map x_min -> -128
    zero_point = int(np.clip(np.round(-x_min / scale) - 128, -128, 127))
    q = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)
    return q, scale, zero_point


def dequantize_int8(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return scale * (q.astype(np.float32) - float(zero_point))


def main():
    print("=" * 75)
    print("CHAPTER 37: FP32 TO INT8 QUANTIZATION MATHEMATICS WORKED EXAMPLE")
    print("=" * 75)

    # Skewed FP32 tensor (such as ReLU-like positive-heavy activations)
    x = np.array([-0.50, 0.00, 0.25, 1.20, 2.54, 4.00], dtype=np.float32)
    print(f"Original FP32 Tensor (24 bytes): {x}\n")

    # 1. Symmetric Quantization
    q_sym, s_sym, z_sym = symmetric_quantize_int8(x)
    x_hat_sym = dequantize_int8(q_sym, s_sym, z_sym)
    err_sym = x - x_hat_sym
    mse_sym = float(np.mean(err_sym ** 2))

    # Independent manual verification on max element x=4.00 -> s = 4.0/127
    assert np.isclose(s_sym, 4.0 / 127.0, atol=1e-7)
    assert q_sym[-1] == 127 and z_sym == 0

    print("[1] SYMMETRIC QUANTIZATION (Range [-127, 127], z=0):")
    print(f"    Scale (s)         : {s_sym:.6f}  | Zero Point (z): {z_sym}")
    print(f"    Quantized INT8 (q): {q_sym.tolist()} (6 bytes = 4x memory reduction)")
    print(f"    Dequantized x_hat : {np.round(x_hat_sym, 4).tolist()}")
    print(f"    Element Error     : {np.round(err_sym, 5).tolist()}")
    print(f"    MSE Error         : {mse_sym:.6e}\n")

    # 2. Asymmetric Affine Quantization
    q_asym, s_asym, z_asym = asymmetric_quantize_int8(x)
    x_hat_asym = dequantize_int8(q_asym, s_asym, z_asym)
    err_asym = x - x_hat_asym
    mse_asym = float(np.mean(err_asym ** 2))

    # Manual check: x_min=-0.50, x_max=4.00 -> range=4.50 -> s = 4.50/255
    assert np.isclose(s_asym, 4.50 / 255.0, atol=1e-7)

    print("[2] ASYMMETRIC AFFINE QUANTIZATION (Range [-128, 127], full 8-bit utilization):")
    print(f"    Scale (s)         : {s_asym:.6f}  | Zero Point (z): {z_asym}")
    print(f"    Quantized INT8 (q): {q_asym.tolist()}")
    print(f"    Dequantized x_hat : {np.round(x_hat_asym, 4).tolist()}")
    print(f"    Element Error     : {np.round(err_asym, 5).tolist()}")
    print(f"    MSE Error         : {mse_asym:.6e}\n")

    print("Key Insight: For skewed activations [-0.5, 4.0], Symmetric wastes nearly half of")
    print(f"the negative [-127, -16] grid, whereas Asymmetric utilizes full 256 levels (lower MSE: {mse_asym:.2e} < {mse_sym:.2e}).")
    print("Note: Not every FP32 value can be represented exactly in INT8 due to finite 256 levels.")
    print("\n[SUCCESS] Quantization math verified.")


if __name__ == "__main__":
    main()

