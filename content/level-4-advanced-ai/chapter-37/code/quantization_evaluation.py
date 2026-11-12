"""Quantization Numerical Fidelity & Output Agreement Evaluation.

Compares FP32 Baseline logits against Dynamic INT8 and Static INT8 across:
1. Top-1 Prediction Agreement Rate (% identical decisions).
2. Mean Absolute Logit Deviation (MAE).
3. Mean Cosine Similarity of Logit Vectors.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from dynamic_quantization import apply_dynamic_quantization
from static_quantization_demo import build_static_quantized_model


def compare_logits(ref_logits: torch.Tensor, q_logits: torch.Tensor):
    ref_preds = torch.argmax(ref_logits, dim=1)
    q_preds = torch.argmax(q_logits, dim=1)
    agreement = float((ref_preds == q_preds).float().mean().item())
    mae = float(torch.mean(torch.abs(ref_logits - q_logits)).item())
    cos_sim = float(F.cosine_similarity(ref_logits, q_logits, dim=1).mean().item())
    return round(agreement * 100, 2), round(mae, 5), round(cos_sim, 6)


def main():
    print("=" * 75)
    print("CHAPTER 37: QUANTIZATION FIDELITY & LOGIT DEVIATION AUDIT")
    print("=" * 75)

    fp32_model, (X_train, _, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    dyn_model = apply_dynamic_quantization(fp32_model)
    stat_model = build_static_quantized_model(fp32_model, X_train[:250])

    with torch.no_grad():
        logits_fp32 = fp32_model(X_test)
        logits_dyn = dyn_model(X_test)
        logits_stat = stat_model(X_test)

    acc_fp32 = evaluate_accuracy(fp32_model, X_test, y_test) * 100
    acc_dyn = evaluate_accuracy(dyn_model, X_test, y_test) * 100
    acc_stat = evaluate_accuracy(stat_model, X_test, y_test) * 100

    agr_d, mae_d, cos_d = compare_logits(logits_fp32, logits_dyn)
    agr_s, mae_s, cos_s = compare_logits(logits_fp32, logits_stat)

    print(f"Test Evaluation (N={len(X_test)} samples):\n")
    print(f"{'Model Variant':<20} | {'Accuracy':<10} | {'Agreement vs FP32':<18} | {'Logit MAE':<10} | {'Cosine Sim'}")
    print("-" * 78)
    print(f"{'FP32 Baseline':<20} | {acc_fp32:<9.2f}% | {'100.00% (Ref)':<18} | {'0.00000':<10} | {'1.000000'}")
    print(f"{'Dynamic INT8':<20} | {acc_dyn:<9.2f}% | {agr_d:<17.2f}% | {mae_d:<10.5f} | {cos_d:.6f}")
    print(f"{'Static INT8':<20} | {acc_stat:<9.2f}% | {agr_s:<17.2f}% | {mae_s:<10.5f} | {cos_s:.6f}")

    print("\n[SUCCESS] Quantization fidelity evaluation completed.")


if __name__ == "__main__":
    main()

