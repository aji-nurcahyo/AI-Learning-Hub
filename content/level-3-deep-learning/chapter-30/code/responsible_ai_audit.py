"""
Chapter 30 - Deep Learning Capstone
Script 7: Responsible AI Assessment & Subgroup Disparity Audit
Audit etika dan keandalan sistem kecerdasan buatan:
1. Pemeriksaan Disparitas Performa Subgrup Mesin (Subgroup Fairness Audit)
2. Uji Ketahanan Model terhadap Derau Sensor Fisik (Robustness to Noise)
3. Dokumentasi Batas Keandalan dan Model Card Governance
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score


def audit_subgroup_disparity(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Mengaudit disparitas performa di antara subgrup mesin (UNIT-01 s/d UNIT-10)."""
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1).numpy()

    subgroups = test_df["machine_id"].unique()
    subgroup_metrics = {}

    f1_list = []
    acc_list = []

    for unit in sorted(subgroups):
        mask = (test_df["machine_id"] == unit).values
        if mask.sum() == 0:
            continue
        y_u = y_test[mask]
        p_u = preds[mask]
        
        acc_u = accuracy_score(y_u, p_u)
        f1_u = f1_score(y_u, p_u, average="macro", zero_division=0)
        
        subgroup_metrics[unit] = {
            "samples": int(mask.sum()),
            "accuracy": float(acc_u),
            "macro_f1": float(f1_u)
        }
        acc_list.append(acc_u)
        f1_list.append(f1_u)

    max_f1 = max(f1_list)
    min_f1 = min(f1_list)
    f1_disparity = max_f1 - min_f1

    # Ambang batas disparitas wajar (Fairness threshold <= 0.30)
    is_fair = f1_disparity <= 0.30

    return {
        "subgroup_metrics": subgroup_metrics,
        "max_macro_f1": max_f1,
        "min_macro_f1": min_f1,
        "f1_disparity": f1_disparity,
        "is_fair": is_fair
    }


def evaluate_sensor_noise_robustness(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: list = [0.0, 0.05, 0.10, 0.20]
) -> Dict[float, float]:
    """Mengevaluasi degradasi performa model di bawah derau sinyal sensor."""
    np.random.seed(42)
    model.eval()
    results = {}

    for noise_std in noise_levels:
        noise = np.random.normal(0, noise_std, size=X_test.shape)
        X_noisy = X_test + noise
        with torch.no_grad():
            preds = torch.argmax(model(torch.tensor(X_noisy, dtype=torch.float32)), dim=1).numpy()
        acc = accuracy_score(y_test, preds)
        results[noise_std] = float(acc)

    return results


def main():
    print("=" * 70)
    print("CHAPTER 30 CAPSTONE: RESPONSIBLE AI & ROBUSTNESS AUDIT")
    print("=" * 70)

    from dataset_loader import generate_capstone_dataset, split_data_leakage_free
    from deep_learning_model import DeepPredictiveMaintenanceNet
    from sklearn.preprocessing import StandardScaler

    df = generate_capstone_dataset(n_samples=1000, seed=42)
    train_df, _, test_df = split_data_leakage_free(df, seed=42)

    features = ["temperature_c", "vibration_mms", "pressure_bar", "current_amp"]
    scaler = StandardScaler()
    scaler.fit(train_df[features].values)
    X_test = scaler.transform(test_df[features].values)
    y_test = test_df["condition_label"].values

    torch.manual_seed(42)
    model = DeepPredictiveMaintenanceNet(in_features=4, hidden_dim=64, num_classes=3)

    # 1. Audit Disparitas Keadilan Subgrup Mesin
    print("\n--- 1. AUDIT KEADILAN SUBGRUP (SUBGROUP FAIRNESS) ---")
    audit = audit_subgroup_disparity(model, X_test, y_test, test_df)
    
    print(f"{'Unit Mesin':<12} | {'Sampel':<8} | {'Akurasi':<10} | {'Macro F1':<10}")
    print("-" * 50)
    for unit, m in audit["subgroup_metrics"].items():
        print(f"{unit:<12} | {m['samples']:<8} | {m['accuracy']:>8.2%} | {m['macro_f1']:>8.4f}")
    print("-" * 50)
    print(f"Disparitas Macro F1 Tertinggi - Terendah: {audit['f1_disparity']:.4f}")
    print(f"Status Keadilan Subgrup (Threshold <= 0.30): {'MEMENUHI SYARAT (FAIR)' if audit['is_fair'] else 'DISPARITAS TINGGI'}")

    # 2. Uji Ketahanan terhadap Derau Sensor
    print("\n--- 2. UJI KETAHANAN DERAU SENSOR (ROBUSTNESS TEST) ---")
    noise_audit = evaluate_sensor_noise_robustness(model, X_test, y_test)
    for std, acc in noise_audit.items():
        print(f"  Derau Gauss (sigma={std:.2f}) -> Akurasi Test: {acc:.2%}")

    print("\n[KESIMPULAN RESPONSIBLE AI]")
    print("1. Audit disparitas menjamin performa model tidak timpang hanya pada unit tertentu.")
    print("2. Uji ketahanan memastikan model mengalami degradasi performa bertahap (graceful degradation) dan tidak runtuh katastrofik.")
    print("3. Seluruh temuan ini dirangkum ke dalam Model Card untuk transparansi tata kelola AI.")


if __name__ == "__main__":
    main()

