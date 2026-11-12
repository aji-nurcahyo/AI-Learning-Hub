"""
Chapter 30 - Deep Learning Capstone
Script 5: Final Evaluation, Multi-Metrics & Error Analysis Diagnostics
Evaluasi komprehensif pada Test Set terisolasi: Akurasi, Macro F1, Matriks Konfusi,
dan diagnosis kasus salah prediksi (error analysis).
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def evaluate_model_on_test_set(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluasi akhir pada Final Test Set terisolasi beserta diseksi kasus galat.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).numpy()
        confidences = torch.max(probs, dim=1).values.numpy()

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, preds)

    # Error Analysis: Ekstrak sampel salah prediksi
    error_mask = (preds != y_test)
    error_indices = np.where(error_mask)[0]
    
    error_samples = []
    class_names = {0: "Normal", 1: "Degraded", 2: "Kritis"}

    for idx in error_indices:
        error_samples.append({
            "sample_index": int(idx),
            "machine_id": test_df.iloc[idx]["machine_id"],
            "true_label": class_names[y_test[idx]],
            "pred_label": class_names[preds[idx]],
            "confidence": float(confidences[idx]),
            "temp_c": float(test_df.iloc[idx]["temperature_c"]),
            "vib_mms": float(test_df.iloc[idx]["vibration_mms"]),
            "press_bar": float(test_df.iloc[idx]["pressure_bar"]),
            "curr_amp": float(test_df.iloc[idx]["current_amp"])
        })

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "total_test_samples": len(y_test),
        "total_errors": len(error_indices),
        "error_rate": len(error_indices) / len(y_test),
        "error_samples": error_samples,
        "classification_report": classification_report(y_test, preds, target_names=["Normal", "Degraded", "Kritis"])
    }


def main():
    print("=" * 70)
    print("CHAPTER 30 CAPSTONE: EVALUATION DIAGNOSTICS & ERROR ANALYSIS")
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

    # Inisialisasi model demo
    torch.manual_seed(42)
    model = DeepPredictiveMaintenanceNet(in_features=4, hidden_dim=64, num_classes=3)

    results = evaluate_model_on_test_set(model, X_test, y_test, test_df)

    print(f"\n[HASIL EVALUASI TEST SET (N = {results['total_test_samples']} Sampel)]")
    print(f"Test Accuracy : {results['accuracy']:.2%}")
    print(f"Test Macro F1 : {results['macro_f1']:.4f}")
    print(f"Total Galat   : {results['total_errors']} sampel ({results['error_rate']:.2%})")

    print("\n[MATRIKS KONFUSI]")
    print(results["confusion_matrix"])

    print("\n[DISEKSI ERROR ANALYSIS: CONTOH SAMPEL SALAH PREDIKSI]")
    for e in results["error_samples"][:3]:
        print(f"  Sampel #{e['sample_index']:03d} [{e['machine_id']}]: Asli = {e['true_label']} -> Diprediksi = {e['pred_label']} (Keyakinan: {e['confidence']:.2%})")
        print(f"    Nilai Sensor: Suhu={e['temp_c']}C, Getaran={e['vib_mms']}mm/s, Tekanan={e['press_bar']}bar, Arus={e['curr_amp']}A")

    print("\n[KESIMPULAN METODOLOGIS]")
    print("1. Evaluasi pada Test Set dilakukan hanya satu kali pada akhir eksperimen.")
    print("2. Error analysis mengungkap apakah galat model terkonsentrasi pada batas keputusan ambang (boundary cases).")


if __name__ == "__main__":
    main()

