"""
Chapter 30 - Deep Learning Capstone
Script 2: Baseline Benchmark Model (Logistic Regression / Simple MLP)
Membangun model acuan pembanding (baseline) berbasis regresi logistik terstandarisasi,
dilatih murni pada Training Set, dan dievaluasi pada Validation Set dan Test Set.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataset_loader import generate_capstone_dataset, split_data_leakage_free


def train_and_eval_baseline():
    # 1. Muat dan partisi data
    df = generate_capstone_dataset(n_samples=1000, seed=42)
    train_df, val_df, test_df = split_data_leakage_free(df, seed=42)

    features = ["temperature_c", "vibration_mms", "pressure_bar", "current_amp"]
    target = "condition_label"

    X_train_raw = train_df[features].values
    y_train = train_df[target].values

    X_val_raw = val_df[features].values
    y_val = val_df[target].values

    X_test_raw = test_df[features].values
    y_test = test_df[target].values

    # 2. Penskalaan fitur: Fit HANYA pada X_train (Zero Data Leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 3. Latih model baseline linear
    baseline_clf = LogisticRegression(max_iter=1000, random_state=42)
    baseline_clf.fit(X_train, y_train)

    # 4. Prediksi dan evaluasi
    y_val_pred = baseline_clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    y_test_pred = baseline_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    return {
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "scaler": scaler,
        "model": baseline_clf,
        "report": classification_report(y_test, y_test_pred, target_names=["Normal", "Degraded", "Kritis"])
    }


def main():
    print("=" * 70)
    print("CHAPTER 30 CAPSTONE: BASELINE BENCHMARK EVALUATION")
    print("=" * 70)

    results = train_and_eval_baseline()

    print("\n[KINERJA MODEL BASELINE (LOGISTIC REGRESSION)]")
    print(f"Validation Accuracy : {results['val_acc']:.2%}")
    print(f"Validation Macro F1 : {results['val_f1']:.4f}")
    print(f"Final Test Accuracy : {results['test_acc']:.2%}")
    print(f"Final Test Macro F1 : {results['test_f1']:.4f}")

    print("\n[LAPORAN KLASIFIKASI DATA UJI (TEST SET)]")
    print(results["report"])

    print("[PRINSIP ILMIAH CAPSTONE]")
    print("1. Baseline menetapkan batas bawah performa (performance lower-bound) yang harus dilampaui oleh model deep learning.")
    print("2. Jika model deep learning yang rumit gagal mengungguli baseline sederhana ini, pemilihannya tidak dapat dijustifikasi secara rekayasa.")


if __name__ == "__main__":
    main()

