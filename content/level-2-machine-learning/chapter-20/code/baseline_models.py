"""Tahap 1 Proyek Capstone: Pemodelan Baseline Acuan
Mengevaluasi performa Dummy Classifier dan Simple Linear Baseline sebagai patokan awal.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score

def run_baseline_evaluation():
    data_path = pathlib.Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "vibration_rms_g",
        "bearing_temperature_c",
        "rotational_speed_rpm",
        "hydraulic_pressure_bar",
        "motor_current_amp",
        "oil_acidity_tan"
    ]
    X = df[features].values
    y = df["machine_failure"].values

    # Pembagian 80:20 terisolasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("======================================================================")
    print("CAPSTONE TAHAP 1: FORMULASI MASALAH & MODEL BASELINE ACUAN")
    print("======================================================================")
    print("Deskripsi Masalah: Deteksi Dini Kerusakan Mesin Kompresor Industri")
    print(f"Data Latih: {len(X_train)} sampel | Data Uji Terisolasi: {len(X_test)} sampel")
    print(f"Distribusi Kelas Uji: Normal (0) = {(y_test==0).sum()}, Failure (1) = {(y_test==1).sum()}")
    print("----------------------------------------------------------------------")

    # 1. Dummy Majority Baseline
    dummy_maj = DummyClassifier(strategy="most_frequent")
    dummy_maj.fit(X_train, y_train)
    y_pred_dum = dummy_maj.predict(X_test)

    # 2. Simple Logistic Regression Baseline
    simple_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42))
    ])
    simple_lr.fit(X_train, y_train)
    y_pred_lr = simple_lr.predict(X_test)
    y_prob_lr = simple_lr.predict_proba(X_test)[:, 1]

    print(f"{'Model Baseline':<28} {'Akurasi':<14} {'F1-Score':<14} {'ROC-AUC'}")
    print("----------------------------------------------------------------------")
    print(f"{'Dummy Majority (Tebak 1)':<28} {accuracy_score(y_test, y_pred_dum)*100:6.2f}%       {f1_score(y_test, y_pred_dum):.4f}         0.5000")
    print(f"{'Simple Logistic Regression':<28} {accuracy_score(y_test, y_pred_lr)*100:6.2f}%       {f1_score(y_test, y_pred_lr):.4f}         {roc_auc_score(y_test, y_prob_lr):.4f}")
    print("======================================================================")
    print("TEMUAN BASELINE:")
    print(f"1. Dummy Majority menghasilkan akurasi semu {accuracy_score(y_test, y_pred_dum)*100:.2f}%,")
    print("   namun tidak memiliki kapasitas diskriminasi sama sekali (ROC-AUC = 0.50).")
    print("2. Simple Logistic Regression menjadi batas performa minimum yang WAJIB dilampaui")
    print("   oleh model kandidat yang lebih canggih (Random Forest / Gradient Boosting).")

if __name__ == "__main__":
    run_baseline_evaluation()

