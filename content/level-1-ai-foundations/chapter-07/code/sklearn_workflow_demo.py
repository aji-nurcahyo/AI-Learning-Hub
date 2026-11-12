"""Scikit-Learn Estimator API & End-to-End Workflow Demonstration.

Chapter 07: Python Ecosystem for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan pola desain dasar pustaka Scikit-learn:
1. Pemisahan Matriks Fitur X (2D) dan Vektor Target y (1D)
2. Pembagian Data Train vs Test (train_test_split) dengan random_state untuk Reproduksibilitas
3. Penyekalaan Fitur (StandardScaler) tanpa Kebocoran Data (Data Leakage)
4. Pola Antarmuka Estimator API: pemanggilan .fit() dan .predict()
5. Evaluasi Kinerja Awal (Accuracy Score & Confusion Matrix)

CATATAN PEDAGOGIS PENTING:
Skrip ini bertujuan memperkenalkan konvensi antarmuka (API convention) dan alur kerja standar
Scikit-learn. Teori mendalam mengenai algoritma klasifikasi, regresi, evaluasi kurva ROC/AUC,
serta hyperparameter tuning akan dipelajari secara menyeluruh pada Level 2 (Chapters 11–16).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.stdout.reconfigure(encoding="utf-8")


def run_sklearn_workflow_demo():
    print("=" * 70)
    print("DEMO 6: SCIKIT-LEARN ESTIMATOR API & BASIC WORKFLOW DEMO")
    print("=" * 70)

    # 1. Memuat Dataset Sensor Industri
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "sensor_readings.csv"

    df = pd.read_csv(csv_path)
    # Penanganan nilai hilang sederhana
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].mean())
    df["humidity_pct"] = df["humidity_pct"].fillna(df["humidity_pct"].mean())
    df["vibration_mms"] = df["vibration_mms"].fillna(df["vibration_mms"].mean())

    print("--- 1. Matriks Fitur (X) & Vektor Target (y) ---")
    # Fitur numerik masukan: [Suhu, Kelembaban, Getaran]
    feature_cols = ["temperature_c", "humidity_pct", "vibration_mms"]
    X = df[feature_cols].values  # Matriks 2D (20 sampel x 3 fitur)

    # Target biner: 0 = NORMAL, 1 = ANOMALI (WARNING / CRITICAL)
    y = (df["status"] != "NORMAL").astype(int).values  # Vektor 1D (20 label)

    print(f"Matriks Fitur X Shape : {X.shape} (Tipe: {type(X)})")
    print(f"Vektor Target y Shape : {y.shape} (Distribusi: {np.bincount(y)} -> {np.bincount(y)[0]} Normal, {np.bincount(y)[1]} Anomali)\n")

    # 2. Train / Test Split
    # random_state menjamin pemisahan yang deterministik dan dapat direproduksi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print("--- 2. Pembagian Data (Train/Test Split) ---")
    print(f"Data Pelatihan (Train) : {X_train.shape[0]} sampel (70%)")
    print(f"Data Pengujian (Test)  : {X_test.shape[0]} sampel (30%)\n")

    # 3. Penyekalaan Fitur (Preprocessing tanpa Data Leakage)
    scaler = StandardScaler()
    # PENTING: fit HANYA pada data latih (X_train), lalu transform ke train dan test!
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- 3. Standarisasi Fitur (StandardScaler) ---")
    print(f"Mean Fitur Train (sebelum scale): {X_train.mean(axis=0).round(2)}")
    print(f"Mean Fitur Train (setelah scale): {X_train_scaled.mean(axis=0).round(2)} (Mendekati 0.0)")
    print(f"Std Fitur Train  (setelah scale): {X_train_scaled.std(axis=0).round(2)} (Tepat 1.0)\n")

    # 4. Estimator API: Inisialisasi, Fit, dan Predict
    print("--- 4. Siklus Estimator API Scikit-learn ---")
    print("Instansiasi Model : LogisticRegression(random_state=42)")
    model = LogisticRegression(random_state=42)

    # Pemanggilan .fit() untuk melatih parameter model
    model.fit(X_train_scaled, y_train)
    print("Model Training Selesai via model.fit(X_train, y_train).")

    # Pemanggilan .predict() untuk melakukan inferensi pada data baru (X_test)
    y_pred = model.predict(X_test_scaled)
    print(f"Target Aktual Test    : {y_test}")
    print(f"Prediksi Model y_pred : {y_pred}\n")

    # 5. Evaluasi Sederhana
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("--- 5. Evaluasi Kinerja Awal ---")
    print(f"Akurasi Data Uji (Test Accuracy): {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(f"  [True Negative={cm[0,0]}  False Positive={cm[0,1]}]")
    print(f"  [False Negative={cm[1,0]} True Positive={cm[1,1]}]")

    print("\n[PENEGASAN PEDAGOGIS]:")
    print("Contoh ini mendemonstrasikan konsistensi pola desain Estimator Scikit-learn:")
    print("  Estimator -> fit(X, y) -> predict(X_new)")
    print("Seluruh model di Scikit-learn (SVM, Random Forest, Decision Tree, Gradient Boosting)")
    print("mengadopsi antarmuka yang persis sama, yang akan dipelajari mendalam di Level 2.")
    print("=" * 70)


if __name__ == "__main__":
    run_sklearn_workflow_demo()

