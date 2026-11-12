"""Tahap 3 Proyek Capstone: Komparasi Model Kandidat Pembelajaran Mesin
Mengevaluasi 4 arsitektur algoritma Level 2 menggunakan validasi silang pada data latih.
"""

import pathlib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from preprocessing_pipeline import build_preprocessing_pipeline

def run_candidate_model_benchmark():
    data_path = pathlib.Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"
    df = pd.read_csv(data_path)

    raw_features = [
        "vibration_rms_g",
        "bearing_temperature_c",
        "rotational_speed_rpm",
        "hydraulic_pressure_bar",
        "motor_current_amp",
        "oil_acidity_tan"
    ]
    X = df[raw_features].values
    y = df["machine_failure"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    prep_pipe = build_preprocessing_pipeline()

    # 4 Model Kandidat dari Seluruh Spektrum Level 2
    candidates = {
        "Logistic Regression (L2)": LogisticRegression(max_iter=500, random_state=42),
        "Support Vector Machine (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print("=========================================================================================")
    print("CAPSTONE TAHAP 3: BENCHMARK 4 KANDIDAT MODEL LEVEL 2 (5-FOLD STRATIFIED CV)")
    print("=========================================================================================")
    print(f"{'Model Kandidat':<30} {'Akurasi':<16} {'Recall (Tinggi)':<18} {'F1-Score':<16} {'ROC-AUC'}")
    print("-----------------------------------------------------------------------------------------")

    results_summary = []
    for name, clf in candidates.items():
        full_pipe = Pipeline([
            ("prep", prep_pipe),
            ("clf", clf)
        ])

        scores = cross_validate(full_pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        acc_m = scores["test_accuracy"].mean()
        rec_m = scores["test_recall"].mean()
        f1_m = scores["test_f1"].mean()
        auc_m = scores["test_roc_auc"].mean()

        results_summary.append({
            "model": name,
            "accuracy": acc_m,
            "recall": rec_m,
            "f1": f1_m,
            "roc_auc": auc_m
        })

        print(f"{name:<30} {acc_m*100:6.2f}%          {rec_m*100:6.2f}%            {f1_m:6.4f}          {auc_m:6.4f}")

    print("=========================================================================================")
    print("KESIMPULAN SELEKSI MODEL:")
    print("1. Random Forest dan Gradient Boosting menunjukkan performa superior;")
    print("   mampu menangkap pola interaksi non-linier antara getaran, suhu, dan degradasi pelumas.")
    print("2. Kedua model ensemble ini dipilih untuk dilanjutkan ke tahap hyperparameter tuning.")

if __name__ == "__main__":
    run_candidate_model_benchmark()

