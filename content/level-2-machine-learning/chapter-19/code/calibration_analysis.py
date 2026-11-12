"""Analisis Kalibrasi Probabilitas (Probability Calibration) & Brier Score
Mengevaluasi keandalan skor probabilitas dan menerapkan kalibrasi Platt / Isotonik.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

def run_calibration_analysis():
    data_path = pathlib.Path(__file__).parent / "data" / "sensor_fleet_eval_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "vibration_amplitude_g",
        "operating_temp_c",
        "pressure_kpa",
        "power_draw_kw",
        "acoustic_noise_db",
        "operating_hours"
    ]
    X = df[features].values
    y = df["failure_status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 1. Model Dasar Tanpa Kalibrasi Tambahan (Uncalibrated Random Forest)
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    base_pipe.fit(X_train, y_train)
    prob_uncal = base_pipe.predict_proba(X_test)[:, 1]

    # 2. Model dengan Kalibrasi Platt (Sigmoid)
    cal_sigmoid = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=5)
    cal_sigmoid.fit(X_train, y_train)
    prob_sig = cal_sigmoid.predict_proba(X_test)[:, 1]

    # 3. Model dengan Kalibrasi Isotonik (Non-parametrik)
    cal_iso = CalibratedClassifierCV(estimator=base_pipe, method="isotonic", cv=5)
    cal_iso.fit(X_train, y_train)
    prob_iso = cal_iso.predict_proba(X_test)[:, 1]

    models = [
        ("Tanpa Kalibrasi (Raw RF)", prob_uncal),
        ("Kalibrasi Sigmoid (Platt)", prob_sig),
        ("Kalibrasi Isotonik (Isotonic)", prob_iso)
    ]

    print("======================================================================")
    print("ANALISIS KALIBRASI PROBABILITAS KLASIFIKASI")
    print("======================================================================")
    print(f"{'Metode Kalibrasi':<32} {'Brier Score (Rendah)':<24} {'Log Loss':<14} {'ROC-AUC'}")
    print("----------------------------------------------------------------------")

    for name, p in models:
        brier = brier_score_loss(y_test, p)
        ll = log_loss(y_test, p)
        auc = roc_auc_score(y_test, p)
        print(f"{name:<32} {brier:<24.4f} {ll:<14.4f} {auc:.4f}")

    print("======================================================================")
    print("DATA KURVA KALIBRASI (CALIBRATION CURVE - 5 BIN TINGKAT KEPERCAYAAN):")
    print("----------------------------------------------------------------------")
    print(f"{'Bin Rentang Probabilitas':<28} {'Frekuensi Aktual (Empiris)':<30} {'Rerata Prob Prediksi'}")
    print("----------------------------------------------------------------------")

    fraction_pos, mean_pred = calibration_curve(y_test, prob_sig, n_bins=5)
    for i in range(len(fraction_pos)):
        print(f"Bin {i+1:<24} {fraction_pos[i]*100:6.2f}% ({fraction_pos[i]:.4f})            {mean_pred[i]*100:6.2f}% ({mean_pred[i]:.4f})")

    print("======================================================================")
    print("PEMAHAMAN TEORETIS KALIBRASI:")
    print("1. Brier Score mengukur galat kuadratik probabilitas: (p - y)^2.")
    print("   Nilai 0 berarti terkalibrasi sempurna; nilai 0.25 setara tebakan acak 50:50.")
    print("2. Kalibrasi probabilitas SANGAT KRUSIAL ketika output model digunakan untuk")
    print("   menghitung ekspektasi biaya risiko atau keputusan berbasis ambang batas utilitas.")

if __name__ == "__main__":
    run_calibration_analysis()

