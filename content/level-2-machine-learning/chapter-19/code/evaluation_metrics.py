"""Perhitungan & Komparasi Metrik Evaluasi Regresi dan Klasifikasi
Mengevaluasi trade-off metrik performa model pada data telemetri sensor.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)

def run_evaluation_metrics():
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
    y_reg = df["failure_risk_score"].values
    y_clf = df["failure_status"].values

    # Pembagian Train-Test 75:25
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.25, random_state=42, stratify=y_clf
    )

    # 1. Evaluasi Model Regresi
    reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    reg_pipe.fit(X_train, y_reg_train)
    y_reg_pred = reg_pipe.predict(X_test)

    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, y_reg_pred)
    mape = np.mean(np.abs((y_reg_test - y_reg_pred) / np.maximum(y_reg_test, 1.0))) * 100

    print("======================================================================")
    print("BAGIAN A: EVALUASI MODEL REGRESI (PREDIKSI SKOR RISIKO 0-100)")
    print("======================================================================")
    print(f"{'Metrik Regresi':<30} {'Nilai Uji':<16} {'Arah Kebaikan'}")
    print("----------------------------------------------------------------------")
    print(f"{'Mean Absolute Error (MAE)':<30} {mae:<16.4f} (Rendah lebih baik)")
    print(f"{'Mean Squared Error (MSE)':<30} {mse:<16.4f} (Rendah lebih baik)")
    print(f"{'Root Mean Squared Error (RMSE)':<30} {rmse:<16.4f} (Rendah lebih baik)")
    print(f"{'R-squared (R2 Score)':<30} {r2:<16.4f} (Tinggi lebih baik, maks 1.0)")
    print(f"{'Mean Absolute Pct Error (MAPE)':<30} {mape:<15.2f}% (Rendah lebih baik)")
    print("----------------------------------------------------------------------")
    print("Analisis Regresi: RMSE > MAE menandakan adanya variasi besaran galat;")
    print(f"R2 bernilai {r2*100:.2f}%, menunjukkan model menangkap hampir seluruh varians target.")

    # 2. Evaluasi Model Klasifikasi
    clf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    clf_pipe.fit(X_train, y_clf_train)
    y_clf_pred = clf_pipe.predict(X_test)
    y_clf_prob = clf_pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_clf_test, y_clf_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_clf_test, y_clf_pred)
    prec = precision_score(y_clf_test, y_clf_pred)
    rec = recall_score(y_clf_test, y_clf_pred)
    f1 = f1_score(y_clf_test, y_clf_pred)
    spec = tn / (tn + fp)
    bal_acc = balanced_accuracy_score(y_clf_test, y_clf_pred)
    roc_auc = roc_auc_score(y_clf_test, y_clf_prob)
    pr_auc = average_precision_score(y_clf_test, y_clf_prob)

    print("\n======================================================================")
    print("BAGIAN B: EVALUASI MODEL KLASIFIKASI (STATUS KERUSAKAN 0 vs 1)")
    print("======================================================================")
    print("CONFUSION MATRIX:")
    print(f"  True Negative (TN) : {tn:<6} | False Positive (FP): {fp}")
    print(f"  False Negative (FN): {fn:<6} | True Positive (TP) : {tp}")
    print("----------------------------------------------------------------------")
    print(f"{'Metrik Klasifikasi':<30} {'Nilai Uji':<16} {'Arah Kebaikan'}")
    print("----------------------------------------------------------------------")
    print(f"{'Akurasi Standar (Accuracy)':<30} {acc*100:<15.2f}% (Tinggi)")
    print(f"{'Balanced Accuracy':<30} {bal_acc*100:<15.2f}% (Tinggi)")
    print(f"{'Presisi (Precision)':<30} {prec*100:<15.2f}% (Tinggi - Minimalkan FP)")
    print(f"{'Sensitivitas / Recall':<30} {rec*100:<15.2f}% (Tinggi - Minimalkan FN)")
    print(f"{'Spesifisitas (Specificity)':<30} {spec*100:<15.2f}% (Tinggi)")
    print(f"{'F1-Score (Harmonic Mean)':<30} {f1:<16.4f} (Tinggi)")
    print(f"{'ROC-AUC Score':<30} {roc_auc:<16.4f} (Tinggi, Random=0.5)")
    print(f"{'PR-AUC (Average Precision)':<30} {pr_auc:<16.4f} (Tinggi, Baseline={y_clf_test.mean():.2f})")
    print("======================================================================")
    print("KESIMPULAN METODOLOGI EVALUASI:")
    print("1. Pada pemeliharaan prediktif IoT, False Negative (kegagalan terlewat)")
    print("   berbiaya jauh lebih mahal daripada False Positive (inspeksi preventif semu).")
    print("2. Oleh karena itu, Recall dan F1-Score lebih kritis daripada Akurasi standar.")

if __name__ == "__main__":
    run_evaluation_metrics()

