"""Tuning Hyperparameter Sistematis Menggunakan GridSearchCV
Menerapkan evaluasi multi-metrik dan pengujian akhir pada set uji terisolasi.
"""

import pathlib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

def run_grid_search():
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

    # Pembagian Train-Test (20% untuk final test set yang TIDAK BOLEH disentuh saat tuning)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "rf__n_estimators": [50, 100, 150],
        "rf__max_depth": [4, 8, 12, None],
        "rf__min_samples_split": [2, 5, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    print("======================================================================")
    print("PENELUSURAN KISI HYPERPARAMETER: GRIDSEARCHCV")
    print("======================================================================")
    print(f"Jumlah Kombinasi Parameter: {3 * 4 * 3} kombinasi x 5-Fold = {3 * 4 * 3 * 5} fitting model.")
    print("Metrik Optimasi Utama (refit): F1-Score")
    print("----------------------------------------------------------------------")

    t0 = time.perf_counter()
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    t_search = time.perf_counter() - t0

    print(f"Waktu Eksekusi GridSearchCV : {t_search:.2f} detik")
    print(f"Hyperparameter Terbaik      : {grid_search.best_params_}")
    print(f"Skor F1 Validasi Terbaik    : {grid_search.best_score_:.4f}")

    # Mengambil kandidat model terbaik untuk evaluasi pada TEST SET
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    test_f1 = f1_score(y_test, y_test_pred)
    test_roc = roc_auc_score(y_test, y_test_prob)

    print("\n======================================================================")
    print("EVALUASI AKHIR PADA TEST SET TERISOLASI (ZERO DATA LEAKAGE)")
    print("======================================================================")
    print(f"Test Set F1-Score : {test_f1:.4f}")
    print(f"Test Set ROC-AUC  : {test_roc:.4f}")
    print("\nLaporan Klasifikasi Lengkap Data Uji:")
    print(classification_report(y_test, y_test_pred, digits=4))
    print("======================================================================")
    print("PRINSIP PENTING:")
    print("1. Set Uji (X_test) hanya disentuh SATU KALI di akhir untuk evaluasi akhir.")
    print("2. Pemilihan hyperparameter dilakukan sepenuhnya di dalam lingkungan Cross-Validation.")

if __name__ == "__main__":
    run_grid_search()

