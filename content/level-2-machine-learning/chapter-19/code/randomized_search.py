"""Tuning Hyperparameter Berbasis Peluang: RandomizedSearchCV
Membandingkan efisiensi ruang pencarian beranggaran tetap terhadap penelusuran kisi.
"""

import pathlib
import time
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

def run_randomized_search():
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
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(random_state=42))
    ])

    # Distribusi probabilitas kontinu dan diskrit
    param_dist = {
        "gb__n_estimators": randint(50, 250),
        "gb__learning_rate": uniform(0.01, 0.25),
        "gb__max_depth": randint(2, 8),
        "gb__subsample": uniform(0.6, 0.4),
        "gb__min_samples_split": randint(2, 12)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_iter_budget = 25
    print("======================================================================")
    print("PENELUSURAN PARAMETER ACAK: RANDOMIZEDSEARCHCV")
    print("======================================================================")
    print(f"Alokasi Anggaran Pencarian : {n_iter_budget} iterasi sampel acak.")
    print("Keunggulan: Mampu mencakup spektrum kontinu tanpa batas grid diskrit kaku.")
    print("----------------------------------------------------------------------")

    t0 = time.perf_counter()
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter_budget,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    t_search = time.perf_counter() - t0

    print(f"Waktu Eksekusi RandomizedSearch : {t_search:.2f} detik")
    print("Hyperparameter Terbaik Ditemukan:")
    for k, v in random_search.best_params_.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  - {k:<25}: {val_str}")
    print(f"Skor F1 Validasi Silang (Mean)  : {random_search.best_score_:.4f}")

    # Evaluasi pada data uji terpisah
    best_model = random_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    print("\n----------------------------------------------------------------------")
    print(f"Hasil Evaluasi Test Set Terisolasi:")
    print(f"  F1-Score Uji   : {f1_score(y_test, y_test_pred):.4f}")
    print(f"  ROC-AUC Uji    : {roc_auc_score(y_test, y_test_prob):.4f}")
    print("======================================================================")
    print("TEMUAN PRAKTIS (Bergstra & Bengio, 2012):")
    print("1. Randomized Search sering kali menemukan konfigurasi yang setara atau lebih baik")
    print("   daripada Grid Search dengan fraksi waktu komputasi yang jauh lebih kecil.")
    print("2. Terutama unggul ketika hanya sedikit hyperparameter yang benar-benar berpengaruh besar.")

if __name__ == "__main__":
    run_randomized_search()

