"""Eksplorasi Strategi Validasi Silang (Cross-Validation)
Membandingkan Holdout, K-Fold, Stratified K-Fold, dan Repeated K-Fold.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def run_cross_validation_experiments():
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

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("======================================================================")
    print("ANALISIS RESIKO DAN VARIABILITAS STRATEGI VALIDASI SILANG")
    print("======================================================================")

    # 1. Holdout Validation (5 Percobaan Seed Acak)
    holdout_scores = []
    for seed in [1, 7, 21, 42, 99]:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        pipe.fit(X_tr, y_tr)
        holdout_scores.append(pipe.score(X_val, y_val))

    print("1. HOLDOUT VALIDATION (5 Pengulangan Split Acak 75:25):")
    for s_idx, sc in enumerate(holdout_scores):
        print(f"   Percobaan {s_idx+1}: Akurasi = {sc*100:.2f}%")
    print(f"   Rerata = {np.mean(holdout_scores)*100:.2f}% | Deviasi Standar = {np.std(holdout_scores)*100:.2f}%")
    print("   Catatan: Estimasi holdout sangat sensitif terhadap kebetulan pembagian data.")

    # 2. Standard K-Fold (k=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_scores = cross_val_score(pipe, X, y, cv=kf, scoring="accuracy")
    print("\n2. STANDARD K-FOLD (k=5):")
    print(f"   Skor per fold: {[round(s*100, 2) for s in kf_scores]}")
    print(f"   Rerata Akurasi = {kf_scores.mean()*100:.2f}% | Std = {kf_scores.std()*100:.2f}%")

    # 3. Stratified K-Fold (k=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    print("\n3. STRATIFIED K-FOLD (k=5):")
    print(f"   Skor per fold: {[round(s*100, 2) for s in skf_scores]}")
    print(f"   Rerata Akurasi = {skf_scores.mean()*100:.2f}% | Std = {skf_scores.std()*100:.2f}%")
    print("   Keunggulan: Menjaga proporsi kelas target konsisten persis di setiap fold.")

    # 4. Repeated Stratified K-Fold (5 folds x 3 repeats = 15 evaluasi)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    rskf_scores = cross_val_score(pipe, X, y, cv=rskf, scoring="accuracy")
    print("\n4. REPEATED STRATIFIED K-FOLD (5-fold x 3 repetisi = 15 partisi):")
    print(f"   Rerata Akurasi = {rskf_scores.mean()*100:.2f}% | Std = {rskf_scores.std()*100:.2f}%")
    print(f"   Rentang Nilai  : Min = {rskf_scores.min()*100:.2f}% s.d. Max = {rskf_scores.max()*100:.2f}%")

    print("======================================================================")
    print("KESIMPULAN METODOLOGI:")
    print("1. Stratified K-Fold adalah standar minimum mutlak untuk klasifikasi.")
    print("2. Repeated K-Fold memberikan estimasi interval kepercayaan yang lebih stabil")
    print("   untuk perbandingan model berskala kecil hingga moderat.")

if __name__ == "__main__":
    run_cross_validation_experiments()

