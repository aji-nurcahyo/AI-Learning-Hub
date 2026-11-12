"""Validasi Silang Khusus: Group K-Fold & Time Series Split
Mencegah kebocoran data kelompok perangkat (group leakage) dan kebocoran temporal.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def run_group_time_validation():
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
    groups = df["device_id"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("======================================================================")
    print("DEMONSTRASI KEBOCORAN KELOMPOK (GROUP LEAKAGE) VS GROUP K-FOLD")
    print("======================================================================")

    # 1. Random K-Fold (Abaikan grup perangkat - rentan bocor)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_scores = cross_val_score(pipe, X, y, cv=kf, scoring="accuracy")
    print("1. Standar Random K-Fold (Abaikan ID Perangkat):")
    print(f"   Rerata Akurasi: {kf_scores.mean()*100:.2f}% (+/- {kf_scores.std()*100:.2f}%)")
    print("   Peringatan: Data dari perangkat yang sama tersebar di train DAN validation!")

    # 2. Group K-Fold (Pisahkan perangkat secara eksklusif)
    gkf = GroupKFold(n_splits=5)
    gkf_scores = cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring="accuracy")
    print("\n2. Group K-Fold (Pengelompokan Berdasarkan device_id):")
    print(f"   Rerata Akurasi: {gkf_scores.mean()*100:.2f}% (+/- {gkf_scores.std()*100:.2f}%)")
    print("   Realitas: Mengukur kemampuan generalisasi model ke perangkat BARU yang belum pernah dilihat.")
    print(f"   Penurunan Akurasi: {(kf_scores.mean() - gkf_scores.mean())*100:.2f}% (bukti optimisme semu Random K-Fold).")

    print("\n======================================================================")
    print("DEMONSTRASI VALIDASI TEMPORAL: TIME SERIES SPLIT")
    print("======================================================================")

    # Urutkan berdasarkan siklus waktu
    df_sorted = df.sort_values(by=["cycle_id", "device_id"]).reset_index(drop=True)
    X_time = df_sorted[features].values
    y_time = df_sorted["failure_status"].values

    tscv = TimeSeriesSplit(n_splits=5)
    time_scores = []
    print(f"{'Fold':<8} {'Ukuran Latih':<16} {'Ukuran Validasi':<18} {'Akurasi'}")
    print("----------------------------------------------------------------------")

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_time)):
        pipe.fit(X_time[tr_idx], y_time[tr_idx])
        score = pipe.score(X_time[val_idx], y_time[val_idx])
        time_scores.append(score)
        print(f"Fold {fold_idx+1:<3} {len(tr_idx):<16} {len(val_idx):<18} {score*100:.2f}%")

    print("----------------------------------------------------------------------")
    print(f"Rerata Akurasi Temporal: {np.mean(time_scores)*100:.2f}% (+/- {np.std(time_scores)*100:.2f}%)")
    print("======================================================================")
    print("PEDOMAN PEMILIHAN STRATEGI VALIDASI:")
    print("1. Gunakan GroupKFold saat data memiliki entitas independen (pasien, pengguna, perangkat).")
    print("2. Gunakan TimeSeriesSplit saat data memiliki autokorelasi serial temporal (tanpa shuffle).")

if __name__ == "__main__":
    run_group_time_validation()

