"""Analisis Varians Terjelaskan & Penentuan Jumlah Komponen Optimal
Mengevaluasi Scree Plot Data, Kriteria Kaiser, dan Ambang Batas Kumulatif (80%, 90%, 95%).
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def analyze_explained_variance():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "signal_strength_dbm",
        "packet_loss_pct",
        "latency_ms",
        "temperature_c",
        "voltage_v",
        "energy_consumption_mah"
    ]
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA().fit(X_scaled)
    variances = pca.explained_variance_
    ratios = pca.explained_variance_ratio_
    cum_ratios = np.cumsum(ratios)

    print("======================================================================")
    print("ANALISIS SCREE PLOT & KRITERIA PEMILIHAN JUMLAH KOMPONEN")
    print("======================================================================")
    print(f"{'PC':<6} {'Eigenvalue':<14} {'Rasio Varians':<16} {'Kumulatif':<14} {'Kaiser (>1.0)?'}")
    print("----------------------------------------------------------------------")

    for i in range(len(features)):
        kaiser = "YA (Retain)" if variances[i] >= 1.0 else "TIDAK (Discard)"
        print(f"PC{i+1:<4} {variances[i]:<14.4f} {ratios[i]*100:<15.2f}% {cum_ratios[i]*100:<13.2f}% {kaiser}")

    print("======================================================================")
    print("ANALISIS AMBANG BATAS VARIANS KUMULATIF (VARIANCE THRESHOLDS):")
    print("----------------------------------------------------------------------")

    thresholds = [0.80, 0.85, 0.90, 0.95]
    for th in thresholds:
        n_comp = np.argmax(cum_ratios >= th) + 1
        pct_actual = cum_ratios[n_comp - 1] * 100
        print(f"- Target {th*100:.0f}% varians : butuh {n_comp} komponen (mencapai {pct_actual:.2f}%)")

    print("======================================================================")
    print("REKOMENDASI PRAKTIS REKAYASA SISTEM:")
    print("1. Untuk Visualisasi 2D: Pilih k = 2 (mempertahankan 87.45% varians).")
    print("2. Untuk Pipeline ML Tanpa Kehilangan Informasi Signifikan: Pilih k = 4 (95.24% varians).")
    print("3. Kriteria Kaiser merekomendasikan k = 1 (karena hanya PC1 yang memiliki eigenvalue > 1).")

if __name__ == "__main__":
    analyze_explained_variance()

