"""Analisis TruncatedSVD vs Standard PCA
Membedakan dekomposisi data tanpa pemusatan (uncentered) vs dengan pemusatan (centered).
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA

def compare_truncated_svd():
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
    X_raw = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. TruncatedSVD pada data mentah (tanpa pemusatan)
    tsvd_raw = TruncatedSVD(n_components=2, random_state=42)
    Z_tsvd_raw = tsvd_raw.fit_transform(X_raw)

    # 2. TruncatedSVD pada data terpusat (Z-score)
    tsvd_centered = TruncatedSVD(n_components=2, random_state=42)
    Z_tsvd_centered = tsvd_centered.fit_transform(X_scaled)

    # 3. PCA Standar (otomatis memusatkan data)
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(X_scaled)

    # Menyelaraskan tanda untuk komparasi numerik
    for col in range(2):
        if np.dot(tsvd_centered.components_[col], pca.components_[col]) < 0:
            tsvd_centered.components_[col] *= -1
            Z_tsvd_centered[:, col] *= -1

    diff_tsvd_pca = np.max(np.abs(Z_tsvd_centered - Z_pca))

    print("======================================================================")
    print("KOMPARASI TRUNCATED SVD VS STANDARD PCA")
    print("======================================================================")
    print(f"Rasio Varians TruncatedSVD (Data Mentah)   : {tsvd_raw.explained_variance_ratio_}")
    print(f"Rasio Varians TruncatedSVD (Data Terpusat) : {tsvd_centered.explained_variance_ratio_}")
    print(f"Rasio Varians PCA Standar (Data Terpusat)  : {pca.explained_variance_ratio_}")
    print("----------------------------------------------------------------------")
    print(f"Maksimum Selisih Proyeksi TruncatedSVD(Terpusat) vs PCA: {diff_tsvd_pca:.8e}")
    print("======================================================================")
    print("KESIMPULAN TEKNIS:")
    print("1. Jika data dipusatkan (mean=0), TruncatedSVD identik 100% secara matematis dengan PCA.")
    print("2. TruncatedSVD didesain untuk matriks jarang (sparse matrix, misal TF-IDF teks),")
    print("   di mana operasi pemusatan (centering) akan merusak sifat kejarangan memori (membuat matriks padat).")

    assert diff_tsvd_pca < 1e-6, "TruncatedSVD pada data terpusat harus identik dengan PCA!"

if __name__ == "__main__":
    compare_truncated_svd()

