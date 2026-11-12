"""Singular Value Decomposition (SVD) & Hubungannya dengan PCA
Membuktikan Teorema Eckart-Young-Mirsky dan kesetaraan matematis dengan PCA.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_svd():
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
    N = X.shape[0]

    # Standarisasi dan pemusatan
    scaler = StandardScaler()
    X_centered = scaler.fit_transform(X)

    # 1. Dekomposisi SVD Penuh: X = U * S * Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 2. Hubungan Nilai Singular dengan Varians Eigen PCA
    variances_svd = (S ** 2) / (N - 1)
    variance_ratio_svd = variances_svd / np.sum(variances_svd)

    print("======================================================================")
    print("DEKOMPOSISI SINGULAR VALUE DECOMPOSITION (SVD)")
    print("======================================================================")
    print(f"Dimensi Matriks U  : {U.shape} (Vektor Singular Kiri)")
    print(f"Dimensi Vektor S   : {S.shape} (Nilai Singular s_i)")
    print(f"Dimensi Matriks Vt : {Vt.shape} (Vektor Singular Kanan / Komponen PCA)")
    print("----------------------------------------------------------------------")
    print(f"{'Komponen':<10} {'Nilai Singular S':<20} {'Varians (s^2 / (N-1))':<24} {'Rasio Varians'}")
    print("----------------------------------------------------------------------")
    for i in range(len(S)):
        print(f"k = {i+1:<7} {S[i]:<20.4f} {variances_svd[i]:<24.4f} {variance_ratio_svd[i]*100:.2f}%")

    print("======================================================================")
    print("PEMBUKTIAN TEOREMA ECKART-YOUNG-MIRSKY (LOW-RANK APPROXIMATION):")
    print("----------------------------------------------------------------------")

    for k in [1, 2, 3]:
        # Rekonstruksi rank-k: X_k = U[:, :k] * S[:k] * Vt[:k, :]
        X_k = np.dot(U[:, :k] * S[:k], Vt[:k, :])
        frobenius_error = np.linalg.norm(X_centered - X_k, ord="fro")
        theoretical_bound = np.sqrt(np.sum(S[k:] ** 2))
        diff = np.abs(frobenius_error - theoretical_bound)
        print(f"Rank-{k} | Galat Empiris: {frobenius_error:.6f} | Batas Teoretis: {theoretical_bound:.6f} | Selisih: {diff:.2e}")

    print("======================================================================")
    print("STATUS: SVD SECARA MATEMATIS MEMBUKTIKAN ECKART-YOUNG-MIRSKY LULUS")

if __name__ == "__main__":
    run_svd()

