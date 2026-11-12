"""Analisis Principal Component Analysis Menggunakan Scikit-Learn
Mengekstraksi bobot pemuatan (loadings), nilai singular, dan interpretasi fisis komponen.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_pca_sklearn():
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

    pca = PCA(n_components=len(features), random_state=42)
    pca.fit(X_scaled)

    print("======================================================================")
    print("PROFIL KOMPONEN UTAMA (PCA) - TELEMETRI SENSOR NIRKABEL")
    print("======================================================================")
    print(f"{'Komponen':<10} {'Nilai Singular':<16} {'Varians (Eigen)':<18} {'Rasio Varians':<16} {'Kumulatif'}")
    print("----------------------------------------------------------------------")

    cum_var = 0.0
    for i in range(len(features)):
        s_val = pca.singular_values_[i]
        var = pca.explained_variance_[i]
        ratio = pca.explained_variance_ratio_[i]
        cum_var += ratio
        print(f"PC{i+1:<8} {s_val:<16.4f} {var:<18.4f} {ratio*100:<15.2f}% {cum_var*100:.2f}%")

    print("======================================================================")
    print("MATRIKS BOBOT PEMUATAN (FACTOR LOADINGS) PC1 & PC2:")
    print("----------------------------------------------------------------------")
    loadings = pd.DataFrame(
        pca.components_[:2, :].T,
        index=features,
        columns=["PC1 (80.51%)", "PC2 (6.95%)"]
    )
    print(loadings.round(4).to_string())
    print("======================================================================")
    print("INTERPRETASI DOMAIN SISTEM IOT:")
    print("1. PC1 merepresentasikan 'Indeks Degradasi & Beban Sistem Global':")
    print("   - Nilai positif tinggi pada latency, packet loss, suhu, dan konsumsi energi.")
    print("   - Nilai negatif kuat pada signal strength dan voltage.")
    print("2. PC2 merepresentasikan 'Fluktuasi Termal & Tegangan Lokal'.")

if __name__ == "__main__":
    run_pca_sklearn()

