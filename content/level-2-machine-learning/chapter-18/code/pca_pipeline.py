"""Integrasi PCA dalam Machine Learning Pipeline Bebas Kebocoran Data (Data Leakage)
Mengevaluasi trade-off dimensi fitur, akurasi klasifikasi, dan waktu komputasi.
"""

import pathlib
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def run_pca_pipeline():
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
    y = df["latent_profile"].values

    # Pembagian Train-Test terisolasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    configs = [
        ("Tanpa PCA (6 Fitur Mentah)", None),
        ("PCA (k=4, 95.2% Varians)", 4),
        ("PCA (k=2, 87.5% Varians)", 2),
        ("PCA (k=1, 80.5% Varians)", 1),
    ]

    print("======================================================================")
    print("BENCHMARK PIPELINE KLASIFIKASI DENGAN REDUKSI DIMENSI PCA")
    print("======================================================================")
    print(f"{'Konfigurasi Model':<30} {'Dimensi':<10} {'Akurasi Uji':<14} {'Waktu Latih (ms)':<18}")
    print("----------------------------------------------------------------------")

    for name, n_comp in configs:
        steps = [("scaler", StandardScaler())]
        if n_comp is not None:
            steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
        steps.append(("clf", LogisticRegression(max_iter=500, random_state=42)))

        pipe = Pipeline(steps)

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        t_fit = (time.perf_counter() - t0) * 1000

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        dim_str = str(n_comp) if n_comp else "6"

        print(f"{name:<30} {dim_str:<10} {acc*100:<13.2f}% {t_fit:<17.3f}")

    print("======================================================================")
    print("ASPEK INTEGRITAS METODOLOGI:")
    print("1. StandardScaler dan PCA di-fit HANYA pada data latih (X_train).")
    print("2. Data uji (X_test) hanya melalui operasi .transform() tanpa fit.")
    print("3. Hal ini menjamin bebas kebocoran data (zero data leakage) secara ketat.")

if __name__ == "__main__":
    run_pca_pipeline()

