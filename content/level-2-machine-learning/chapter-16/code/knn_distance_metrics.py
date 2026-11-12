"""Analisis Metrik Jarak pada k-Nearest Neighbors
Membandingkan Jarak Euclidean (L2), Manhattan (L1), Chebyshev (L_inf), dan Minkowski (p=3).
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def manual_distance_example():
    # Contoh manual perhitungan jarak 2 sampel sensor (Sinyal dBm, Latensi ms)
    p = np.array([-70.0, 150.0])
    q = np.array([-50.0, 50.0])

    d_euc = np.sqrt(np.sum((p - q) ** 2))
    d_man = np.sum(np.abs(p - q))
    d_cheb = np.max(np.abs(p - q))

    print("=== PERHITUNGAN MANUAL METRIK JARAK ===")
    print(f"Titik P: Sinyal={p[0]} dBm, Latensi={p[1]} ms")
    print(f"Titik Q: Sinyal={q[0]} dBm, Latensi={q[1]} ms")
    print(f"1. Jarak Euclidean (L2 Norma) : sqrt((-20)^2 + 100^2) = {d_euc:.4f}")
    print(f"2. Jarak Manhattan (L1 Norma) : |-20| + |100|          = {d_man:.4f}")
    print(f"3. Jarak Chebyshev (L_inf)    : max(|-20|, |100|)      = {d_cheb:.4f}\n")

def run_metrics_comparison():
    manual_distance_example()

    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    metrics = {
        "Euclidean (L2 norm)": {"metric": "euclidean", "p": 2},
        "Manhattan (L1 norm)": {"metric": "manhattan", "p": 1},
        "Chebyshev (L_inf)": {"metric": "chebyshev", "p": np.inf},
        "Minkowski (p = 3)": {"metric": "minkowski", "p": 3}
    }

    print("=== KOMPARASI METRIK JARAK KNN PADA DATASET SENSOR (k = 5) ===")
    print("Metrik Jarak         | CV Accuracy (5-Fold) | Test Accuracy | Karakteristik Geometri")
    print("-" * 80)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, params in metrics.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, metric=params["metric"], p=params["p"] if params["p"] != np.inf else None))
        ])

        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test))

        if "Euclidean" in name:
            desc = "Garis lurus terpendek di ruang berdimensi D"
        elif "Manhattan" in name:
            desc = "Jarak grid kota (grid-based), lebih tahan outlier"
        elif "Chebyshev" in name:
            desc = "Jarak maksimum sepanjang salah satu dimensi koordinat"
        else:
            desc = "Generalisasi norma berbobot p=3"

        print(f"{name:<20} | {scores.mean()*100:5.2f}% ± {scores.std()*100:4.2f}%      | {test_acc*100:5.2f}%       | {desc}")

    print("-" * 80)

if __name__ == "__main__":
    run_metrics_comparison()

