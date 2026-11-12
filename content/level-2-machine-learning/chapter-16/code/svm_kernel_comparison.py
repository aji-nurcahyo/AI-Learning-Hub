"""Komparasi Fungsi Kernel pada Support Vector Machine
Membandingkan Kernel Linear, Polynomial, Radial Basis Function (RBF), dan Sigmoid.
"""

import time
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def run_kernel_comparison():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    kernels = {
        "Linear Kernel": SVC(kernel="linear", C=1.0, random_state=42),
        "Polynomial (degree=3)": SVC(kernel="poly", degree=3, C=1.0, random_state=42),
        "RBF Kernel (gamma='scale')": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "Sigmoid Kernel": SVC(kernel="sigmoid", C=1.0, random_state=42)
    }

    print("=== KOMPARASI FUNGSI KERNEL SVM ===")
    print(f"Data Evaluasi: 5-Fold Stratified CV pada {len(X_train)} data latih, diuji pada {len(X_test)} data uji.\n")
    print(f"{'Kernel Type':<28} | {'CV Accuracy (Mean ± Std)':<24} | {'Test Acc':<9} | {'Support Vectors':<15} | {'Fit (ms)':<8}")
    print("-" * 92)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in kernels.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", model)
        ])

        # 5-fold CV
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time_ms = (time.perf_counter() - t0) * 1000

        y_pred = pipe.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        n_sv = sum(pipe.named_steps["svm"].n_support_)

        print(f"{name:<28} | {scores.mean()*100:5.2f}% ± {scores.std()*100:4.2f}%          | {test_acc*100:5.2f}%   | {n_sv:<15} | {fit_time_ms:6.1f}")

    print("-" * 92)
    print("Takeaway Kunci:")
    print("1. RBF Kernel secara umum memberikan akurasi tertinggi pada batas keputusan non-linier.")
    print("2. Linear Kernel tetap menjadi pilihan utama jika data berdimensi sangat tinggi (p >> N).")
    print("3. Polynomial Kernel menangkap interaksi derajat tinggi namun lebih sensitif terhadap skala fitur.")

if __name__ == "__main__":
    run_kernel_comparison()

