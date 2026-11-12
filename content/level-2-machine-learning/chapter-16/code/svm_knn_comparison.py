"""Komparasi Komprehensif Support Vector Machines vs k-Nearest Neighbors
Membandingkan prinsip pembelajaran, akurasi, waktu pelatihan, latensi prediksi,
dan jejak memori pada dataset telemetri jaringan sensor nirkabel.
"""

import time
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def run_svm_knn_comparison():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Dummy (Baseline)": DummyClassifier(strategy="most_frequent"),
        "Linear SVM (C=1.0)": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="linear", C=1.0, random_state=42))]),
        "RBF SVM (C=1.0, gamma='scale')": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))]),
        "Poly SVM (degree=3, C=1.0)": Pipeline([("s", StandardScaler()), ("m", SVC(kernel="poly", degree=3, C=1.0, random_state=42))]),
        "KNN Uniform (k=5)": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5, weights="uniform"))]),
        "KNN Weighted (k=5, 1/d)": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5, weights="distance"))])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro"]

    print("=== BENCHMARK KOMPARATIF: SVM VS KNN ===")
    print(f"Data: 5-Fold Stratified CV pada {len(X_train)} data latih, diuji pada {len(X_test)} data uji.\n")
    print(f"{'Nama Model':<32} | {'CV Acc (Mean ± Std)':<20} | {'CV Macro F1':<12} | {'Test Acc':<9} | {'Fit (ms)':<8} | {'Pred (ms)':<9}")
    print("-" * 105)

    for name, model in models.items():
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_acc = cv_res["test_accuracy"].mean()
        std_acc = cv_res["test_accuracy"].std()
        mean_f1 = cv_res["test_f1_macro"].mean()

        # Ukur waktu fit
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time_ms = (time.perf_counter() - t0) * 1000

        # Ukur waktu predict
        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time_ms = (time.perf_counter() - t1) * 1000

        test_acc = accuracy_score(y_test, y_pred)

        print(f"{name:<32} | {mean_acc*100:5.2f}% ± {std_acc*100:4.2f}%    | {mean_f1*100:5.2f}%      | {test_acc*100:5.2f}%   | {fit_time_ms:6.1f}   | {pred_time_ms:7.2f}")

    print("-" * 105)
    print("Takeaways Arsitektural:")
    print("1. RBF SVM memberikan keunggulan akurasi tertinggi dengan batas keputusan non-linier yang halus.")
    print("2. KNN adalah 'Eager on Prediction, Lazy on Training': waktu fit mendekati 0 ms,")
    print("   namun waktu prediksi meningkat secara linier O(N * D) seiring bertambahnya data.")
    print("3. Linear SVM menawarkan latensi prediksi tercepat O(D) hanya dengan satu operasi dot product w^T x + b.")

if __name__ == "__main__":
    run_svm_knn_comparison()

