"""Linear Support Vector Machine (Linear SVM)
Mendemonstrasikan prinsip hiperbidang pemisah, vektor pendukung (support vectors),
margin geometris, dan klasifikasi terstandarisasi.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_linear_svm():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== LINEAR SUPPORT VECTOR MACHINE ===")
    print(f"Data latih: {len(X_train)} sampel | Data uji: {len(X_test)} sampel")

    # Pipeline: StandardScaler Wajib untuk SVM agar margin geometris adil lintas satuan
    pipe_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0, random_state=42))
    ])

    pipe_svm.fit(X_train, y_train)
    svm_model = pipe_svm.named_steps["svm"]

    train_acc = accuracy_score(y_train, pipe_svm.predict(X_train))
    test_acc = accuracy_score(y_test, pipe_svm.predict(X_test))

    print(f"\nAkurasi Latih: {train_acc * 100:.2f}%")
    print(f"Akurasi Uji  : {test_acc * 100:.2f}%")

    # Jumlah Support Vectors per kelas
    classes = svm_model.classes_
    n_sv = svm_model.n_support_
    total_sv = sum(n_sv)
    print(f"\nTotal Support Vectors: {total_sv} ({total_sv / len(X_train) * 100:.1f}% dari data latih)")
    for cls_name, count in zip(classes, n_sv):
        print(f"  - Kelas {cls_name:<8}: {count} support vectors")

    print("\nLaporan Klasifikasi Data Uji:")
    y_pred = pipe_svm.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix (CRITICAL, NORMAL, WARNING):")
    print(confusion_matrix(y_test, y_pred, labels=classes))

if __name__ == "__main__":
    run_linear_svm()

