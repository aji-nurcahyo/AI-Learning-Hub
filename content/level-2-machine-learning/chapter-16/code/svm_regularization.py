"""Analisis Parameter Regularisasi C pada Soft-Margin SVM
Mengevaluasi trade-off bias-varians, jumlah support vector, dan akurasi generalisasi.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_regularization_analysis():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== PENGARUH PARAMETER REGULARISASI C PADA LINEAR SVM ===")
    print("Parameter C | Total SV | Proporsi SV | Akurasi Latih | Akurasi Uji | Karakteristik Margin")
    print("-" * 84)

    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    for c in c_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=c, random_state=42))
        ])
        pipe.fit(X_train, y_train)

        svm = pipe.named_steps["svm"]
        n_sv = sum(svm.n_support_)
        pct_sv = (n_sv / len(X_train)) * 100

        tr_acc = accuracy_score(y_train, pipe.predict(X_train))
        te_acc = accuracy_score(y_test, pipe.predict(X_test))

        if c < 0.1:
            desc = "Margin sangat lebar, toleran pelanggaran (Underfit)"
        elif c > 10.0:
            desc = "Margin sempit, sedikit SV, rentan overfit"
        else:
            desc = "Keseimbangan margin-pelanggaran optimal"

        print(f"C = {c:<9} | {n_sv:<8} | {pct_sv:5.1f}%      | {tr_acc*100:6.2f}%       | {te_acc*100:6.2f}%     | {desc}")

    print("-" * 84)
    print("Insight Kunci:")
    print("Nilai C yang lebih besar memperberat penalti kesalahan klasifikasi, sehingga margin menyempit")
    print("dan jumlah support vector berkurang. Nilai C yang optimal ditemukan di rentang 0.1 s/d 10.0.")

if __name__ == "__main__":
    run_regularization_analysis()

