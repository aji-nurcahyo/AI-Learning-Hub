"""Analisis Pengaruh Parameter Gamma pada RBF Kernel SVM
Mendemonstrasikan radius pengaruh support vector dan spektrum bias-varians.
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_gamma_analysis():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== PENGARUH PARAMETER GAMMA PADA RBF KERNEL SVM (C = 1.0) ===")
    print("Parameter Gamma | Total SV | Akurasi Latih | Akurasi Uji | Karakteristik Wilayah Keputusan")
    print("-" * 88)

    gammas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    for g in gammas:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma=g, random_state=42))
        ])
        pipe.fit(X_train, y_train)

        svm = pipe.named_steps["svm"]
        n_sv = sum(svm.n_support_)

        tr_acc = accuracy_score(y_train, pipe.predict(X_train))
        te_acc = accuracy_score(y_test, pipe.predict(X_test))

        if g < 0.01:
            desc = "Radius sangat lebar, batas terlalu mulus (Underfitting)"
        elif g > 5.0:
            desc = "Radius sangat sempit, pulau-pulau terisolasi (Overfitting parah)"
        else:
            desc = "Fleksibilitas kontur optimal (Keseimbangan Generalisasi)"

        print(f"gamma = {g:<9} | {n_sv:<8} | {tr_acc*100:6.2f}%       | {te_acc*100:6.2f}%     | {desc}")

    print("-" * 88)
    print("Prinsip Matematika RBF:")
    print("Formula RBF: K(x, x') = exp(-gamma * ||x - x'||^2).")
    print("Gamma sebanding dengan 1 / (2 * sigma^2). Gamma besar mempersempit lonceng Gaussian")
    print("sehingga setiap support vector hanya memengaruhi area kecil di sekitarnya.")

if __name__ == "__main__":
    run_gamma_analysis()

