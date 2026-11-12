"""K-Nearest Neighbors Classifier (KNN)
Mendemonstrasikan instance-based learning (lazy learning),
pencarian tetangga terdekat, dan agregasi suara mayoritas.
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_knn_classifier():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== K-NEAREST NEIGHBORS CLASSIFIER ===")
    print(f"Data latih disimpan di memori: {len(X_train)} sampel")
    print(f"Data uji untuk inferensi jarak: {len(X_test)} sampel")

    # Pipeline: Standarisasi wajib untuk perhitungan metrik jarak (Euclidean)
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean"))
    ])

    pipe_knn.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, pipe_knn.predict(X_train))
    test_acc = accuracy_score(y_test, pipe_knn.predict(X_test))

    print(f"\nAkurasi Latih (k=5): {train_acc * 100:.2f}%")
    print(f"Akurasi Uji   (k=5): {test_acc * 100:.2f}%")

    print("\nLaporan Klasifikasi Detail:")
    y_pred = pipe_knn.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=pipe_knn.classes_))

if __name__ == "__main__":
    run_knn_classifier()

