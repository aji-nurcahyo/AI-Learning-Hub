"""Dampak Krusial Feature Scaling pada SVM dan KNN
Mendemonstrasikan dominasi fitur berskala besar (Latensi) terhadap fitur berskala kecil (Voltase)
jika standarisasi diabaikan.
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_scaling_experiment():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    print("=== PROFIL SKALA FITUR MENTAH ===")
    for f in features:
        print(f"  {f:<22}: Min={X[f].min():6.1f} | Max={X[f].max():6.1f} | Std={X[f].std():6.2f} (Rentang: {X[f].max() - X[f].min():6.1f})")

    print("\nPerhatikan bahwa Latensi memiliki rentang ~350, sedangkan Voltase hanya ~1.7.")
    print("Tanpa standarisasi, kuadrat selisih latensi (350^2 ~ 122.500) mendominasi voltase (1.7^2 ~ 2.9)")
    print("sebesar lebih dari 40.000 KALI LIPAT dalam rumus jarak Euclidean!\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Linear SVM (C=1)": SVC(kernel="linear", C=1.0, random_state=42),
        "RBF SVM (C=1, gamma='scale')": SVC(kernel="rbf", C=1.0, random_state=42)
    }

    print("=== DAMPAK FEATURE SCALING TERHADAP AKURASI UJI ===")
    print(f"{'Nama Model':<30} | {'Tanpa Scaling (Mentah)':<22} | {'StandardScaler (Z-Score)':<24} | {'MinMaxScaler (0-1)':<20}")
    print("-" * 102)

    for name, model in models.items():
        # 1. Tanpa scaling
        m_raw = model.__class__(**model.get_params())
        m_raw.fit(X_train, y_train)
        acc_raw = accuracy_score(y_test, m_raw.predict(X_test))

        # 2. StandardScaler
        scaler_std = StandardScaler()
        X_tr_std = scaler_std.fit_transform(X_train)
        X_te_std = scaler_std.transform(X_test)
        m_std = model.__class__(**model.get_params())
        m_std.fit(X_tr_std, y_train)
        acc_std = accuracy_score(y_test, m_std.predict(X_te_std))

        # 3. MinMaxScaler
        scaler_minmax = MinMaxScaler()
        X_tr_mm = scaler_minmax.fit_transform(X_train)
        X_te_mm = scaler_minmax.transform(X_test)
        m_mm = model.__class__(**model.get_params())
        m_mm.fit(X_tr_mm, y_train)
        acc_mm = accuracy_score(y_test, m_mm.predict(X_te_mm))

        print(f"{name:<30} | {acc_raw*100:6.2f}%                | {acc_std*100:6.2f}%                  | {acc_mm*100:6.2f}%")

    print("-" * 102)
    print("Kesimpulan Kritis:")
    print("Feature scaling adalah persyaratan MUTLAK untuk SVM dan KNN!")
    print("Mengabaikan scaling mendistorsi ruang metrik geometris dan merusak generalisasi model secara fatal.")

if __name__ == "__main__":
    run_scaling_experiment()

