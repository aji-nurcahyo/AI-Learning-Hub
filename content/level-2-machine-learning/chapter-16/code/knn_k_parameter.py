"""Analisis Parameter k dan Skema Voting pada k-Nearest Neighbors
Mengevaluasi trade-off bias-varians dari k=1 (Voronoi) hingga k=50,
serta membandingkan Uniform Voting vs Distance-Weighted Voting.
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run_k_analysis():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== PENGARUH NILAI K PADA KNN (UNIFORM VS DISTANCE WEIGHTING) ===")
    print("Nilai k | Train Acc | CV Acc (5-Fold)   | Test Acc (Uniform) | Test Acc (Weighted) | Status")
    print("-" * 88)

    k_values = [1, 3, 5, 7, 11, 15, 25, 45]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        # 1. Uniform
        pipe_uni = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="uniform"))
        ])
        cv_scores = cross_val_score(pipe_uni, X_train, y_train, cv=cv, scoring="accuracy")
        pipe_uni.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, pipe_uni.predict(X_train))
        te_acc_uni = accuracy_score(y_test, pipe_uni.predict(X_test))

        # 2. Distance-Weighted
        pipe_dist = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))
        ])
        pipe_dist.fit(X_train, y_train)
        te_acc_dist = accuracy_score(y_test, pipe_dist.predict(X_test))

        if k == 1:
            desc = "Hafalan 100% (High Variance / Overfit)"
        elif k >= 25:
            desc = "Batas terlalu mulus (High Bias / Underfit)"
        else:
            desc = "Rentang k optimal"

        print(f"k = {k:<4} | {tr_acc*100:6.2f}%   | {cv_scores.mean()*100:5.2f}% ± {cv_scores.std()*100:4.2f}% | {te_acc_uni*100:6.2f}%             | {te_acc_dist*100:6.2f}%               | {desc}")

    print("-" * 88)
    print("Takeaway Kunci:")
    print("1. Nilai k=1 selalu menghasilkan Train Accuracy = 100% karena setiap titik adalah tetangga terdekat dirinya sendiri.")
    print("2. Distance-Weighted Voting (1/d) memberikan bobot lebih besar pada tetangga yang sangat dekat,")
    print("   sehingga lebih tahan terhadap pengaruh tetangga jauh saat nilai k diperbesar.")

if __name__ == "__main__":
    run_k_analysis()

