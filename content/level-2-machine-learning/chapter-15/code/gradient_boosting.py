"""Gradient Boosting Classifier
Mendemonstrasikan pemodelan aditif bertahap pada pseudo-residual fungsi loss,
regularisasi shrinkage (learning rate), dan evolusi galat bertahap.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

def run_gradient_boosting():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== GRADIENT BOOSTING CLASSIFIER ===")

    # Bandingkan kombinasi n_estimators dan learning_rate
    print("Estimator | Learning Rate | Max Depth | Train Acc | Test Acc  | Status")
    print("-" * 65)

    configs = [
        (30, 0.2, 2),
        (50, 0.1, 3),
        (100, 0.05, 3),
        (150, 0.1, 4),
    ]

    for n_est, lr, depth in configs:
        gb = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=depth,
            random_state=42
        )
        gb.fit(X_train, y_train)

        tr_acc = accuracy_score(y_train, gb.predict(X_train))
        te_acc = accuracy_score(y_test, gb.predict(X_test))
        print(f"{n_est:<9} | {lr:<13} | {depth:<9} | {tr_acc*100:6.2f}%   | {te_acc*100:6.2f}%   | OK")

    # Analisis Konvergensi Bertahap (Staged Predict)
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)

    train_losses = []
    test_losses = []
    for y_pred_train, y_pred_test in zip(gb_model.staged_predict_proba(X_train), gb_model.staged_predict_proba(X_test)):
        train_losses.append(log_loss(y_train, y_pred_train))
        test_losses.append(log_loss(y_test, y_pred_test))

    print("\nEvolusi Staged Log-Loss:")
    print(f"  - Iterasi 1   -> Train Loss: {train_losses[0]:.4f} | Test Loss: {test_losses[0]:.4f}")
    print(f"  - Iterasi 25  -> Train Loss: {train_losses[24]:.4f} | Test Loss: {test_losses[24]:.4f}")
    print(f"  - Iterasi 50  -> Train Loss: {train_losses[49]:.4f} | Test Loss: {test_losses[49]:.4f}")
    print(f"  - Iterasi 100 -> Train Loss: {train_losses[99]:.4f} | Test Loss: {test_losses[99]:.4f}")

    print("\nLaporan Klasifikasi Model Final Gradient Boosting (100 Pohon):")
    y_pred = gb_model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    run_gradient_boosting()

