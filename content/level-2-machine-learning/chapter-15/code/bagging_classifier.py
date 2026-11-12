"""Bootstrap Aggregating (Bagging) untuk Klasifikasi
Mendemonstrasikan prinsip agregasi voting, reduksi varians, dan stabilitas model.
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_bagging():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== BAGGING CLASSIFIER (BOOTSTRAP AGGREGATING) ===")

    # 1. Base Estimator: Single High-Variance Decision Tree
    base_tree = DecisionTreeClassifier(random_state=42)
    base_tree.fit(X_train, y_train)
    single_acc = accuracy_score(y_test, base_tree.predict(X_test))
    print(f"1. Single Decision Tree (Base Estimator) Test Accuracy: {single_acc * 100:.2f}%")

    # 2. Ensemble Bagging dengan variasi jumlah pohon (n_estimators)
    print("\n2. Peningkatan Akurasi dan Stabilitas seiring Pertambahan Estimator:")
    for n_est in [5, 10, 25, 50, 100]:
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_est,
            max_samples=1.0,
            bootstrap=True,
            random_state=42
        )
        bag.fit(X_train, y_train)
        bag_acc = accuracy_score(y_test, bag.predict(X_test))
        print(f"   n_estimators = {n_est:<3} -> Test Accuracy: {bag_acc * 100:.2f}%")

    # Evaluasi detail model final dengan 50 estimator
    final_bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        bootstrap=True,
        random_state=42
    )
    final_bag.fit(X_train, y_train)
    y_pred = final_bag.predict(X_test)

    print("\n3. Laporan Klasifikasi Final Bagging (50 Estimator):")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    run_bagging()

