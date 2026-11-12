"""Teknik Pruning pada Pohon Keputusan (Pre-Pruning & Cost-Complexity Post-Pruning)
Mendemonstrasikan pencegahan overfitting melalui pemangkasan kompleksitas pohon.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def run_tree_pruning():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("=== EKSPERIMEN TREE PRUNING ===")

    # 1. Pohon Penuh Tanpa Batas (Unconstrained / Overfitting)
    unpruned = DecisionTreeClassifier(random_state=42)
    unpruned.fit(X_train, y_train)
    train_acc_full = accuracy_score(y_train, unpruned.predict(X_train))
    test_acc_full = accuracy_score(y_test, unpruned.predict(X_test))
    print("1. Pohon Tanpa Batas (Unpruned):")
    print(f"   Kedalaman: {unpruned.get_depth()} | Jumlah Daun: {unpruned.get_n_leaves()}")
    print(f"   Akurasi Latih: {train_acc_full * 100:.2f}% | Akurasi Uji: {test_acc_full * 100:.2f}%")
    print(f"   Gap Generalisasi (Overfitting): {(train_acc_full - test_acc_full) * 100:.2f}%")

    # 2. Pre-Pruning: Membatasi kedalaman dan sampel daun
    print("\n2. Eksperimen Pre-Pruning (Hyperparameter Constraints):")
    for max_d, min_leaf in [(2, 10), (3, 5), (4, 4), (5, 2)]:
        pre_tree = DecisionTreeClassifier(max_depth=max_d, min_samples_leaf=min_leaf, random_state=42)
        pre_tree.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, pre_tree.predict(X_train))
        te_acc = accuracy_score(y_test, pre_tree.predict(X_test))
        print(f"   max_depth={max_d}, min_samples_leaf={min_leaf:<2} -> Train: {tr_acc*100:.2f}% | Test: {te_acc*100:.2f}% | Daun: {pre_tree.get_n_leaves()}")

    # 3. Post-Pruning: Minimal Cost-Complexity Pruning (ccp_alpha)
    print("\n3. Post-Pruning: Minimal Cost-Complexity Pruning Path (ccp_alpha):")
    path = unpruned.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Ambil sampel alphas representatif (hindari alpha terbesar yang memangkas ke 1 daun)
    selected_alphas = ccp_alphas[::max(1, len(ccp_alphas) // 6)]
    if selected_alphas[-1] != ccp_alphas[-1]:
        selected_alphas = np.append(selected_alphas, ccp_alphas[-2])

    print("   Nilai Alpha   | Jumlah Node | Akurasi Latih | Akurasi Uji | Status")
    print("   " + "-" * 65)

    best_alpha = 0.0
    best_test_acc = 0.0

    for alpha in selected_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, clf.predict(X_train))
        te_acc = accuracy_score(y_test, clf.predict(X_test))
        n_nodes = clf.tree_.node_count

        status = ""
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_alpha = alpha
            status = "<- TERBAIK"

        print(f"   alpha = {alpha:.5f} | {n_nodes:<11} | {tr_acc*100:.2f}%        | {te_acc*100:.2f}%      | {status}")

    print(f"\nKesimpulan Post-Pruning: Nilai ccp_alpha optimal = {best_alpha:.5f} menghasilkan akurasi uji {best_test_acc*100:.2f}% dengan kompleksitas pohon yang jauh lebih ringkas.")

if __name__ == "__main__":
    run_tree_pruning()

