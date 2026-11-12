"""Perhitungan Manual dan Komparasi Kriteria Ketidakmurnian Node
(Entropy, Gini Impurity, dan Information Gain)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

def calculate_entropy(labels: list) -> float:
    """Menghitung Entropy Shannon: H(S) = -sum(p_i * log2(p_i))"""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    # Hindari log2(0) dengan memfilter p > 0
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return float(entropy)

def calculate_gini(labels: list) -> float:
    """Menghitung Gini Impurity: Gini(S) = 1 - sum(p_i^2)"""
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1.0 - np.sum(probabilities ** 2)
    return float(gini)

def calculate_information_gain(parent_labels: list, left_labels: list, right_labels: list, criterion: str = "entropy") -> float:
    """Menghitung Penurunan Impurity (Information Gain):
    IG = Impurity(Parent) - [ (|Left|/|Total|) * Impurity(Left) + (|Right|/|Total|) * Impurity(Right) ]
    """
    total_n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    imp_fn = calculate_entropy if criterion == "entropy" else calculate_gini

    parent_imp = imp_fn(parent_labels)
    left_imp = imp_fn(left_labels)
    right_imp = imp_fn(right_labels)

    weighted_child_imp = (n_left / total_n) * left_imp + (n_right / total_n) * right_imp
    return float(parent_imp - weighted_child_imp)

def run_demonstration():
    print("=== PERHITUNGAN MANUAL ENTROPY, GINI, & INFORMATION GAIN ===")

    # Contoh subset sampel industri: 10 sampel di node induk (4 NORMAL, 6 FAULT)
    parent_node = ["NORMAL"] * 4 + ["FAULT"] * 6

    # Simulasi split kandidat fitur: Suhu > 75°C
    # Cabang Kiri (Suhu <= 75°C): 4 NORMAL, 1 FAULT
    left_child = ["NORMAL"] * 4 + ["FAULT"] * 1
    # Cabang Kanan (Suhu > 75°C): 0 NORMAL, 5 FAULT (Daun murni!)
    right_child = ["FAULT"] * 5

    # 1. Perhitungan Induk
    h_parent = calculate_entropy(parent_node)
    g_parent = calculate_gini(parent_node)
    print(f"Node Induk (Total: {len(parent_node)} sampel | 4 Normal, 6 Fault):")
    print(f"  - Shannon Entropy : {h_parent:.4f} bits (Maksimal untuk 2 kelas = 1.0000)")
    print(f"  - Gini Impurity   : {g_parent:.4f} (Maksimal untuk 2 kelas = 0.5000)")

    # 2. Perhitungan Anak
    h_left = calculate_entropy(left_child)
    g_left = calculate_gini(left_child)
    h_right = calculate_entropy(right_child)
    g_right = calculate_gini(right_child)

    print(f"\nCabang Kiri (Total: {len(left_child)} | 4 Normal, 1 Fault):")
    print(f"  - Entropy: {h_left:.4f} | Gini: {g_left:.4f}")

    print(f"\nCabang Kanan (Total: {len(right_child)} | 0 Normal, 5 Fault [Murni]):")
    print(f"  - Entropy: {h_right:.4f} | Gini: {g_right:.4f}")

    # 3. Information Gain
    ig_entropy = calculate_information_gain(parent_node, left_child, right_child, criterion="entropy")
    ig_gini = calculate_information_gain(parent_node, left_child, right_child, criterion="gini")

    print(f"\nInformation Gain (Penurunan Ketidakmurnian):")
    print(f"  - Information Gain (Entropy): {ig_entropy:.4f} bits")
    print(f"  - Gini Gain (Penurunan Gini): {ig_gini:.4f}")

    # 4. Validasi Independen dengan Scikit-learn
    # Representasi numerik fitur biner dummy
    X = np.array([[0]] * 5 + [[1]] * 5)
    y = np.array([0] * 4 + [1] + [1] * 5)

    tree_ent = DecisionTreeClassifier(criterion="entropy", max_depth=1).fit(X, y)
    tree_gin = DecisionTreeClassifier(criterion="gini", max_depth=1).fit(X, y)

    print("\nValidasi Menggunakan Scikit-learn Tree Impurity:")
    print(f"  - Sklearn Parent Entropy: {tree_ent.tree_.impurity[0]:.4f} (Cocok manual: {np.isclose(tree_ent.tree_.impurity[0], h_parent)})")
    print(f"  - Sklearn Parent Gini   : {tree_gin.tree_.impurity[0]:.4f} (Cocok manual: {np.isclose(tree_gin.tree_.impurity[0], g_parent)})")

if __name__ == "__main__":
    run_demonstration()

