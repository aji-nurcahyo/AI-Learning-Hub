"""Visualisasi Struktur Pohon Keputusan (Rules & Text Representation)
Mendemonstrasikan aturan IF-THEN, splitting threshold, dan distribusi sampel daun.
"""

import pathlib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

def run_tree_visualization():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a"]
    X = df[features]
    y = df["fault_category"]

    # Latih pohon dangkal agar mudah diinterpretasikan dan dipahami aturan IF-THEN nya
    tree = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=10, random_state=42)
    tree.fit(X, y)

    print("=== STRUKTUR POHON KEPUTUSAN (IF-THEN RULES) ===")
    print(f"Total Sampel Terlatih: {len(X)}")
    print(f"Fitur: {features}")
    print(f"Kelas Target: {list(tree.classes_)}\n")

    tree_rules = export_text(tree, feature_names=features, show_weights=True)
    print(tree_rules)

    print("\nDiagram Alur Keputusan Teratas:")
    root_feat = features[tree.tree_.feature[0]]
    root_thresh = tree.tree_.threshold[0]
    print(f"       [ Root Node: Apakah {root_feat} <= {root_thresh:.2f}? ]")
    print("                  /                     \\")
    print("           (YA)  /                       \\ (TIDAK)")
    print("                v                         v")
    left_feat = features[tree.tree_.feature[tree.tree_.children_left[0]]]
    left_thresh = tree.tree_.threshold[tree.tree_.children_left[0]]
    right_feat = features[tree.tree_.feature[tree.tree_.children_right[0]]]
    right_thresh = tree.tree_.threshold[tree.tree_.children_right[0]]
    print(f"      [{left_feat} <= {left_thresh:.2f}]           [{right_feat} <= {right_thresh:.2f}]")
    print("\nPohon keputusan mengonversi relasi numerik kompleks menjadi rangkaian")
    print("proposisi aturan IF-THEN yang dapat diverifikasi secara audit forensik oleh insinyur industri.")

if __name__ == "__main__":
    run_tree_visualization()

