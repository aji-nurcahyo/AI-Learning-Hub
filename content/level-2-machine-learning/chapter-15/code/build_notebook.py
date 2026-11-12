"""Pembuat dan Pengeksekusi Companion Notebook Chapter 15"""

import json
import pathlib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def build_chapter_15_notebook(output_path: pathlib.Path):
    nb = new_notebook()
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.12.10"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}
    }

    cells = [
        new_markdown_cell(
            "# Chapter 15 Companion Notebook: Decision Trees & Ensemble Learning\n\n"
            "**Artificial Intelligence Learning Hub** — Level 2: Machine Learning\n\n"
            "Notebook ini memandu eksperimen interaktif:\n"
            "1. Eksplorasi Dataset Telemetri Sensor Kerusakan Mesin\n"
            "2. Baseline Decision Tree Classifier (Gini vs Entropy)\n"
            "3. Visualisasi Struktur Pohon & Aturan IF-THEN\n"
            "4. Analisis Impurity (Entropy, Gini, dan Information Gain)\n"
            "5. Eksperimen Pre-Pruning & Cost-Complexity Post-Pruning (ccp_alpha)\n"
            "6. Random Forest Classifier & Analisis MDI Feature Importance\n"
            "7. Evaluasi Out-of-Bag (OOB)\n"
            "8. Adaptive Boosting (AdaBoost)\n"
            "9. Gradient Boosting Classifier\n"
            "10. Komparasi Komprehensif Model Ensemble & Error Analysis"
        ),
        new_code_cell(
            "import pathlib\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n"
            "from sklearn.tree import DecisionTreeClassifier, export_text\n"
            "from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier\n"
            "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n\n"
            "# Muat dataset\n"
            "data_path = pathlib.Path('../code/data/equipment_fault_sensor_data.csv')\n"
            "df = pd.read_csv(data_path)\n"
            "print(f'Total sampel: {len(df)}')\n"
            "df.head()"
        ),
        new_markdown_cell(
            "## 1. Pemisahan Data Bebas Kebocoran (Leak-Free Split)\n"
            "Data dibagi menjadi 75% Data Latih (270 sampel) dan 25% Data Uji (90 sampel) dengan stratifikasi kelas target."
        ),
        new_code_cell(
            "features = ['temperature_c', 'vibration_mm_s', 'voltage_v', 'current_a', 'operating_hours', 'load_percentage']\n"
            "X = df[features]\n"
            "y = df['fault_category']\n\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.25, random_state=42, stratify=y\n"
            ")\n"
            "print(f'Train size: {X_train.shape}, Test size: {X_test.shape}')\n"
            "print('Distribusi kelas latih:\\n', y_train.value_counts())"
        ),
        new_markdown_cell(
            "## 2. Decision Tree Classifier: Gini Impurity vs Entropy\n"
            "Bandingkan kedua kriteria ketidakmurnian node pada kedalaman terbatas."
        ),
        new_code_cell(
            "tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)\n"
            "tree_gini.fit(X_train, y_train)\n"
            "y_pred_gini = tree_gini.predict(X_test)\n"
            "acc_gini = accuracy_score(y_test, y_pred_gini)\n\n"
            "tree_ent = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)\n"
            "tree_ent.fit(X_train, y_train)\n"
            "y_pred_ent = tree_ent.predict(X_test)\n"
            "acc_ent = accuracy_score(y_test, y_pred_ent)\n\n"
            "print(f'Akurasi Gini (max_depth=4)   : {acc_gini * 100:.2f}%')\n"
            "print(f'Akurasi Entropy (max_depth=4): {acc_ent * 100:.2f}%')"
        ),
        new_markdown_cell(
            "## 3. Ekspor Aturan Pohon Keputusan (IF-THEN Rules)"
        ),
        new_code_cell(
            "rules = export_text(tree_gini, feature_names=features, show_weights=True)\n"
            "print(rules[:800])  # Tampilkan 800 karakter awal aturan pohon"
        ),
        new_markdown_cell(
            "## 4. Eksperimen Pruning (Pre-Pruning vs Post-Pruning)"
        ),
        new_code_cell(
            "unpruned = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)\n"
            "print(f'Unpruned Tree: Depth = {unpruned.get_depth()}, Leaves = {unpruned.get_n_leaves()}')\n"
            "print(f'Train Acc: {unpruned.score(X_train, y_train)*100:.2f}%, Test Acc: {unpruned.score(X_test, y_test)*100:.2f}%')\n\n"
            "# Cost Complexity Pruning Path\n"
            "path = unpruned.cost_complexity_pruning_path(X_train, y_train)\n"
            "alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 5)]\n"
            "for alpha in alphas:\n"
            "    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train)\n"
            "    print(f'alpha={alpha:.5f} | Nodes: {clf.tree_.node_count:<3} | Test Acc: {clf.score(X_test, y_test)*100:.2f}%')"
        ),
        new_markdown_cell(
            "## 5. Random Forest & Feature Importance (MDI)"
        ),
        new_code_cell(
            "rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)\n"
            "rf.fit(X_train, y_train)\n"
            "test_acc_rf = rf.score(X_test, y_test)\n"
            "print(f'Random Forest Test Accuracy: {test_acc_rf * 100:.2f}%')\n\n"
            "# Feature importances\n"
            "feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)\n"
            "print('Feature Importances (MDI):\\n', feat_imp)"
        ),
        new_markdown_cell(
            "## 6. Evaluasi Out-of-Bag (OOB) vs 5-Fold Cross Validation"
        ),
        new_code_cell(
            "rf_oob = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=42)\n"
            "rf_oob.fit(X_train, y_train)\n"
            "cv_scores = cross_val_score(rf_oob, X_train, y_train, cv=5)\n"
            "print(f'OOB Score          : {rf_oob.oob_score_ * 100:.2f}%')\n"
            "print(f'5-Fold CV Score    : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std()*100:.2f}%)')\n"
            "print(f'Hold-out Test Score: {rf_oob.score(X_test, y_test) * 100:.2f}%')"
        ),
        new_markdown_cell(
            "## 7. Boosting: AdaBoost vs Gradient Boosting"
        ),
        new_code_cell(
            "ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.5, random_state=42)\n"
            "ada.fit(X_train, y_train)\n"
            "print(f'AdaBoost (100 Stumps) Test Accuracy: {ada.score(X_test, y_test)*100:.2f}%')\n\n"
            "gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)\n"
            "gb.fit(X_train, y_train)\n"
            "print(f'Gradient Boosting (100 Trees) Test Accuracy: {gb.score(X_test, y_test)*100:.2f}%')"
        ),
        new_markdown_cell(
            "## 8. Laporan Evaluasi Klasifikasi Model Terbaik (Gradient Boosting)"
        ),
        new_code_cell(
            "y_pred_gb = gb.predict(X_test)\n"
            "print(classification_report(y_test, y_pred_gb, digits=4))\n"
            "print('Confusion Matrix:')\n"
            "print(confusion_matrix(y_test, y_pred_gb, labels=['NORMAL', 'WARNING', 'FAULT']))"
        )
    ]

    nb.cells = cells
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Notebook written to: {output_path}")

if __name__ == "__main__":
    out = pathlib.Path(__file__).parent.parent / "notebooks" / "chapter_15_companion.ipynb"
    build_chapter_15_notebook(out)

