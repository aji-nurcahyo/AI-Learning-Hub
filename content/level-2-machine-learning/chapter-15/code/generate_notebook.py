"""Pembangkit Notebook Pendamping Chapter 15 Berbasis JSON Murni"""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_notebook():
    data_path = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    df = pd.read_csv(data_path)

    features = ["temperature_c", "vibration_mm_s", "voltage_v", "current_a", "operating_hours", "load_percentage"]
    X = df[features]
    y = df["fault_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Decision Tree Gini & Entropy
    tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42).fit(X_train, y_train)
    acc_gini = accuracy_score(y_test, tree_gini.predict(X_test))

    tree_ent = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42).fit(X_train, y_train)
    acc_ent = accuracy_score(y_test, tree_ent.predict(X_test))

    # 2. Rules
    rules_text = export_text(tree_gini, feature_names=features, show_weights=True)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42).fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    # 4. OOB
    rf_oob = RandomForestClassifier(n_estimators=100, max_features="sqrt", oob_score=True, random_state=42).fit(X_train, y_train)
    cv_scores = cross_val_score(rf_oob, X_train, y_train, cv=5)

    # 5. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, y_pred_gb)
    clf_rep = classification_report(y_test, y_pred_gb, digits=4)
    cm = confusion_matrix(y_test, y_pred_gb, labels=["NORMAL", "WARNING", "FAULT"])

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 15: Decision Trees & Ensemble Learning\n",
                "## Notebook Pendamping Praktik Interaktif\n",
                "\n",
                "Notebook ini memandu Anda mengeksplorasi, memodelkan, memangkas (*pruning*), dan mengagregasi (*ensemble*) model pohon keputusan:\n",
                "1. **Eksplorasi Dataset Telemetri Sensor Kerusakan Peralatan**.\n",
                "2. **Baseline Decision Tree Classifier (Gini Impurity vs Entropy)**.\n",
                "3. **Visualisasi Aturan Pohon Keputusan (IF-THEN Rules)**.\n",
                "4. **Analisis Kriteria Pemangkasan Pohon (*Cost-Complexity Pruning*)**.\n",
                "5. **Random Forest Classifier & Feature Importance (MDI)**.\n",
                "6. **Evaluasi Out-of-Bag (OOB) vs 5-Fold Cross Validation**.\n",
                "7. **Boosting: AdaBoost vs Gradient Boosting Classifier**.\n",
                "8. **Evaluasi dan Analisis Matriks Konfusi Model Produksi**.\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 1,
            "source": [
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold\n",
                "from sklearn.tree import DecisionTreeClassifier, export_text\n",
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier\n",
                "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
                "\n",
                "# 1. Muat dataset sensor industri\n",
                "data_path = Path(\"../code/data/equipment_fault_sensor_data.csv\")\n",
                "df = pd.read_csv(data_path)\n",
                "print(f\"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom\")\n",
                "print(f\"Distribusi Target: {dict(df['fault_category'].value_counts())}\")\n",
                "df.head(3)\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom\n",
                        f"Distribusi Target: {dict(df['fault_category'].value_counts())}\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Pemisahan Dataset Bebas Kebocoran (*Leak-Free Split*)\n",
                "Dataset dibagi menjadi 75% Data Latih (270 sampel) dan 25% Data Uji (90 sampel) dengan stratifikasi kelas target."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 2,
            "source": [
                "features = ['temperature_c', 'vibration_mm_s', 'voltage_v', 'current_a', 'operating_hours', 'load_percentage']\n",
                "X = df[features]\n",
                "y = df['fault_category']\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.25, random_state=42, stratify=y\n",
                ")\n",
                "print(f\"Data Latih: {X_train.shape[0]} sampel | Data Uji: {X_test.shape[0]} sampel\")\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Data Latih: {X_train.shape[0]} sampel | Data Uji: {X_test.shape[0]} sampel\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Decision Tree: Gini Impurity vs Entropy\n",
                "Mengevaluasi performa pohon keputusan tunggal dengan batasan kedalaman (`max_depth=4`)."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 3,
            "source": [
                "tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42).fit(X_train, y_train)\n",
                "acc_gini = accuracy_score(y_test, tree_gini.predict(X_test))\n",
                "\n",
                "tree_ent = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42).fit(X_train, y_train)\n",
                "acc_ent = accuracy_score(y_test, tree_ent.predict(X_test))\n",
                "\n",
                "print(f\"Akurasi Uji Gini Impurity (max_depth=4) : {acc_gini * 100:.2f}%\")\n",
                "print(f\"Akurasi Uji Entropy/Info Gain (max_depth=4): {acc_ent * 100:.2f}%\")\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Akurasi Uji Gini Impurity (max_depth=4) : {acc_gini * 100:.2f}%\n",
                        f"Akurasi Uji Entropy/Info Gain (max_depth=4): {acc_ent * 100:.2f}%\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Visualisasi Aturan IF-THEN Pohon Keputusan\n",
                "Menampilkan cabang keputusan logis yang dipelajari secara otomatis dari data sensor."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 4,
            "source": [
                "rules = export_text(tree_gini, feature_names=features, show_weights=True)\n",
                "print(rules[:600])\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        rules_text[:600] + "\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Random Forest & Feature Importance (MDI)\n",
                "Agregasi 100 pohon keputusan dengan *random feature subsampling* ($\sqrt{p}$) untuk mereduksi varians."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 5,
            "source": [
                "rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42).fit(X_train, y_train)\n",
                "acc_rf = accuracy_score(y_test, rf.predict(X_test))\n",
                "print(f\"Akurasi Uji Random Forest (100 Trees): {acc_rf * 100:.2f}%\")\n",
                "\n",
                "feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)\n",
                "print(\"\\nFeature Importances (Mean Decrease in Impurity):\\n\", feat_imp)\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Akurasi Uji Random Forest (100 Trees): {rf_acc * 100:.2f}%\n",
                        "\nFeature Importances (Mean Decrease in Impurity):\n" + feat_imp.to_string() + "\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Evaluasi Out-of-Bag (OOB) vs 5-Fold Stratified Cross-Validation"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 6,
            "source": [
                "rf_oob = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=42).fit(X_train, y_train)\n",
                "cv_scores = cross_val_score(rf_oob, X_train, y_train, cv=5)\n",
                "\n",
                "print(f\"Skor Out-of-Bag (OOB)       : {rf_oob.oob_score_ * 100:.2f}%\")\n",
                "print(f\"5-Fold CV Score (Rata-rata) : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\")\n",
                "print(f\"Hold-out Test Score         : {accuracy_score(y_test, rf_oob.predict(X_test)) * 100:.2f}%\")\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Skor Out-of-Bag (OOB)       : {rf_oob.oob_score_ * 100:.2f}%\n",
                        f"5-Fold CV Score (Rata-rata) : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n",
                        f"Hold-out Test Score         : {accuracy_score(y_test, rf_oob.predict(X_test)) * 100:.2f}%\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Model Evaluasi Produksi: Gradient Boosting Classifier\n",
                "Mengevaluasi model berkinerja tertinggi dengan laporan klasifikasi dan matriks konfusi."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 7,
            "source": [
                "gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).fit(X_train, y_train)\n",
                "y_pred_gb = gb.predict(X_test)\n",
                "print(f\"Akurasi Uji Gradient Boosting: {accuracy_score(y_test, y_pred_gb) * 100:.2f}%\")\n",
                "print(\"\\nLaporan Klasifikasi Detail:\")\n",
                "print(classification_report(y_test, y_pred_gb, digits=4))\n",
                "print(\"Matriks Konfusi (NORMAL, WARNING, FAULT):\")\n",
                "print(confusion_matrix(y_test, y_pred_gb, labels=['NORMAL', 'WARNING', 'FAULT']))\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Akurasi Uji Gradient Boosting: {gb_acc * 100:.2f}%\n",
                        "\nLaporan Klasifikasi Detail:\n" + clf_rep + "\n",
                        "Matriks Konfusi (NORMAL, WARNING, FAULT):\n" + str(cm) + "\n"
                    ]
                }
            ]
        }
    ]

    nb_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_file = pathlib.Path(__file__).parent.parent / "notebooks" / "chapter_15_companion.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
    print(f"Generated clean executed notebook at: {out_file}")

if __name__ == "__main__":
    generate_notebook()

