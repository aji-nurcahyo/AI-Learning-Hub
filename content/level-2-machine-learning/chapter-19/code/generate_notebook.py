"""Pembangkit Notebook Pendamping Chapter 19 Berbasis JSON Murni"""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance

def generate_notebook():
    data_path = pathlib.Path(__file__).parent / "data" / "sensor_fleet_eval_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "vibration_amplitude_g",
        "operating_temp_c",
        "pressure_kpa",
        "power_draw_kw",
        "acoustic_noise_db",
        "operating_hours"
    ]
    X = df[features].values
    y = df["failure_status"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    f1_test = f1_score(y_test, y_pred)
    roc_test = roc_auc_score(y_test, y_prob)
    brier_test = brier_score_loss(y_test, y_prob)

    perm = permutation_importance(pipe, X_test, y_test, scoring="f1", n_repeats=10, random_state=42)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 19: Model Evaluation, Tuning & Explainability\n",
                "## Notebook Pendamping Praktik Interaktif\n",
                "\n",
                "Notebook ini memandu Anda mengeksekusi alur evaluasi model standar industri:\n",
                "1. **Dataset Understanding & Baseline Modeling**.\n",
                "2. **Evaluasi Metrik Komprehensif** (Regresi & Klasifikasi).\n",
                "3. **Validasi Silang Khusus**: Stratified K-Fold, Group K-Fold, dan Time Series Split.\n",
                "4. **Hyperparameter Tuning**: GridSearchCV dan RandomizedSearchCV.\n",
                "5. **Diagnostik Kurva**: Learning Curves & Validation Curves.\n",
                "6. **Kalibrasi Probabilitas** & Brier Score.\n",
                "7. **Model Explainability**: Permutation Feature Importance & Partial Dependence.\n",
                "8. **Analisis Kegagalan (Error Analysis)** & Responsible AI Audit."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 1,
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Dimensi dataset telemetri: {df.shape[0]} baris x {df.shape[1]} kolom\n",
                        f"Perangkat yang terdaftar : {df['device_id'].nunique()} unit\n",
                        "Distribusi target status kerusakan:\n",
                        f"{df['failure_status'].value_counts().to_string()}\n"
                    ]
                }
            ],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.metrics import classification_report, f1_score, roc_auc_score, brier_score_loss\n",
                "from sklearn.inspection import permutation_importance\n",
                "\n",
                "# 1. Memuat dataset telemetri armada sensor\n",
                "df = pd.read_csv('data/sensor_fleet_eval_telemetry.csv')\n",
                "features = [\n",
                "    'vibration_amplitude_g', 'operating_temp_c', 'pressure_kpa',\n",
                "    'power_draw_kw', 'acoustic_noise_db', 'operating_hours'\n",
                "]\n",
                "print(f'Dimensi dataset telemetri: {df.shape[0]} baris x {df.shape[1]} kolom')\n",
                "print(f'Perangkat yang terdaftar : {df[\"device_id\"].nunique()} unit')\n",
                "print('Distribusi target status kerusakan:')\n",
                "print(df['failure_status'].value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Pembagian Data Terisolasi & Validasi Silang (Cross-Validation)\n",
                "Memisahkan 25% data sebagai test set yang tidak boleh disentuh saat pelatihan atau penyetelan hyperparameter."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 2,
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Ukuran Data Latih: {len(X_train)} sampel | Data Uji: {len(X_test)} sampel\n",
                        f"Hasil Stratified 5-Fold CV (Data Latih Saja):\n",
                        f"  Skor F1 per fold : {[round(s, 4) for s in cv_scores]}\n",
                        f"  Rerata Skor F1   : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n"
                    ]
                }
            ],
            "source": [
                "X = df[features].values\n",
                "y = df['failure_status'].values\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.25, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "pipe = Pipeline([\n",
                "    ('scaler', StandardScaler()),\n",
                "    ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))\n",
                "])\n",
                "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')\n",
                "print(f'Ukuran Data Latih: {len(X_train)} sampel | Data Uji: {len(X_test)} sampel')\n",
                "print(f'Hasil Stratified 5-Fold CV (Data Latih Saja):')\n",
                "print(f'  Skor F1 per fold : {[round(s, 4) for s in cv_scores]}')\n",
                "print(f'  Rerata Skor F1   : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Evaluasi Akhir pada Set Uji Terisolasi (*Final Test Evaluation*)\n",
                "Mengukur performa generalisasi model terlatih pada data baru yang belum pernah dilihat sebelumnya."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 3,
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Hasil Evaluasi Test Set:\n",
                        f"  F1-Score Uji   : {f1_test:.4f}\n",
                        f"  ROC-AUC Uji    : {roc_test:.4f}\n",
                        f"  Brier Score Uji: {brier_test:.4f}\n\n",
                        "Laporan Klasifikasi Data Uji:\n",
                        f"{classification_report(y_test, y_pred, digits=4)}\n"
                    ]
                }
            ],
            "source": [
                "pipe.fit(X_train, y_train)\n",
                "y_pred = pipe.predict(X_test)\n",
                "y_prob = pipe.predict_proba(X_test)[:, 1]\n",
                "\n",
                "print('Hasil Evaluasi Test Set:')\n",
                "print(f'  F1-Score Uji   : {f1_score(y_test, y_pred):.4f}')\n",
                "print(f'  ROC-AUC Uji    : {roc_auc_score(y_test, y_prob):.4f}')\n",
                "print(f'  Brier Score Uji: {brier_score_loss(y_test, y_prob):.4f}')\n",
                "print('\\nLaporan Klasifikasi Data Uji:')\n",
                "print(classification_report(y_test, y_pred, digits=4))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Model Explainability: Permutation Feature Importance\n",
                "Mengukur penurunan F1-score aktual pada data uji saat nilai masing-masing fitur diacak."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 4,
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Peringkat Permutation Importance pada Data Uji:\n",
                        f"{pd.DataFrame({'Fitur': features, 'Permutation Drop (F1)': perm.importances_mean}).sort_values(by='Permutation Drop (F1)', ascending=False).to_string(index=False)}\n"
                    ]
                }
            ],
            "source": [
                "perm = permutation_importance(pipe, X_test, y_test, scoring='f1', n_repeats=10, random_state=42)\n",
                "perm_df = pd.DataFrame({\n",
                "    'Fitur': features,\n",
                "    'Permutation Drop (F1)': perm.importances_mean\n",
                "}).sort_values(by='Permutation Drop (F1)', ascending=False)\n",
                "print('Peringkat Permutation Importance pada Data Uji:')\n",
                "print(perm_df.to_string(index=False))"
            ]
        }
    ]

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    out_path = pathlib.Path(__file__).parent.parent / "notebooks" / "chapter_19_companion.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)

    print(f"Companion notebook Chapter 19 berhasil dibuat di: {out_path}")

if __name__ == "__main__":
    generate_notebook()

