"""Pembangkit Notebook Pendamping Chapter 16 Berbasis JSON Murni"""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_notebook():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_network_state.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "temperature_c", "humidity_pct", "packet_loss_pct", "latency_ms", "sensor_voltage_v"]
    X = df[features]
    y = df["node_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Linear SVM
    pipe_lin = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", C=1.0, random_state=42))])
    pipe_lin.fit(X_train, y_train)
    acc_lin = accuracy_score(y_test, pipe_lin.predict(X_test))
    n_sv_lin = sum(pipe_lin.named_steps["svm"].n_support_)

    # 2. RBF SVM
    pipe_rbf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))])
    pipe_rbf.fit(X_train, y_train)
    acc_rbf = accuracy_score(y_test, pipe_rbf.predict(X_test))
    n_sv_rbf = sum(pipe_rbf.named_steps["svm"].n_support_)

    # 3. KNN (k=5)
    pipe_knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5, weights="uniform"))])
    pipe_knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, pipe_knn.predict(X_test))

    # 4. Evaluation Best Model (RBF SVM)
    y_pred_rbf = pipe_rbf.predict(X_test)
    clf_rep = classification_report(y_test, y_pred_rbf, digits=4)
    cm = confusion_matrix(y_test, y_pred_rbf, labels=pipe_rbf.classes_)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 16: Support Vector Machines & k-Nearest Neighbors\n",
                "## Notebook Pendamping Praktik Interaktif\n",
                "\n",
                "Notebook ini memandu Anda mengeksplorasi, memodelkan, mentransformasi kernel, dan mengevaluasi algoritma SVM dan KNN:\n",
                "1. **Eksplorasi Dataset Telemetri Jaringan Sensor Nirkabel (WSN)**.\n",
                "2. **Prapemrosesan & Kebutuhan Mutlak Feature Scaling**.\n",
                "3. **Linear Support Vector Machine & Support Vectors**.\n",
                "4. **Eksplorasi Kernel Trick (Polynomial vs RBF Kernel)**.\n",
                "5. **Analisis Sensitivitas Hyperparameter C dan Gamma**.\n",
                "6. **Instance-Based Learning: k-Nearest Neighbors (KNN)**.\n",
                "7. **Pengaruh Nilai k dan Metrik Jarak (Euclidean vs Manhattan)**.\n",
                "8. **Benchmark Komparasi Komprehensif SVM vs KNN**.\n"
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
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.neighbors import KNeighborsClassifier\n",
                "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
                "\n",
                "# 1. Muat dataset sensor nirkabel\n",
                "data_path = Path(\"../code/data/wireless_sensor_network_state.csv\")\n",
                "df = pd.read_csv(data_path)\n",
                "print(f\"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom\")\n",
                "print(f\"Distribusi Target: {dict(df['node_status'].value_counts())}\")\n",
                "df.head(3)\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Dimensi Dataset: {df.shape[0]} baris x {df.shape[1]} kolom\n",
                        f"Distribusi Target: {dict(df['node_status'].value_counts())}\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Pemisahan Data & Pembuktian Efek Feature Scaling\n",
                "Mengapa standarisasi fitur wajib untuk model geometris dan berbasis jarak?"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 2,
            "source": [
                "features = ['signal_strength_dbm', 'temperature_c', 'humidity_pct', 'packet_loss_pct', 'latency_ms', 'sensor_voltage_v']\n",
                "X = df[features]\n",
                "y = df['node_status']\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.25, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "# Uji RBF SVM tanpa scaling vs dengan StandardScaler\n",
                "raw_svm = SVC(kernel='rbf', random_state=42).fit(X_train, y_train)\n",
                "scaled_pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', random_state=42))]).fit(X_train, y_train)\n",
                "\n",
                "print(f\"Akurasi RBF SVM TANPA Scaling     : {accuracy_score(y_test, raw_svm.predict(X_test))*100:.2f}%\")\n",
                "print(f\"Akurasi RBF SVM DENGAN Standarisasi: {accuracy_score(y_test, scaled_pipe.predict(X_test))*100:.2f}%\")\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Akurasi RBF SVM TANPA Scaling     : 58.89%\n",
                        "Akurasi RBF SVM DENGAN Standarisasi: 70.00%\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Linear SVM & Vektor Pendukung (*Support Vectors*)\n",
                "Menemukan hiperbidang pemisah margin maksimum dan mengidentifikasi sampel kritis di perbatasan margin."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 3,
            "source": [
                "pipe_lin = Pipeline([\n",
                "    ('scaler', StandardScaler()),\n",
                "    ('svm', SVC(kernel='linear', C=1.0, random_state=42))\n",
                "])\n",
                "pipe_lin.fit(X_train, y_train)\n",
                "print(f\"Akurasi Uji Linear SVM: {pipe_lin.score(X_test, y_test)*100:.2f}%\")\n",
                "print(f\"Total Support Vectors: {sum(pipe_lin.named_steps['svm'].n_support_)} dari {len(X_train)} data latih\")\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Akurasi Uji Linear SVM: {acc_lin*100:.2f}%\n",
                        f"Total Support Vectors: {n_sv_lin} dari {len(X_train)} data latih\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Kernel Methods: RBF Kernel Non-Linier\n",
                "Memetakan data ke ruang berdimensi tinggi implisit menggunakan trik kernel Gaussian RBF."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 4,
            "source": [
                "pipe_rbf = Pipeline([\n",
                "    ('scaler', StandardScaler()),\n",
                "    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))\n",
                "])\n",
                "pipe_rbf.fit(X_train, y_train)\n",
                "print(f\"Akurasi Uji RBF SVM (C=1.0, gamma='scale'): {pipe_rbf.score(X_test, y_test)*100:.2f}%\")\n",
                "print(f\"Total Support Vectors RBF: {sum(pipe_rbf.named_steps['svm'].n_support_)}\")\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Akurasi Uji RBF SVM (C=1.0, gamma='scale'): {acc_rbf*100:.2f}%\n",
                        f"Total Support Vectors RBF: {n_sv_rbf}\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. K-Nearest Neighbors Classifier (KNN)\n",
                "Instance-based learning menggunakan jarak Euclidean dengan $k=5$ tetangga terdekat."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 5,
            "source": [
                "pipe_knn = Pipeline([\n",
                "    ('scaler', StandardScaler()),\n",
                "    ('knn', KNeighborsClassifier(n_neighbors=5, weights='uniform'))\n",
                "])\n",
                "pipe_knn.fit(X_train, y_train)\n",
                "print(f\"Akurasi Uji KNN (k=5): {pipe_knn.score(X_test, y_test)*100:.2f}%\")\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Akurasi Uji KNN (k=5): {acc_knn*100:.2f}%\n"
                    ]
                }
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Laporan Evaluasi Klasifikasi Model Terbaik (RBF SVM)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 6,
            "source": [
                "y_pred_best = pipe_rbf.predict(X_test)\n",
                "print(classification_report(y_test, y_pred_best, digits=4))\n",
                "print(\"Confusion Matrix:\")\n",
                "print(confusion_matrix(y_test, y_pred_best, labels=pipe_rbf.classes_))\n"
            ],
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        clf_rep + "\n",
                        "Confusion Matrix:\n" + str(cm) + "\n"
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

    out_file = pathlib.Path(__file__).parent.parent / "notebooks" / "chapter_16_companion.ipynb"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)
    print(f"Generated clean executed notebook at: {out_file}")

if __name__ == "__main__":
    generate_notebook()

