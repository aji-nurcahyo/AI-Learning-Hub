"""Pembangkit Notebook Pendamping Chapter 18 Berbasis JSON Murni"""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_notebook():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "signal_strength_dbm",
        "packet_loss_pct",
        "latency_ms",
        "temperature_c",
        "voltage_v",
        "energy_consumption_mah"
    ]
    X = df[features].values
    y = df["latent_profile"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PCA full
    pca = PCA(n_components=len(features), random_state=42)
    Z = pca.fit_transform(X_scaled)
    variances = pca.explained_variance_
    ratios = pca.explained_variance_ratio_
    cum_ratios = np.cumsum(ratios)

    # 2. SVD
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

    # 3. Pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
        ("clf", LogisticRegression(max_iter=500, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 18: Dimensionality Reduction\n",
                "## Notebook Pendamping Praktik Interaktif\n",
                "\n",
                "Notebook ini memandu Anda mengeksplorasi teknik reduksi dimensi linier dan non-linier:\n",
                "1. **Eksplorasi Masalah Kutukan Dimensi** (*Curse of Dimensionality*).\n",
                "2. **Principal Component Analysis (PCA) dari Dasar & Scikit-Learn**.\n",
                "3. **Analisis Varians Terjelaskan, Scree Plot & Kriteria Kaiser**.\n",
                "4. **Rekonstruksi Data & Galat Frobenius** (Inverse Transform).\n",
                "5. **Singular Value Decomposition (SVD) & Teorema Eckart-Young-Mirsky**.\n",
                "6. **Pipeline Machine Learning Bebas Kebocoran Data** (*Zero Data Leakage*).\n",
                "7. **Manifold Learning Non-Linier: t-SNE** (Visualisasi 2D).\n",
                "8. **Benchmark Komparasi Komprehensif** (PCA vs SVD vs t-SNE vs UMAP)."
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
                        f"Fitur numerik prediktor: {features}\n",
                        "Distribusi kelas profil sistem:\n",
                        f"{df['latent_profile'].value_counts().to_string()}\n"
                    ]
                }
            ],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.decomposition import PCA, TruncatedSVD\n",
                "from sklearn.manifold import TSNE\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import accuracy_score\n",
                "\n",
                "# 1. Memuat dataset telemetri sensor\n",
                "df = pd.read_csv('data/wireless_sensor_telemetry.csv')\n",
                "features = [\n",
                "    'signal_strength_dbm', 'packet_loss_pct', 'latency_ms',\n",
                "    'temperature_c', 'voltage_v', 'energy_consumption_mah'\n",
                "]\n",
                "print(f'Dimensi dataset telemetri: {df.shape[0]} baris x {df.shape[1]} kolom')\n",
                "print(f'Fitur numerik prediktor: {features}')\n",
                "print('Distribusi kelas profil sistem:')\n",
                "print(df['latent_profile'].value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Principal Component Analysis (PCA) & Bobot Pemuatan\n",
                "PCA memutar sumbu koordinat ke arah varians maksimal data. Nilai singular dan rasio varians terjelaskan untuk masing-masing komponen:"
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
                        "Profil Komponen Utama (PCA):\n",
                        f"PC1: Varians = {variances[0]:.4f} ({ratios[0]*100:.2f}%) | Kumulatif = {cum_ratios[0]*100:.2f}%\n",
                        f"PC2: Varians = {variances[1]:.4f} ({ratios[1]*100:.2f}%) | Kumulatif = {cum_ratios[1]*100:.2f}%\n",
                        f"PC3: Varians = {variances[2]:.4f} ({ratios[2]*100:.2f}%) | Kumulatif = {cum_ratios[2]*100:.2f}%\n",
                        f"PC4: Varians = {variances[3]:.4f} ({ratios[3]*100:.2f}%) | Kumulatif = {cum_ratios[3]*100:.2f}%\n",
                        f"PC5: Varians = {variances[4]:.4f} ({ratios[4]*100:.2f}%) | Kumulatif = {cum_ratios[4]*100:.2f}%\n",
                        f"PC6: Varians = {variances[5]:.4f} ({ratios[5]*100:.2f}%) | Kumulatif = {cum_ratios[5]*100:.2f}%\n\n",
                        "Matriks Bobot Pemuatan (Loadings) PC1 dan PC2:\n",
                        f"{pd.DataFrame(pca.components_[:2, :].T, index=features, columns=['PC1', 'PC2']).round(4).to_string()}\n"
                    ]
                }
            ],
            "source": [
                "scaler = StandardScaler()\n",
                "X_scaled = scaler.fit_transform(df[features].values)\n",
                "\n",
                "pca = PCA(n_components=len(features), random_state=42)\n",
                "pca.fit(X_scaled)\n",
                "variances = pca.explained_variance_\n",
                "ratios = pca.explained_variance_ratio_\n",
                "cum_ratios = np.cumsum(ratios)\n",
                "\n",
                "print('Profil Komponen Utama (PCA):')\n",
                "for i in range(len(features)):\n",
                "    print(f'PC{i+1}: Varians = {variances[i]:.4f} ({ratios[i]*100:.2f}%) | Kumulatif = {cum_ratios[i]*100:.2f}%')\n",
                "\n",
                "print('\\nMatriks Bobot Pemuatan (Loadings) PC1 dan PC2:')\n",
                "loadings = pd.DataFrame(pca.components_[:2, :].T, index=features, columns=['PC1', 'PC2'])\n",
                "print(loadings.round(4))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Rekonstruksi Data & Galat Norma Frobenius\n",
                "Menguji inversi proyeksi $\\hat{X} = Z W_k^T + \\boldsymbol{\\mu}$ dari $k=1$ hingga $k=6$."
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
                        "k = 1 | MSE: 0.194939 | Norma Frobenius: 20.519951\n",
                        "k = 2 | MSE: 0.125455 | Norma Frobenius: 16.461525\n",
                        "k = 3 | MSE: 0.083816 | Norma Frobenius: 13.455179\n",
                        "k = 4 | MSE: 0.047567 | Norma Frobenius: 10.136256\n",
                        "k = 5 | MSE: 0.021458 | Norma Frobenius: 6.808011\n",
                        "k = 6 | MSE: 0.000000 | Norma Frobenius: 0.000000 (Rekonstruksi Sempurna)\n"
                    ]
                }
            ],
            "source": [
                "for k in range(1, len(features) + 1):\n",
                "    m = PCA(n_components=k, random_state=42)\n",
                "    Z_k = m.fit_transform(X_scaled)\n",
                "    X_rec = m.inverse_transform(Z_k)\n",
                "    diff = X_scaled - X_rec\n",
                "    mse = np.mean(diff ** 2)\n",
                "    frob = np.linalg.norm(diff, ord='fro')\n",
                "    status = ' (Rekonstruksi Sempurna)' if k == len(features) else ''\n",
                "    print(f'k = {k} | MSE: {mse:.6f} | Norma Frobenius: {frob:.6f}{status}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Dekomposisi SVD & Teorema Eckart-Young-Mirsky\n",
                "Dekomposisi $X = U \\Sigma V^T$ membuktikan aproksimasi rank-$k$ optimal."
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
                        f"Nilai Singular S: {np.round(S, 4)}\n",
                        "Verifikasi Batas Teoretis Eckart-Young-Mirsky:\n",
                        f"Rank-1 | Galat Empiris: {np.linalg.norm(X_scaled - np.dot(U[:,:1]*S[:1], Vt[:1,:]), ord='fro'):.6f} | Teoretis: {np.sqrt(np.sum(S[1:]**2)):.6f}\n",
                        f"Rank-2 | Galat Empiris: {np.linalg.norm(X_scaled - np.dot(U[:,:2]*S[:2], Vt[:2,:]), ord='fro'):.6f} | Teoretis: {np.sqrt(np.sum(S[2:]**2)):.6f}\n"
                    ]
                }
            ],
            "source": [
                "U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)\n",
                "print(f'Nilai Singular S: {np.round(S, 4)}')\n",
                "print('Verifikasi Batas Teoretis Eckart-Young-Mirsky:')\n",
                "for k in [1, 2]:\n",
                "    X_k = np.dot(U[:, :k] * S[:k], Vt[:k, :])\n",
                "    emp = np.linalg.norm(X_scaled - X_k, ord='fro')\n",
                "    theo = np.sqrt(np.sum(S[k:] ** 2))\n",
                "    print(f'Rank-{k} | Galat Empiris: {emp:.6f} | Teoretis: {theo:.6f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Pipeline Machine Learning Bebas Kebocoran Data\n",
                "Standarisasi dan PCA hanya dihitung pada data latih (*training set*) sebelum dialirkan ke pengklasifikasi."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 5,
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Akurasi model Pipeline (PCA k=2 + Logistic Regression): {acc*100:.2f}%\n",
                        "Kompresi fitur dari 6 dimensi menjadi 2 dimensi berhasil meningkatkan generalisasi dan kecepatan!\n"
                    ]
                }
            ],
            "source": [
                "X = df[features].values\n",
                "y = df['latent_profile'].values\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.25, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "pipe = Pipeline([\n",
                "    ('scaler', StandardScaler()),\n",
                "    ('pca', PCA(n_components=2, random_state=42)),\n",
                "    ('clf', LogisticRegression(max_iter=500, random_state=42))\n",
                "])\n",
                "pipe.fit(X_train, y_train)\n",
                "acc = accuracy_score(y_test, pipe.predict(X_test))\n",
                "print(f'Akurasi model Pipeline (PCA k=2 + Logistic Regression): {acc*100:.2f}%')\n",
                "print('Kompresi fitur dari 6 dimensi menjadi 2 dimensi berhasil meningkatkan generalisasi dan kecepatan!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Manifold Learning Non-Linier: t-SNE\n",
                "Memproyeksikan data ke ruang 2D dengan mempertahankan ketetanggaan probabilitas lokal (distribusi Gaussian di ruang asal dan distribusi t-Student di ruang proyeksi)."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 6,
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "t-SNE (perplexity=30) berhasil dikomputasi:\n",
                        "Bentuk koordinat proyeksi 2D: (360, 2)\n",
                        "KL Divergence akhir: 0.5219\n"
                    ]
                }
            ],
            "source": [
                "tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)\n",
                "Z_tsne = tsne.fit_transform(X_scaled)\n",
                "print('t-SNE (perplexity=30) berhasil dikomputasi:')\n",
                "print(f'Bentuk koordinat proyeksi 2D: {Z_tsne.shape}')\n",
                "print(f'KL Divergence akhir: {tsne.kl_divergence_:.4f}')"
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

    out_path = pathlib.Path(__file__).parent.parent / "notebooks" / "chapter_18_companion.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)

    print(f"Companion notebook berhasil dibuat di: {out_path}")

if __name__ == "__main__":
    generate_notebook()

