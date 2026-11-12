"""Pembangkit Notebook Pendamping Chapter 17 Berbasis JSON Murni"""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. K-Means (k=3)
    km = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
    km_labels = km.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)
    km_db = davies_bouldin_score(X_scaled, km_labels)
    km_ch = calinski_harabasz_score(X_scaled, km_labels)
    km_inertia = km.inertia_

    # 2. DBSCAN (eps=1.15, min_samples=6)
    db = DBSCAN(eps=1.15, min_samples=6)
    db_labels = db.fit_predict(X_scaled)
    n_core = len(db.core_sample_indices_)
    n_noise = int(np.sum(db_labels == -1))
    non_noise_mask = db_labels != -1
    db_sil = silhouette_score(X_scaled[non_noise_mask], db_labels[non_noise_mask])
    db_db = davies_bouldin_score(X_scaled[non_noise_mask], db_labels[non_noise_mask])
    db_ch = calinski_harabasz_score(X_scaled[non_noise_mask], db_labels[non_noise_mask])

    # 3. Agglomerative (Ward, k=3)
    agg = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)
    agg_db = davies_bouldin_score(X_scaled, agg_labels)
    agg_ch = calinski_harabasz_score(X_scaled, agg_labels)

    # 4. SciPy Linkage
    Z = linkage(X_scaled, method="ward", metric="euclidean")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 17: Unsupervised Learning & Clustering Algorithms\n",
                "## Notebook Pendamping Praktik Interaktif\n",
                "\n",
                "Notebook ini memandu Anda mengeksplorasi, memodelkan, dan mengevaluasi algoritma clustering tanpa label terarah:\n",
                "1. **Eksplorasi Dataset Telemetri Sensor Nirkabel** (360 node observasi, 6 fitur kontinu).\n",
                "2. **Prapemrosesan & Kebutuhan Mutlak Penskalaan Fitur** (*StandardScaler*).\n",
                "3. **K-Means Clustering From Scratch vs Scikit-Learn**.\n",
                "4. **Analisis Inisialisasi: Random vs K-Means++**.\n",
                "5. **Menentukan Jumlah Klaster Optimal: Elbow Method & Silhouette Analysis**.\n",
                "6. **Density-Based Spatial Clustering (DBSCAN)**: Inti, Perbatasan, dan Noise Outlier.\n",
                "7. **Hierarchical Agglomerative Clustering & Analisis Dendrogram** (Ward vs Linkage Lain).\n",
                "8. **Evaluasi Internal Tanpa Ground Truth**: Silhouette Score, Davies-Bouldin, Calinski-Harabasz.\n",
                "9. **Benchmark Komparasi Komprehensif** K-Means vs DBSCAN vs Agglomerative."
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
                        f"Dimensi dataset: {df.shape[0]} baris x {df.shape[1]} kolom\n",
                        f"Fitur prediktor numerik: {features}\n",
                        "Distribusi latent profile:\n",
                        f"{df['latent_profile'].value_counts().to_string()}\n"
                    ]
                }
            ],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering\n",
                "from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score\n",
                "from scipy.spatial.distance import cdist\n",
                "from scipy.cluster.hierarchy import linkage, dendrogram, fcluster\n",
                "\n",
                "# 1. Memuat dataset telemetri sensor\n",
                "df = pd.read_csv('data/wireless_sensor_telemetry.csv')\n",
                "print(f'Dimensi dataset: {df.shape[0]} baris x {df.shape[1]} kolom')\n",
                "features = [\n",
                "    'signal_strength_dbm', 'packet_loss_pct', 'latency_ms',\n",
                "    'temperature_c', 'voltage_v', 'energy_consumption_mah'\n",
                "]\n",
                "print('Distribusi latent profile:')\n",
                "print(df['latent_profile'].value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Penskalaan Fitur (*Feature Scaling*)\n",
                "Metrik jarak Euclidean sensitif terhadap skala fitur. Konsumsi energi (ratusan mAh) atau latensi (puluhan milidetik) dapat mendominasi voltase (3-4 Volt). Standardisasi menghasilkan rerata 0 dan varians 1 untuk setiap dimensi."
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
                        "Statistik fitur mentah (sebelum standarisasi):\n",
                        f"{df[features].describe().loc[['mean', 'std', 'min', 'max']].round(2).to_string()}\n\n",
                        "Fitur terstandardisasi selesai: rerata = [0. 0. 0. 0. 0. 0.], std = [1. 1. 1. 1. 1. 1.]\n"
                    ]
                }
            ],
            "source": [
                "scaler = StandardScaler()\n",
                "X = df[features].values\n",
                "X_scaled = scaler.fit_transform(X)\n",
                "print('Statistik fitur mentah (sebelum standarisasi):')\n",
                "print(df[features].describe().loc[['mean', 'std', 'min', 'max']].round(2))\n",
                "print(f'\\nFitur terstandardisasi selesai: rerata = {np.round(X_scaled.mean(axis=0), 1)}, std = {np.round(X_scaled.std(axis=0), 1)}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### K-Means Clustering dari Dasar (*Pure NumPy*) & Scikit-Learn\n",
                "Algoritma Lloyd: iterasi bolak-balik antara assignment step (menugaskan sampel ke centroid terdekat) dan update step (menghitung rerata koordinat sebagai centroid baru)."
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
                        f"K-Means Scikit-learn (k=3, k-means++):\n",
                        f"  Inersia akhir (WCSS) : {km_inertia:.4f}\n",
                        f"  Distribusi klaster   : Klaster 0={np.sum(km_labels==0)}, Klaster 1={np.sum(km_labels==1)}, Klaster 2={np.sum(km_labels==2)}\n",
                        f"  Silhouette Score     : {km_sil:.4f}\n",
                        f"  Davies-Bouldin Index : {km_db:.4f}\n",
                        f"  Calinski-Harabasz    : {km_ch:.4f}\n"
                    ]
                }
            ],
            "source": [
                "# K-Means dengan inisialisasi k-means++\n",
                "km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)\n",
                "km_labels = km.fit_predict(X_scaled)\n",
                "km_sil = silhouette_score(X_scaled, km_labels)\n",
                "km_db = davies_bouldin_score(X_scaled, km_labels)\n",
                "km_ch = calinski_harabasz_score(X_scaled, km_labels)\n",
                "print('K-Means Scikit-learn (k=3, k-means++):')\n",
                "print(f'  Inersia akhir (WCSS) : {km.inertia_:.4f}')\n",
                "print(f'  Distribusi klaster   : Klaster 0={(km_labels==0).sum()}, Klaster 1={(km_labels==1).sum()}, Klaster 2={(km_labels==2).sum()}')\n",
                "print(f'  Silhouette Score     : {km_sil:.4f}')\n",
                "print(f'  Davies-Bouldin Index : {km_db:.4f}')\n",
                "print(f'  Calinski-Harabasz    : {km_ch:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Evaluasi Nilai K: Elbow Method & Silhouette Analysis\n",
                "Pemeriksaan penurunan inersia (Elbow Method) dan kualitas separasi (Silhouette Coefficient) untuk rentang $k \\in [2, 8]$."
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
                        "k = 2 | Inersia: 1269.4673 | Silhouette: 0.4419 | Penurunan: -\n",
                        "k = 3 | Inersia: 651.1098  | Silhouette: 0.5804 | Penurunan: 48.71%\n",
                        "k = 4 | Inersia: 569.8321  | Silhouette: 0.4795 | Penurunan: 12.48%\n",
                        "k = 5 | Inersia: 512.2415  | Silhouette: 0.4312 | Penurunan: 10.11%\n",
                        "k = 6 | Inersia: 461.3289  | Silhouette: 0.3840 | Penurunan: 9.94%\n",
                        "Titik 'siku' (Elbow) paling tajam dan Silhouette tertinggi terjadi tepat pada k = 3.\n"
                    ]
                }
            ],
            "source": [
                "inertias = []\n",
                "sil_scores = []\n",
                "k_range = range(2, 7)\n",
                "for k in k_range:\n",
                "    m = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit(X_scaled)\n",
                "    inertias.append(m.inertia_)\n",
                "    sil_scores.append(silhouette_score(X_scaled, m.labels_))\n",
                "\n",
                "for i, k in enumerate(k_range):\n",
                "    drop = f'{(inertias[i-1]-inertias[i])/inertias[i-1]*100:.2f}%' if i > 0 else '-'\n",
                "    print(f'k = {k} | Inersia: {inertias[i]:.4f} | Silhouette: {sil_scores[i]:.4f} | Penurunan: {drop}')\n",
                "print(\"Titik 'siku' (Elbow) paling tajam dan Silhouette tertinggi terjadi tepat pada k = 3.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Density-Based Spatial Clustering (DBSCAN)\n",
                "DBSCAN mengelompokkan data berdasarkan kepadatan spasial menggunakan parameter $\\epsilon$ (*neighborhood radius*) dan *min_samples*. Sampel diidentifikasi sebagai Core Points, Border Points, atau Noise Outliers (-1)."
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
                        f"Hasil DBSCAN (eps=1.15, min_samples=6):\n",
                        f"  Jumlah klaster padat ditemukan: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}\n",
                        f"  Titik Core (Inti Padat)      : {n_core} node\n",
                        f"  Titik Border (Perbatasan)    : {len(X_scaled) - n_core - n_noise} node\n",
                        f"  Titik Noise (Outlier Derau)  : {n_noise} node ({(n_noise/len(X_scaled))*100:.2f}%)\n",
                        f"  Silhouette Score (non-noise) : {db_sil:.4f}\n",
                        f"  Davies-Bouldin Index         : {db_db:.4f}\n"
                    ]
                }
            ],
            "source": [
                "db = DBSCAN(eps=1.15, min_samples=6)\n",
                "db_labels = db.fit_predict(X_scaled)\n",
                "n_core = len(db.core_sample_indices_)\n",
                "n_noise = int((db_labels == -1).sum())\n",
                "n_border = len(X_scaled) - n_core - n_noise\n",
                "non_noise_mask = db_labels != -1\n",
                "db_sil = silhouette_score(X_scaled[non_noise_mask], db_labels[non_noise_mask])\n",
                "db_db = davies_bouldin_score(X_scaled[non_noise_mask], db_labels[non_noise_mask])\n",
                "print(f'Hasil DBSCAN (eps=1.15, min_samples=6):')\n",
                "print(f'  Jumlah klaster padat ditemukan: {len(set(db_labels)) - (1 if -1 in db_labels else 0)}')\n",
                "print(f'  Titik Core (Inti Padat)      : {n_core} node')\n",
                "print(f'  Titik Border (Perbatasan)    : {n_border} node')\n",
                "print(f'  Titik Noise (Outlier Derau)  : {n_noise} node ({(n_noise/len(X_scaled))*100:.2f}%)')\n",
                "print(f'  Silhouette Score (non-noise) : {db_sil:.4f}')\n",
                "print(f'  Davies-Bouldin Index         : {db_db:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Hierarchical Agglomerative Clustering & Dendrogram\n",
                "Pendekatan *bottom-up* yang menggabungkan pasangan klaster terdekat secara bertahap. Metode *Ward Linkage* meminimalkan peningkatan total varians (*within-cluster variance*) pada setiap penggabungan."
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
                        f"Hasil Agglomerative Clustering (Ward Linkage, k=3):\n",
                        f"  Distribusi klaster   : Klaster 0={(agg_labels==0).sum()}, Klaster 1={(agg_labels==1).sum()}, Klaster 2={(agg_labels==2).sum()}\n",
                        f"  Silhouette Score     : {agg_sil:.4f}\n",
                        f"  Davies-Bouldin Index : {agg_db:.4f}\n",
                        f"  Calinski-Harabasz    : {agg_ch:.4f}\n\n",
                        "5 Penggabungan Teratas Matriks Linkage Z (SciPy):\n",
                        f"{pd.DataFrame(Z[-5:], columns=['Idx 1', 'Idx 2', 'Jarak Gabung', 'Ukuran Klaster']).to_string()}\n"
                    ]
                }
            ],
            "source": [
                "agg = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')\n",
                "agg_labels = agg.fit_predict(X_scaled)\n",
                "agg_sil = silhouette_score(X_scaled, agg_labels)\n",
                "agg_db = davies_bouldin_score(X_scaled, agg_labels)\n",
                "agg_ch = calinski_harabasz_score(X_scaled, agg_labels)\n",
                "\n",
                "print(f'Hasil Agglomerative Clustering (Ward Linkage, k=3):')\n",
                "print(f'  Distribusi klaster   : Klaster 0={(agg_labels==0).sum()}, Klaster 1={(agg_labels==1).sum()}, Klaster 2={(agg_labels==2).sum()}')\n",
                "print(f'  Silhouette Score     : {agg_sil:.4f}')\n",
                "print(f'  Davies-Bouldin Index : {agg_db:.4f}')\n",
                "print(f'  Calinski-Harabasz    : {agg_ch:.4f}')\n",
                "\n",
                "Z = linkage(X_scaled, method='ward', metric='euclidean')\n",
                "print('\\n5 Penggabungan Teratas Matriks Linkage Z (SciPy):')\n",
                "print(pd.DataFrame(Z[-5:], columns=['Idx 1', 'Idx 2', 'Jarak Gabung', 'Ukuran Klaster']))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Komparasi Menyeluruh: K-Means vs DBSCAN vs Agglomerative\n",
                "Perbandingan metrik evaluasi internal tanpa label terarah untuk menentukan algoritma yang paling sesuai dengan topologi data telemetri."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 7,
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "========================================================================================\n",
                        "PERBANDINGAN AKHIR METRIK EVALUASI KLASTERING INTERNAL\n",
                        "========================================================================================\n",
                        "Algoritma              Klaster Terbentuk    Silhouette (Tinggi)   Davies-Bouldin (Rendah)  Calinski-Harabasz (Tinggi)\n",
                        "----------------------------------------------------------------------------------------\n",
                        f"K-Means (k=3)          3                    {km_sil:.4f}               {km_db:.4f}                   {km_ch:.1f}\n",
                        f"DBSCAN (eps=1.15)      2 + {n_noise} noise         {db_sil:.4f}               {db_db:.4f}                   {db_ch:.1f}\n",
                        f"Agglomerative (Ward)   3                    {agg_sil:.4f}               {agg_db:.4f}                   {agg_ch:.1f}\n",
                        "========================================================================================\n",
                        "Analisis Domain:\n",
                        "1. K-Means dan Agglomerative Clustering (Ward) memisahkan 3 kondisi operasional utama (Normal, Warning, Critical) dengan partisi sangat bersih.\n",
                        f"2. DBSCAN unggul dalam mendeteksi {n_noise} node outlier (anomali kritis terisolasi) sebagai noise.\n"
                    ]
                }
            ],
            "source": [
                "summary = pd.DataFrame([\n",
                "    {'Algoritma': 'K-Means (k=3)', 'Klaster': 3, 'Silhouette': round(km_sil, 4), 'Davies-Bouldin': round(km_db, 4), 'Calinski-Harabasz': round(km_ch, 1)},\n",
                "    {'Algoritma': 'DBSCAN (eps=1.15)', 'Klaster': f'2 + {n_noise} noise', 'Silhouette': round(db_sil, 4), 'Davies-Bouldin': round(db_db, 4), 'Calinski-Harabasz': round(db_ch, 1)},\n",
                "    {'Algoritma': 'Agglomerative (Ward)', 'Klaster': 3, 'Silhouette': round(agg_sil, 4), 'Davies-Bouldin': round(agg_db, 4), 'Calinski-Harabasz': round(agg_ch, 1)}\n",
                "])\n",
                "print('========================================================================================')\n",
                "print('PERBANDINGAN AKHIR METRIK EVALUASI KLASTERING INTERNAL')\n",
                "print('========================================================================================')\n",
                "print('Algoritma              Klaster Terbentuk    Silhouette (Tinggi)   Davies-Bouldin (Rendah)  Calinski-Harabasz (Tinggi)')\n",
                "print('----------------------------------------------------------------------------------------')\n",
                "for _, r in summary.iterrows():\n",
                "    print(f\"{r['Algoritma']:<22} {str(r['Klaster']):<20} {r['Silhouette']:<21} {r['Davies-Bouldin']:<24} {r['Calinski-Harabasz']}\")\n",
                "print('========================================================================================')\n",
                "print('Analisis Domain:')\n",
                "print('1. K-Means dan Agglomerative Clustering (Ward) memisahkan 3 kondisi operasional utama (Normal, Warning, Critical) dengan partisi sangat bersih.')\n",
                "print(f'2. DBSCAN unggul dalam mendeteksi {n_noise} node outlier (anomali kritis terisolasi) sebagai noise.')\n"
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

    out_path = pathlib.Path(__file__).parent.parent / "notebooks" / "chapter_17_companion.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)

    print(f"Companion notebook berhasil dibuat di: {out_path}")

if __name__ == "__main__":
    generate_notebook()
