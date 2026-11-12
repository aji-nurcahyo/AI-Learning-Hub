"""Benchmark Komparatif Menyeluruh Metode Reduksi Dimensi
Membandingkan PCA, TruncatedSVD, t-SNE, dan UMAP secara empiris dan teoretis.
"""

import pathlib
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

def compare_all_methods():
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

    # 1. PCA (k=2)
    t0 = time.perf_counter()
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(X_scaled)
    t_pca = (time.perf_counter() - t0) * 1000

    # 2. TruncatedSVD (k=2)
    t0 = time.perf_counter()
    tsvd = TruncatedSVD(n_components=2, random_state=42)
    Z_tsvd = tsvd.fit_transform(X_scaled)
    t_tsvd = (time.perf_counter() - t0) * 1000

    # 3. t-SNE (k=2, perplexity=30)
    t0 = time.perf_counter()
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    Z_tsne = tsne.fit_transform(X_scaled)
    t_tsne = (time.perf_counter() - t0) * 1000

    # 4. UMAP (k=2, conditional)
    t_umap = None
    umap_status = "Tersedia"
    try:
        import umap
        t0 = time.perf_counter()
        um = umap.UMAP(n_components=2, random_state=42)
        Z_um = um.fit_transform(X_scaled)
        t_umap = (time.perf_counter() - t0) * 1000
    except ImportError:
        umap_status = "Opsional (Belum Terpasang)"

    print("=========================================================================================")
    print("BENCHMARK KOMPARATIF ALGORITMA REDUKSI DIMENSI (DIMENSIONALITY REDUCTION)")
    print("=========================================================================================")
    print(f"{'Metode':<16} {'Tipe':<12} {'Waktu (ms)':<14} {'Struktur':<16} {'Invertible?':<14} {'Out-of-Sample?'}")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'PCA':<16} {'Linier':<12} {t_pca:<14.2f} {'Global':<16} {'YA':<14} {'YA (.transform)'}")
    print(f"{'TruncatedSVD':<16} {'Linier':<12} {t_tsvd:<14.2f} {'Global':<16} {'YA':<14} {'YA (.transform)'}")
    print(f"{'t-SNE':<16} {'Non-Linier':<12} {t_tsne:<14.2f} {'Lokal':<16} {'TIDAK':<14} {'TIDAK (Transduktif)'}")

    if t_umap is not None:
        print(f"{'UMAP':<16} {'Non-Linier':<12} {t_umap:<14.2f} {'Lokal & Global':<16} {'YA (Aproksimasi)':<14} {'YA (.transform)'}")
    else:
        print(f"{'UMAP':<16} {'Non-Linier':<12} {'[Opsional]':<14} {'Lokal & Global':<16} {'YA (Aproksimasi)':<14} {'YA (.transform)'}")

    print("=========================================================================================")
    print("PANDUAN SELEKSI REKAYASA SISTEM:")
    print("1. Gunakan PCA: Ketika membutuhkan kompresi fitur cepat, interpretable, menjaga varians global,")
    print("   atau sebagai tahap prapemrosesan sebelum model regresi/klasifikasi.")
    print("2. Gunakan TruncatedSVD: Khusus untuk data berdimensi sangat besar dengan matriks jarang (sparse),")
    print("   seperti pemrosesan teks (TF-IDF) atau sistem rekomendasi interaksi pengguna.")
    print("3. Gunakan t-SNE: Eksklusif untuk eksplorasi visual 2D/3D struktur klaster lokal yang rumit.")
    print("4. Gunakan UMAP: Ketika membutuhkan visualisasi berkualitas tinggi yang mempertahankan struktur")
    print("   global sekaligus lokal dengan performa komputasi jauh lebih cepat daripada t-SNE.")

if __name__ == "__main__":
    compare_all_methods()

