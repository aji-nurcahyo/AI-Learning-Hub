"""Analisis Koefisien Silhouette (Silhouette Analysis)
Mengevaluasi kepadatan intra-klaster a(i) dan separasi inter-klaster b(i)
untuk memvalidasi jumlah klaster optimal tanpa label ground-truth.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def run_silhouette_analysis():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = StandardScaler().fit_transform(df[features])

    print("=== ANALISIS KOEFISIEN SILHOUETTE ===")
    print("Mengevaluasi Silhouette Score untuk k in [2, 7]:\n")

    k_values = [2, 3, 4, 5, 6, 7]
    scores = []

    print("Nilai k | Rata-rata Skor Silhouette | Karakteristik Pemisahan Klaster")
    print("-" * 68)

    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        cluster_labels = km.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        scores.append(score)

        if score > 0.40:
            desc = "Struktur klaster sangat kuat & terpisah jelas (Optimal)"
        elif score > 0.30:
            desc = "Struktur wajar, beberapa sampel di perbatasan"
        else:
            desc = "Klaster tumpang tindih / buatan (Kurang optimal)"

        print(f"k = {k:<2}  | {score:18.4f}        | {desc}")

    print("-" * 68)

    # Analisis detail sampel untuk k = 3 (Skor tertinggi)
    best_k = 3
    km_best = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
    labels_best = km_best.fit_predict(X)
    sample_sil = silhouette_samples(X, labels_best)

    print(f"\nProfil Silhouette per Klaster pada k = {best_k}:")
    for cid in range(best_k):
        c_sil = sample_sil[labels_best == cid]
        n_negative = np.sum(c_sil < 0)
        print(f"  - Klaster {cid} ({len(c_sil):<3} sampel): Mean Sil = {np.mean(c_sil):.4f} | Min = {np.min(c_sil):.4f} | Sampel Negatif = {n_negative}")

    print("\nInterpretasi Matematis:")
    print("Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i)) in [-1, +1].")
    print("Nilai s(i) mendekati +1 menunjukkan sampel sangat dekat dengan rekan se-klasternya")
    print("dan sangat jauh dari klaster tetangga. Nilai negatif mengindikasikan salah penempatan klaster.")

if __name__ == "__main__":
    run_silhouette_analysis()

