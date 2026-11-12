# Katalog Praktikum & Eksperimen Kode — Chapter 17: Unsupervised Learning & Clustering

Direktori ini memuat seluruh implementasi kode Python mandiri (*runnable*), skrip pembangkit data deterministik segmentasi sensor nirkabel (*Wireless Sensor Networks* / WSN), implementasi K-Means murni dari dasar (*from scratch*), metode Elbow, analisis Silhouette, algoritma berbasis kepadatan DBSCAN, klasterisasi hirarki aglomeratif dengan dendrogram, dan benchmark evaluasi komprehensif.

---

## 1. Lingkungan Eksekusi & Kebutuhan Dependensi

Seluruh skrip dikembangkan dan diverifikasi menggunakan Python 3.12 pada lingkungan virtual `.venv`:

```bash
# Aktifkan virtual environment
.\.venv\Scripts\activate

# Dependensi utama (terinstal):
# - numpy >= 1.26.0
# - pandas >= 2.2.0
# - scipy >= 1.12.0
# - scikit-learn >= 1.4.0
# - matplotlib >= 3.8.0
```

---

## 2. Struktur Berkas & Deskripsi Modul

| Berkas Skrip | Deskripsi Algoritmik & Pembelajaran | Keluaran Utama / Metrik |
|---|---|---|
| `generate_dataset.py` | Pembangkit data sintetis telemetri sensor nirkabel deterministik (360 baris) | `data/wireless_sensor_telemetry.csv` (3 klaster alamiah + noise) |
| `clustering_foundations.py` | Perbedaan Supervised vs Unsupervised, kalkulasi matriks jarak pasangan | Matriks jarak Euclidean 4x4, bukti kebutuhan feature scaling |
| `kmeans_from_scratch.py` | Implementasi K-Means murni NumPy (assignment step, centroid update, WCSS) | Verifikasi identik 100% konvergensi dengan Scikit-learn API |
| `kmeans_sklearn.py` | Scikit-learn KMeans produksi (`k-means++`, $n_{\text{init}}=10$), interpretasi centroid | Koordinat centroid dalam satuan fisik asli, deskripsi domain |
| `kmeans_initialization.py` | Analisis komparatif Random vs K-Means++ pada 20 uji independen | Bukti penekanan risiko optimum lokal oleh K-Means++ |
| `elbow_method.py` | Evaluasi laju penurunan Inertia (WCSS) untuk $k \in [1, 9]$ | Identifikasi titik belok siku (*elbow point*) optimal pada $k=3$ |
| `silhouette_analysis.py` | Formulasi koefisien Silhouette $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ untuk $k \in [2, 7]$ | Skor Silhouette rata-rata & sebaran nilai per klaster |
| `dbscan_clustering.py` | Algoritma kepadatan DBSCAN (`eps=1.15`, `min_samples=6`) | Klasifikasi titik Core vs Border vs Noise (label -1) |
| `dbscan_parameter_analysis.py` | Eksplorasi grid parameter `eps` dan `min_samples` | Analisis proporsi titik derau dan pembentukan klaster |
| `hierarchical_clustering.py` | Klasterisasi hirarki aglomeratif (*bottom-up*) lintas 4 metode linkage | Evaluasi Ward, Complete, Average, Single (chaining effect) |
| `dendrogram_visualization.py` | Rekonstruksi pohon hirarki linkage SciPy dan analisis pemotongan (*cutting*) | Riwayat 5 penggabungan teratas, ketinggian jarak penggabungan |
| `clustering_evaluation.py` | Evaluasi metrik validasi internal (Silhouette, Davies-Bouldin, Calinski-Harabasz) | Komparasi metrik internal tanpa ground-truth vs metrik eksternal (ARI) |
| `clustering_comparison.py` | Benchmark komparatif 4 arsitektur klasterisasi pada telemetri sensor | Tabel perbandingan jumlah klaster, noise, Silhouette, DB, dan waktu |

---

## 3. Dataset Telemetri Segmentasi Sensor Nirkabel

Dataset disimpan pada `data/wireless_sensor_telemetry.csv` dengan metadata:
- **Jumlah Baris**: 360 sampel telemetri simpul sensor nirkabel.
- **Fitur Masukan (6 Prediktor)**:
  1. `signal_strength_dbm`: Kekuatan sinyal radio RSSI (-85.0 s/d -35.0 dBm).
  2. `packet_loss_pct`: Persentase paket data yang hilang (0.0% s/d 35.0%).
  3. `latency_ms`: Waktu tunda transmisi (*Round Trip Time*) (10.0 s/d 400.0 ms).
  4. `temperature_c`: Suhu operasional simpul sensor (15.0 s/d 70.0 °C).
  5. `voltage_v`: Tegangan catu daya baterai lithium (2.0 s/d 3.8 V).
  6. `energy_consumption_mah`: Konsumsi energi kumulatif per siklus (5.0 s/d 200.0 mAh).
- **Sifat Data**: Unsupervised (Label kelas tidak pernah diberikan kepada algoritma klasterisasi).

---

## 4. Cara Menjalankan Skrip

Jalankan skrip secara individual melalui terminal:

```bash
# Menjalankan K-Means dari dasar (from scratch)
python content/level-2-machine-learning/chapter-17/code/kmeans_from_scratch.py

# Menjalankan analisis Silhouette
python content/level-2-machine-learning/chapter-17/code/silhouette_analysis.py

# Menjalankan perbandingan metode klasterisasi
python content/level-2-machine-learning/chapter-17/code/clustering_comparison.py
```

