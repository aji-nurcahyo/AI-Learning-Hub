# Panduan Praktikum Terbimbing: Unsupervised Learning & Clustering Algorithms

Selamat datang di laboratorium praktikum Chapter 17. Modul praktikum ini dirancang untuk memberikan pengalaman komputasional langsung dalam mengimplementasikan, memvisualisasikan, dan mengevaluasi algoritma *unsupervised learning* (pembelajaran tanpa supervisi) pada dataset telemetri jaringan sensor nirkabel (*Wireless Sensor Network*).

---

## 1. Lingkungan Kerja & Prapemrosesan

Pastikan lingkungan virtual Python Anda telah terkonfigurasi dengan pustaka saintifik standar:

```bash
# Mengaktifkan environment (PowerShell)
.venv\Scripts\Activate.ps1

# Menjalankan script pembangkit dataset (jika belum dijalankan)
python content/level-2-machine-learning/chapter-17/code/generate_dataset.py
```

Dataset tersimpan di `content/level-2-machine-learning/chapter-17/code/data/wireless_sensor_telemetry.csv` dengan struktur:
- **Jumlah Sampel**: 360 baris observasi node sensor IoT.
- **Fitur Kontinu**: `signal_strength_dbm`, `packet_loss_pct`, `latency_ms`, `temperature_c`, `voltage_v`, `energy_consumption_mah`.
- **Kolom Laten Referensi**: `latent_profile` (hanya digunakan untuk verifikasi eksternal Adjusted Rand Index, tidak diikutsertakan saat melatih model).

---

## 2. Katalog Praktikum Terstruktur (12 Langkah Mandiri)

Praktikum ini terbagi menjadi 12 modul skrip terisolasi yang dapat dijalankan secara berurutan:

```
content/level-2-machine-learning/chapter-17/code/
├── generate_dataset.py            # Pembangkit 360 sampel data multivariat
├── clustering_foundations.py      # Eksplorasi penskalaan & matriks jarak pairwise
├── kmeans_from_scratch.py         # Algoritma Lloyd murni berbasis NumPy
├── kmeans_sklearn.py              # K-Means industri dengan Scikit-Learn
├── kmeans_initialization.py       # Eksperimen 20 kali Random vs K-Means++
├── elbow_method.py                # Kurva penurunan inersia (k = 1..9)
├── silhouette_analysis.py         # Koefisien silhouette & profil per klaster
├── dbscan_clustering.py           # Density clustering (Core, Border, Noise)
├── dbscan_parameter_analysis.py   # Grid search parameter epsilon & min_samples
├── hierarchical_clustering.py     # Perbandingan keterkaitan (Ward, Complete, dll)
├── dendrogram_visualization.py    # Visualisasi pohon hirarkis SciPy
├── clustering_evaluation.py       # Evaluasi internal (Silhouette, DBI, CHI)
└── clustering_comparison.py       # Benchmark komprehensif seluruh algoritma
```

---

## 3. Langkah-demi-Langkah Pelaksanaan Praktikum

### Langkah 1: Eksplorasi Jarak & Kebutuhan Standarisasi
Jalankan skrip dasar untuk memahami bagaimana perbedaan skala fitur mendistorsi jarak geometris:
```bash
python content/level-2-machine-learning/chapter-17/code/clustering_foundations.py
```
*Observasi Kunci*: Jarak Euclidean mentah didominasi oleh variabel berorde ratusan (energi mAh), sedangkan fitur berorde satuan kecil (voltase) tereduksi pengaruhnya menjadi mendekati nol. Standarisasi $Z$-score menghasilkan bobot geometris yang seimbang.

### Langkah 2: K-Means dari Dasar (Pure NumPy)
Pelajari mekanisme iterasi algoritma Lloyd tanpa bantuan pustaka tingkat tinggi:
```bash
python content/level-2-machine-learning/chapter-17/code/kmeans_from_scratch.py
```
*Observasi Kunci*: Verifikasi bahwa inersia akhir implementasi NumPy murni identik dengan `sklearn.cluster.KMeans` (perbedaan $< 10^{-3}$).

### Langkah 3: K-Means Tingkat Produksi dengan Scikit-Learn
Eksekusi K-Means dengan inisialisasi cerdas dan profil sentroid:
```bash
python content/level-2-machine-learning/chapter-17/code/kmeans_sklearn.py
```
*Observasi Kunci*: Catat nilai inersia akhir pada $k=3$ sebesar **651.1098** dan distribusi anggota: Klaster 0 (143 node), Klaster 1 (118 node), Klaster 2 (99 node).

### Langkah 4: Eksperimen Inisialisasi (Random vs K-Means++)
Uji stabilitas konvergensi algoritma melalui 20 kali pengulangan independen:
```bash
python content/level-2-machine-learning/chapter-17/code/kmeans_initialization.py
```
*Observasi Kunci*: Inisialisasi acak (*random*) memiliki varians inersia tinggi dan risiko terjebak pada minimum lokal buruk. K-Means++ menghasilkan inersia stabil yang konsisten pada nilai global minimum.

### Langkah 5: Penentuan K Optimal dengan Elbow Method
Hitung kurva penurunan inersia untuk nilai $k \in [1, 9]$:
```bash
python content/level-2-machine-learning/chapter-17/code/elbow_method.py
```
*Observasi Kunci*: Terjadi penurunan drastis inersia saat berpindah dari $k=2$ (1269.47) ke $k=3$ (651.11, penurunan 48.71%), dan melandai tajam setelah $k=3$ (hanya turun 12.48% ke $k=4$).

### Langkah 6: Analisis Siluet (Silhouette Analysis)
Hitung koefisien siluet individu untuk mengukur kerapatan intra-klaster vs keterpisahan inter-klaster:
```bash
python content/level-2-machine-learning/chapter-17/code/silhouette_analysis.py
```
*Observasi Kunci*: Skor siluet tertinggi dicapai pada $k=3$ (0.5804), dengan 0 sampel bernilai siluet negatif.

### Langkah 7: Pengelompokan Berbasis Densitas (DBSCAN)
Kelompokkan data berdasarkan kerapatan spasial dan identifikasi derau:
```bash
python content/level-2-machine-learning/chapter-17/code/dbscan_clustering.py
```
*Observasi Kunci*: Pada $\epsilon=1.15$ dan $\text{min\_samples}=6$, DBSCAN menemukan **308 Core Points**, **27 Border Points**, dan **25 Noise Outliers** (label `-1`).

### Langkah 8: Analisis Sensitivitas Hyperparameter DBSCAN
Lakukan pencarian kisi (*grid search*) parameter $\epsilon$ dan $\text{min\_samples}$:
```bash
python content/level-2-machine-learning/chapter-17/code/dbscan_parameter_analysis.py
```
*Observasi Kunci*: Jika $\epsilon$ terlalu kecil ($< 0.8$), mayoritas sampel terdeteksi sebagai noise. Jika $\epsilon$ terlalu besar ($> 1.8$), seluruh klaster menyatu menjadi satu super-klaster tunggal.

### Langkah 9: Agglomerative Hierarchical Clustering
Bandingkan 4 metode keterkaitan (*linkage methods*):
```bash
python content/level-2-machine-learning/chapter-17/code/hierarchical_clustering.py
```
*Observasi Kunci*: Ward Linkage menghasilkan skor siluet tertinggi (0.5739) dan ukuran klaster paling berimbang, mengungguli Single, Complete, dan Average Linkage.

### Langkah 10: Visualisasi Dendrogram & Pemotongan Pohon
Analisis matriks linkage SciPy dan ambang batas pemotongan:
```bash
python content/level-2-machine-learning/chapter-17/code/dendrogram_visualization.py
```
*Observasi Kunci*: Tiga klaster utama terbentuk pada jarak disimilaritas penggabungan di atas 18.0.

### Langkah 11: Evaluasi Internal Komparatif
Hitung ketiga metrik evaluasi internal standar industri:
```bash
python content/level-2-machine-learning/chapter-17/code/clustering_evaluation.py
```
*Observasi Kunci*: Lakukan triangulasi nilai Silhouette, Davies-Bouldin Index, dan Calinski-Harabasz Index.

### Langkah 12: Benchmark Komprehensif Seluruh Algoritma
Jalankan skrip perbandingan akhir untuk merangkum seluruh hasil praktikum:
```bash
python content/level-2-machine-learning/chapter-17/code/clustering_comparison.py
```

---

## 4. Tabel Hasil Pengujian Empiris Terverifikasi

Gunakan tabel hasil terverifikasi berikut sebagai acuan validasi eksekusi praktikum Anda:

| Algoritma | Parameter Konfigurasi | Jumlah Klaster | Silhouette Score (Tinggi) | Davies-Bouldin (Rendah) | Calinski-Harabasz (Tinggi) | Sampel Noise |
|---|---|---|---|---|---|---|
| **K-Means** | $k=3$, `k-means++` | 3 | **0.5804** | **0.6419** | **291.8** | 0 |
| **DBSCAN** | $\epsilon=1.15, \text{min}=6$ | 2 (+ noise) | **0.6538** (non-noise) | **0.4907** (non-noise) | **263.4** | **25 (6.94%)** |
| **Agglomerative (Ward)** | $k=3$, Euclidean | 3 | **0.5739** | **0.6548** | **286.3** | 0 |
| **Agglomerative (Complete)** | $k=3$, Euclidean | 3 | 0.5218 | 0.7412 | 231.5 | 0 |
| **Agglomerative (Average)** | $k=3$, Euclidean | 3 | 0.5482 | 0.6931 | 254.1 | 0 |
| **Agglomerative (Single)** | $k=3$, Euclidean | 3 | 0.3120 | 1.1205 | 114.7 | 0 |

---

## 5. Tantangan Terbimbing (*Guided Challenges*)

### Tantangan A: Menguji Dampak Penskalaan Min-Max vs Z-Score
Ubah skrip `clustering_foundations.py` untuk menggunakan `MinMaxScaler` alih-alih `StandardScaler`. Apakah Elbow Method pada K-Means tetap mengidentifikasi $k=3$? Mengapa $Z$-Score lebih direkomendasikan pada data sensor yang memiliki sebaran Gaussian alami?

### Tantangan B: Deteksi Outlier Ekstrem Berbasis DBSCAN
Gunakan label noise (`-1`) dari DBSCAN pada `dbscan_clustering.py` untuk mengekstrak 25 node sensor paling bermasalah. Tampilkan statistik deskriptif dari ke-25 node tersebut (misal nilai rata-rata packet loss dan latensinya). Analisis apakah node-node ini merepresentasikan kondisi kegagalan perangkat keras kritis.

### Tantangan C: Menentukan Nilai Epsilon dengan Metode K-Distance Graph
Tulis fungsi Python pendek untuk menghitung jarak ke tetangga ke-6 terdekat (`NearestNeighbors(n_neighbors=6)`), urutkan hasilnya dari kecil ke besar, dan tentukan secara visual di manakah letak titik siku (*knee point*) kurva jarak. Apakah nilai tersebut bersesuaian dengan $\epsilon=1.15$?

---

## 6. Rubrik Penilaian Praktikum (Skala 100 Poin)

| Kriteria Penilaian | Bobot | Sangat Baik (85–100) | Baik (70–84) | Cukup (55–69) | Kurang (< 55) |
|---|---|---|---|---|---|
| **Eksekusi & Integritas Kode** | 25% | Seluruh 12 skrip dieksekusi tanpa galat; output numerik presisi sesuai tabel verifikasi. | 10–11 skrip berjalan lancar dengan sedikit variasi numerik yang dapat dijelaskan. | 7–9 skrip berjalan; terdapat galat runtime yang belum ditangani tuntas. | Kurang dari 7 skrip berjalan; manipulasi data salah. |
| **Analisis Matematis & Algoritma** | 25% | Mampu menjelaskan formulasi Lloyd, inersia WCSS, kriteria Ward, dan keterjangkauan kepadatan DBSCAN secara mendalam. | Memahami alur kerja algoritma secara umum namun kurang presisi dalam formulasi matematis. | Hanya mengulang definisi tanpa analisis hubungan antar-konsep. | Tidak memahami prinsip dasar clustering. |
| **Evaluasi Internal & Triangulasi Metrik** | 25% | Menginterpretasikan Silhouette, Davies-Bouldin, dan Calinski-Harabasz secara komprehensif tanpa salah kaprah. | Menjelaskan arti metrik namun hanya bertumpu pada satu indikator (misal Silhouette saja). | Menyajikan angka metrik tanpa interpretasi makna fisis/kualitas klaster. | Salah mengartikan arah kebaikan metrik (misal menganggap DBI tinggi itu baik). |
| **Penyelesaian Kasus & Tantangan** | 25% | Menyelesaikan ketiga tantangan terbimbing dengan kode bersih, metodologi tepat, dan interpretasi domain IoT yang kuat. | Menyelesaikan 2 tantangan dengan benar atau 3 tantangan dengan analisis terbatas. | Menyelesaikan 1 tantangan dengan benar. | Tidak menyelesaikan tantangan atau kode tidak dapat dieksekusi. |

