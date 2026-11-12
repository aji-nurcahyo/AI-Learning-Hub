# Panduan Praktikum Terbimbing: Dimensionality Reduction

Selamat datang di laboratorium praktikum Chapter 18. Modul praktikum ini dirancang untuk memberikan pemahaman empiris mendalam mengenai teknik reduksi dimensi linier dan non-linier. Anda akan menguji fenomena kutukan dimensi, menurunkan PCA dari dasar berbasis aljabar linear NumPy, membuktikan Teorema Eckart-Young-Mirsky melalui SVD, membangun machine learning pipeline bebas kebocoran data, serta mengeksplorasi manifold learning non-linier dengan t-SNE dan UMAP.

---

## 1. Lingkungan Kerja & Dataset

Pastikan lingkungan virtual Python Anda telah terkonfigurasi dengan pustaka saintifik standar:

```bash
# Mengaktifkan environment virtual (PowerShell)
.venv\Scripts\Activate.ps1
```

Dataset yang digunakan adalah `content/level-2-machine-learning/chapter-18/code/data/wireless_sensor_telemetry.csv` dengan struktur:
- **Jumlah Sampel**: 360 observasi node sensor jaringan IoT.
- **Fitur Kontinu (6 Dimensi)**: `signal_strength_dbm`, `packet_loss_pct`, `latency_ms`, `temperature_c`, `voltage_v`, `energy_consumption_mah`.
- **Target Profil Laten**: `latent_profile` (digunakan dalam evaluasi pipeline klasifikasi).

---

## 2. Katalog Praktikum Terstruktur (11 Skrip Mandiri)

```
content/level-2-machine-learning/chapter-18/code/
├── high_dimensional_data.py           # Eksplorasi fenomena kutukan dimensi
├── pca_from_scratch.py                 # Algoritma PCA dari dasar (NumPy murni)
├── pca_sklearn.py                      # Ekstraksi komponen, bobot pemuatan & interpretasi
├── explained_variance.py               # Scree plot, kriteria Kaiser & ambang batas varians
├── pca_reconstruction.py               # Inverse transform & galat rekonstruksi Frobenius
├── svd_decomposition.py                # SVD penuh & pembuktian Eckart-Young-Mirsky
├── truncated_svd.py                    # TruncatedSVD vs Standard PCA (efek pemusatan)
├── pca_pipeline.py                     # Pipeline ML bebas kebocoran data
├── tsne_visualization.py               # Manifold learning t-SNE & tuning perplexity
├── umap_visualization.py               # Eksplorasi UMAP (penanganan gracefully fallback)
└── dimensionality_comparison.py        # Benchmark komprehensif seluruh metode
```

---

## 3. Langkah-demi-Langkah Pelaksanaan Praktikum

### Langkah 1: Demonstrasi Fenomena Konsentrasi Jarak
Uji hilangnya diskriminasi metrik jarak saat dimensi meningkat dari 2 hingga 1000:
```bash
python content/level-2-machine-learning/chapter-18/code/high_dimensional_data.py
```
*Observasi Kunci*: Rasio jarak minimum terhadap jarak maksimum melompat dari 0.0021 pada $D=2$ menjadi 0.8682 pada $D=1000$.

### Langkah 2: Membangun PCA dari Dasar (Pure NumPy)
Pelajari dekomposisi nilai eigen dari matriks kovarians sampel:
```bash
python content/level-2-machine-learning/chapter-18/code/pca_from_scratch.py
```
*Observasi Kunci*: Verifikasi bahwa komponen dan koordinat proyeksi implementasi NumPy murni identik dengan Scikit-Learn dengan selisih maksimum $< 10^{-15}$.

### Langkah 3: Ekstraksi Bobot Pemuatan & Interpretasi Fisik
Ekstrak korelasi antar-variabel asli terhadap sumbu komponen utama:
```bash
python content/level-2-machine-learning/chapter-18/code/pca_sklearn.py
```
*Observasi Kunci*: PC1 merangkum 80.51% varians total dan merepresentasikan indeks degradasi global sistem (berkorelasi positif dengan latensi, packet loss, suhu, dan arus; negatif dengan daya sinyal dan voltase).

### Langkah 4: Analisis Scree Plot & Kriteria Pemilihan Komponen
Evaluasi penurunan varians dan kriteria penentuan $k$:
```bash
python content/level-2-machine-learning/chapter-18/code/explained_variance.py
```
*Observasi Kunci*: Kriteria Kaiser ($\lambda \ge 1.0$) merekomendasikan $k=1$. Ambang batas 95% varians dicapai pada $k=4$ (95.24%). Proyeksi 2D mempertahankan 87.45% varians.

### Langkah 5: Rekonstruksi Data & Galat Frobenius
Uji transformasi balik (*inverse transform*) dan kompresi *lossy*:
```bash
python content/level-2-machine-learning/chapter-18/code/pca_reconstruction.py
```
*Observasi Kunci*: Pada $k=6$, galat rekonstruksi adalah tepat 0.000000 (rekonstruksi sempurna / lossless). Pada $k=2$, norma Frobenius galat adalah 16.461525.

### Langkah 6: Dekomposisi SVD & Teorema Eckart-Young-Mirsky
Buktikan kesetaraan nilai singular SVD terhadap varians PCA dan batas galat aproksimasi rank rendah:
```bash
python content/level-2-machine-learning/chapter-18/code/svd_decomposition.py
```
*Observasi Kunci*: Galat empiris aproksimasi rank-$k$ persis menyentuh batas teoretis $\sqrt{\sum_{j=k+1}^D s_j^2}$ (selisih $0.00\text{e}+00$).

### Langkah 7: TruncatedSVD vs PCA (Pemusatan Data)
Pahami perbedaan mendasar dekomposisi data tanpa pemusatan vs dengan pemusatan:
```bash
python content/level-2-machine-learning/chapter-18/code/truncated_svd.py
```
*Observasi Kunci*: Pada data yang terstandarisasi/terpusat, TruncatedSVD menghasilkan proyeksi yang identik dengan PCA (selisih $< 10^{-14}$). TruncatedSVD digunakan khusus untuk matriks jarang (sparse) pada pemrosesan teks.

### Langkah 8: Pipeline Bebas Kebocoran Data (Zero Data Leakage)
Bandingkan performa klasifikasi model hilir dengan dan tanpa PCA:
```bash
python content/level-2-machine-learning/chapter-18/code/pca_pipeline.py
```
*Observasi Kunci*: Reduksi dari 6 fitur ke 2 komponen utama meningkatkan akurasi dari 94.44% menjadi **96.67%** (berkat eliminasi derau) dan mempercepat latensi latih hingga 50%.

### Langkah 9: Visualisasi Manifold Non-Linier t-SNE
Eksplorasi proyeksi 2D berbasis probabilitas ketetanggaan t-Student:
```bash
python content/level-2-machine-learning/chapter-18/code/tsne_visualization.py
```
*Observasi Kunci*: t-SNE dengan perplexity=30 menghasilkan visualisasi klaster lokal yang sangat terpisah bersih dengan KL Divergence akhir 0.5219.

### Langkah 10: Eksplorasi Algoritma UMAP
Pahami konsep teori himpunan fuzzy dan preservasi struktur global:
```bash
python content/level-2-machine-learning/chapter-18/code/umap_visualization.py
```
*Observasi Kunci*: Skrip menangani ketergantungan `umap-learn` secara tangguh dengan memberikan panduan instalasi dan sintesis teoretis komparatif.

### Langkah 11: Benchmark Komparatif Seluruh Metode
Bandingkan seluruh karakteristik metode dalam tabel terpadu:
```bash
python content/level-2-machine-learning/chapter-18/code/dimensionality_comparison.py
```

---

## 4. Tabel Hasil Pengujian Empiris Terverifikasi

| Pengujian / Metrik | Parameter Konfigurasi | Nilai Terverifikasi | Status Verifikasi |
|---|---|---|---|
| **Rasio Jarak (D=2)** | $N=200, D=2$ | **0.0021** (Sangat Tajam) | VALID |
| **Rasio Jarak (D=1000)** | $N=200, D=1000$ | **0.8682** (Konsentrasi Ekstrem) | VALID |
| **PCA Eigenvalue PC1** | $Z$-Score Terpusat | **4.8438** (80.51% Varians) | VALID |
| **PCA Eigenvalue PC2** | $Z$-Score Terpusat | **0.4181** (6.95% Varians) | VALID |
| **Varians Kumulatif (k=2)** | $k=2$ Komponen | **87.45%** | VALID |
| **Varians Kumulatif (k=4)** | $k=4$ Komponen | **95.24%** (Ambang 95%) | VALID |
| **Galat Frobenius (k=2)** | Inverse Transform $k=2$ | **16.461525** | VALID |
| **Galat Frobenius (k=6)** | Inverse Transform $k=6$ | **0.000000** (Lossless) | VALID |
| **Akurasi Raw (6 Fitur)** | Logistic Regression | **94.44%** | VALID |
| **Akurasi PCA (k=2)** | Pipeline Bebas Bocor | **96.67%** (Optimal) | VALID |
| **KL Divergence t-SNE** | Perplexity = 30 | **0.5219** | VALID |

---

## 5. Tantangan Terbimbing (*Guided Challenges*)

### Tantangan A: Menguji Dampak Pemusatan Data pada PCA
Modifikasi fungsi `PCAFromScratch` untuk melewatkan langkah pemusatan (`X_centered = X`). Amati apa yang terjadi pada vektor komponen utama pertama $\mathbf{w}_1$. Mengapa tanpa pemusatan, PC1 hanya merefleksikan rata-rata posisi data daripada sumbu varians terbesar?

### Tantangan B: Deteksi Anomali Berbasis Galat Rekonstruksi PCA
Gunakan model PCA yang dilatih dengan $k=2$. Hitung galat rekonstruksi kuadratik $\| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2$ untuk setiap sampel dalam dataset. Urutkan 10 sampel dengan galat rekonstruksi terbesar. Apakah sampel-sampel ini bersesuaian dengan 25 node outlier yang sebelumnya ditemukan oleh DBSCAN pada Chapter 17?

### Tantangan C: Eksperimen Sensitivitas Perplexity pada t-SNE
Jalankan `tsne_visualization.py` dengan perplexity ekstrem: $\text{perp} = 2$ dan $\text{perp} = 100$. Analisis mengapa perplexity yang terlalu kecil memecah klaster menjadi titik-titik terisolasi palsu, sedangkan perplexity yang terlalu besar menggumpalkan seluruh kelompok menjadi satu lingkaran seragam.

---

## 6. Rubrik Penilaian Praktikum (Skala 100 Poin)

| Kriteria Penilaian | Bobot | Sangat Baik (85–100) | Baik (70–84) | Cukup (55–69) | Kurang (< 55) |
|---|---|---|---|---|---|
| **Eksekusi & Integritas Kode** | 25% | Seluruh 11 skrip dieksekusi tanpa galat; output numerik presisi sesuai tabel verifikasi. | 9–10 skrip berjalan lancar dengan sedikit variasi numerik yang dapat dijelaskan. | 7–8 skrip berjalan; terdapat galat runtime yang belum ditangani tuntas. | Kurang dari 7 skrip berjalan; manipulasi data salah. |
| **Pemahaman Teoretis SVD & PCA** | 25% | Menjelaskan hubungan spektral antara nilai eigen kovarians, nilai singular SVD, dan Teorema Eckart-Young-Mirsky dengan sangat tepat. | Memahami langkah komputasi PCA namun kurang mendalam dalam penurunan hubungan SVD. | Hanya mengutip rumus tanpa memahami kaitan aljabar linear antar-konsep. | Tidak memahami konsep reduksi dimensi linier. |
| **Pipeline Bebas Kebocoran Data** | 25% | Menjelaskan bahaya kebocoran data (*data leakage*) dan mengimplementasikan Pipeline Scikit-Learn dengan isolasi train-test sempurna. | Menerapkan train-test split namun penjelasan mengenai risiko kebocoran data kurang komprehensif. | Menggunakan scaler sebelum pembagian data train-test. | Tidak menerapkan pembagian train-test sama sekali. |
| **Penyelesaian Kasus & Tantangan** | 25% | Menyelesaikan ketiga tantangan terbimbing dengan kode bersih, metodologi tepat, dan interpretasi domain IoT yang kuat. | Menyelesaikan 2 tantangan dengan benar atau 3 tantangan dengan analisis terbatas. | Menyelesaikan 1 tantangan dengan benar. | Tidak menyelesaikan tantangan atau kode tidak dapat dieksekusi. |

