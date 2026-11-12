# Panduan Praktik Laboratorium: MLOps, Model Deployment & Monitoring
**Artificial Intelligence Learning Hub — Level 4 (Chapter 36)**  
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## 1. DESKRIPSI DAN PRASYARAT PRAKTIKUM

Praktikum ini membimbing mahasiswa membangun ekosistem rekayasa **MLOps** lengkap secara lokal pada CPU—mencakup *reproducible training* bebas kebocoran data, *experiment tracking* & *model registry* berbasis SQLite, validasi checksum artefak SHA-256, *local REST API model serving* (`127.0.0.1`), pengukuran latensi aktual, analisis *Data Drift* vs *Concept Drift*, serta simulasi *rollback* terkendali.

Seluruh kode berjalan 100% lokal tanpa ketergantungan Docker, Kubernetes, server MLflow eksternal, maupun cloud hosting.

### Prasyarat Pengetahuan
- Alur kerja Machine Learning dan pencegahan data leakage (Chapter 12 & 19).
- Pemrograman Python standar (`sqlite3`, `http.server`, `hashlib`, `json`, `urllib`), NumPy, dan Scikit-learn.

### Lingkungan Eksekusi
- **Bahasa:** Python 3.10+ (Direkomendasikan Python 3.12).
- **Pustaka Utama:** `numpy`, `scikit-learn`, `matplotlib`, pustaka standar Python (`sqlite3`, `http.server`, `hashlib`).

---

## 2. PANDUAN LANGKAH DEMI LANGKAH EKSEKUSI

Seluruh 17 script Python mandiri tersimpan di `content/level-4-advanced-ai/chapter-36/code/`:

### Langkah 1: Kontrak Tahapan Siklus Hidup MLOps
```bash
python content/level-4-advanced-ai/chapter-36/code/mlops_lifecycle.py
```
- **Tujuan:** Memahami ketujuh fase siklus hidup MLOps beserta *governance gate* pada setiap transisi fase.

### Langkah 2: Pelatihan Reproduktif Bebas Kebocoran Data
```bash
python content/level-4-advanced-ai/chapter-36/code/reproducible_training.py
```
- **Tujuan:** Membagi data berstrata sebelum penskalaan (`StandardScaler` terbungkus `Pipeline`), menghitung hash SHA-256 artefak, dan merekam versi lingkungan.

### Langkah 3: Experiment Tracking Lokal Berbasis SQLite
```bash
python content/level-4-advanced-ai/chapter-36/code/experiment_tracking.py
```
- **Tujuan:** Menyimpan metadata run eksperimen (parameter, metrik, dataset version, checksum) ke database SQLite `experiments/mlops_tracker.db`.

### Langkah 4: Perbandingan Eksperimen & Pemilihan Kandidat
```bash
python content/level-4-advanced-ai/chapter-36/code/experiment_comparison.py
```
- **Tujuan:** Menjalankan komparasi objektif antara `LogisticRegression` ($C=0.1, 1.0$) dan `RandomForest` ($n=20, 80$) hingga menemukan kandidat terbaik (AUC `0.9698`).

### Langkah 5: Semantic Model Versioning & Silsilah (*Lineage*)
```bash
python content/level-4-advanced-ai/chapter-36/code/model_versioning.py
```
- **Tujuan:** Mempraktikkan aturan *Semantic Versioning* (`MAJOR.MINOR.PATCH`) dan pelacakan silsilah model.

### Langkah 6: Local Model Registry & Tata Kelola Promosi
```bash
python content/level-4-advanced-ai/chapter-36/code/local_model_registry.py
```
- **Tujuan:** Mengelola status siklus (`CANDIDATE` $\to$ `VALIDATED` $\to$ `STAGED_LOCAL_SERVING`) dengan penolakan otomatis bila tanpa persetujuan manusia (`human_approved=False`).

### Langkah 7: Verifikasi Integritas Artefak & Keamanan Pickle
```bash
python content/level-4-advanced-ai/chapter-36/code/model_artifact_validation.py
```
- **Tujuan:** Memvalidasi hash SHA-256 sebelum `pickle.load()` dan memblokir berkas artefak yang dimodifikasi (*tampered bytes*).

### Langkah 8: Validasi Skema Permintaan REST API
```bash
python content/level-4-advanced-ai/chapter-36/code/request_validation.py
```
- **Tujuan:** Menolak masukan ilegal (dimensi salah, string injection, boolean, `NaN`/`Inf`, dan nilai di luar batas wajar).

### Langkah 9: Menjalankan Server REST API Model Lokal (`127.0.0.1`)
```bash
python content/level-4-advanced-ai/chapter-36/code/model_serving_api.py
```
- **Tujuan:** Memulai layanan `ThreadingHTTPServer` secara eksklusif pada `127.0.0.1`, melayani `/health` serta `/predict`, lalu menutup server secara bersih.

### Langkah 10: Pengujian Integrasi Klien HTTP Menyeluruh
```bash
python content/level-4-advanced-ai/chapter-36/code/api_client_demo.py
```
- **Tujuan:** Menguji lengkap `GET /health`, `GET /model/info`, `POST /predict` valid (HTTP 200), skema cacat (HTTP 400), serta pencatatan `GET /metrics`.

### Langkah 11: Deteksi Data Drift Statistik (PSI & Uji KS)
```bash
python content/level-4-advanced-ai/chapter-36/code/data_drift_detection.py
```
- **Tujuan:** Menghitung *Population Stability Index* (PSI) dan uji dua sampel *Kolmogorov–Smirnov* (KS Test) membandingkan referensi vs aliran baru.

### Langkah 12: Simulasi Concept Drift vs Covariate Data Drift
```bash
python content/level-4-advanced-ai/chapter-36/code/concept_drift_simulation.py
```
- **Tujuan:** Membuktikan perbedaan antara *Pure Covariate Shift* (PSI `2.6366`, akurasi bertahan `100.00%`) versus *Abrupt Concept Drift* (PSI `0.0302`, akurasi anjlok ke `64.67%`).

### Langkah 13: Pemantauan Distribusi Prediksi (*Prediction Drift*)
```bash
python content/level-4-advanced-ai/chapter-36/code/prediction_monitoring.py
```
- **Tujuan:** Mendeteksi pergeseran rasio kelas positif dan pita ketidakpastian probabilitas sebelum label kebenaran tiba.

### Langkah 14: Benchmark Latensi Inferensi & Throughput CPU
```bash
python content/level-4-advanced-ai/chapter-36/code/inference_latency.py
```
- **Tujuan:** Mengukur rata-rata, P50 Median, P95, P99 (ms), serta *throughput* (req/s) pada inferensi langsung memori vs HTTP loopback `127.0.0.1`.

### Langkah 15: Dashboard Observability & Grafik Multi-Panel
```bash
python content/level-4-advanced-ai/chapter-36/code/monitoring_dashboard.py
```
- **Tujuan:** Mengagregasikan metrik PSI, akurasi bergulir, dan histogram latensi melintasi 5 jendela waktu serta menyimpan gambar di `experiments/monitoring_dashboard.png`.

### Langkah 16: Simulasi Model Rollback Terkendali
```bash
python content/level-4-advanced-ai/chapter-36/code/model_rollback_simulation.py
```
- **Tujuan:** Memicu peringatan pelanggaran SLA akurasi `< 80%`, mencegat rollback otomatis, dan memulihkan `v1.0.0` setelah konfirmasi persetujuan manusia.

### Langkah 17: Demonstrasi Pipeline MLOps Ujung-ke-Ujung
```bash
python content/level-4-advanced-ai/chapter-36/code/mlops_pipeline_demo.py
```
- **Tujuan:** Menyatukan pelatihan reproduktif, tracking, registry, API serving lokal, dan deteksi drift dalam satu eksekusi terpadu.

---

## 3. LATIHAN BERTINGKAT & PEMBAHASAN

### Tingkat Dasar
1. **Latihan 1.1 (MLOps Lifecycle):** Sebutkan perbedaan paling mendasar antara berkas kode model biasa dengan *Model Artifact* di dalam *Model Registry*!  
   *Pembahasan:* *Model Artifact* mencakup bobot/objek terlatih lengkap beserta pra-pemrosesan (`Pipeline`), metadata silsilah (versi dataset, parameter, metrik), dan checksum kriptografis guna menjamin reproduksibilitas dan integritas.
2. **Latihan 1.2 (Local Binding):** Mengapa praktik melarang keras server uji lokal diikat pada `0.0.0.0` dan mewajibkan `127.0.0.1`?  
   *Pembahasan:* Mengikat ke `0.0.0.0` membuka port pada seluruh antarmuka jaringan (Wi-Fi/LAN publik), mengekspos layanan yang belum diautentikasi kepada perangkat lain di jaringan yang sama.

### Tingkat Menengah
3. **Latihan 2.1 (API Schema Validation):** Tambahkan validasi batas rentang fitur spesifik pada `request_validation.py` sehingga fitur indeks ke-0 ditolak bila bernilai $< -10.0$ atau $> 10.0$, lalu uji dengan `api_client_demo.py`.
4. **Latihan 2.2 (Inference Latency):** Jalankan `inference_latency.py` dan bandingkan mengapa latensi `POST /predict` via HTTP (`4.009 ms` rata-rata) puluhan kali lebih lambat dibandingkan `predict_proba()` langsung di memori (`0.167 ms`)!  
   *Pembahasan:* Komunikasi HTTP melibatkan serialisasi/deserialisasi JSON, pembentukan soket TCP loopback `127.0.0.1`, serta parsing header HTTP oleh handler server.

### Tingkat Mahir
5. **Latihan 3.1 (Concept vs Data Drift Diagnosis):** Diberikan skenario monitoring: PSI seluruh fitur = `1.85` (Kritis), tingkat prediksi positif melonjak dari `35%` menjadi `62%`, namun setelah label aktual masuk ternyata akurasi tetap `95%`. Apa diagnosis Anda dan apa rekomendasi tindakannya?  
   *Pembahasan:* Terjadi **Pure Covariate Data Drift** di mana distribusi populasi input bergeser ke zona kelas positif, namun batas keputusan $P(y \mid X)$ tetap akurat. Rekomendasi: Perbarui statistik acuan (*reference baseline*) agar tidak memicu *false alarm* berulang, namun **tidak perlu** memicu *rollback*.
6. **Latihan 3.2 (Keamanan Deserialisasi):** Jelaskan mekanisme eksploitasi `__reduce__` pada `pickle` Python dan rancang protokol pertahanan berlapis untuk repositori model institusi.

---

## 4. PANDUAN MINI PROJECT

### Judul Proyek
**Building a Local MLOps Pipeline with Model Serving & Drift Monitoring**

### Deliverables Wajib
1. Definisi masalah klasifikasi dan dataset deterministik dengan provenance hash.
2. Pipeline pelatihan reproduktif bebas *data leakage*.
3. Implementasi SQLite Experiment Tracker & perbandingan minimal 3 eksperimen.
4. Local Model Registry dengan status terkendali (`CANDIDATE` $\to$ `VALIDATED` $\to$ `STAGED_LOCAL_SERVING`) & verifikasi SHA-256.
5. REST API lokal pada `127.0.0.1` lengkap dengan `/health`, `/model/info`, `/predict`, dan `/metrics` plus skrip uji integrasi.
6. Pengukuran latensi (P50, P95, Throughput) & analisis perbedaan Data Drift (PSI/KS) vs Concept Drift.
7. Simulasi *rollback* yang membutuhkan persetujuan manusia eksplisit serta audit keamanan deserialisasi.

---

## 5. RUBRIK PENILAIAN (100 POIN)

| Komponen Penilaian | Bobot | Kriteria Pemenuhan Standar Penuh |
| :--- | :---: | :--- |
| **1. Pipeline & Reproducibility** | **20 Poin** | Bebas *data leakage*, pra-pemrosesan terikat dalam `Pipeline`, seed deterministik, provenance tercatat jelas. |
| **2. Experiment Tracking & Registry** | **20 Poin** | Pelacakan minimal 3 run di SQLite, seleksi kandidat objektif, Model Registry memakai status jujur dan gerbang persetujuan. |
| **3. Local Model Serving & Validation** | **25 Poin** | REST API hanya mengikat ke `127.0.0.1`, validasi payload ketat (HTTP 400 pada masukan cacat), pengujian latensi P50/P95 terekam. |
| **4. Monitoring, Drift & Rollback** | **25 Poin** | Kalkulasi PSI & KS tepat, simulasi mampu membedakan *Covariate Shift* dari *Concept Drift*, serta simulasi rollback berjalan aman. |
| **5. Documentation & Security** | **10 Poin** | Verifikasi checksum SHA-256 aktif sebelum `pickle.load()`, dokumentasi batas benchmark lokal jelas, dan reproduksibilitas 100% di CPU. |

---

## 6. REFERENSI AKADEMIK & STANDAR RESMI

1. **Sculley, D., et al. (2015).** *Hidden Technical Debt in Machine Learning Systems.* Advances in Neural Information Processing Systems (NeurIPS 2015), 28.
2. **Zinkevich, M. (2017).** *Rules of Machine Learning: Best Practices for ML Engineering.* Google Research.
3. **Google Cloud. (2021).** *Practitioners Guide to MLOps: A Framework for Continuous Delivery and Automation of Machine Learning.*
4. **Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014).** *A Survey on Concept Drift Adaptation.* ACM Computing Surveys (CSUR), 46(4), 1–37.
5. **National Institute of Standards and Technology (NIST). (2023).** *Artificial Intelligence Risk Management Framework (AI RMF 1.0).* NIST AI 100-1.

