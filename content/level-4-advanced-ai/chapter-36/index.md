# Chapter 36: MLOps, Model Deployment & Monitoring

> **Artificial Intelligence Learning Hub — Level 4: Applied & Advanced AI Systems**  
> *Pengembang:* Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera  
> *Identitas Mata Kuliah:* Bab 36 dari 40 Master Curriculum  
> *Prasyarat:* Chapter 12 (ML Workflow), Chapter 19 (Model Evaluation), Chapter 30 (DL Capstone)

---

## DAFTAR ISI
1. [Bagian A: Pengantar MLOps & Siklus Hidup Sistem ML](#bagian-a-pengantar-mlops--siklus-hidup-sistem-ml)
2. [Bagian B: Reproducible ML Pipelines & Pencegahan Kebocoran Data](#bagian-b-reproducible-ml-pipelines--pencegahan-kebocoran-data)
3. [Bagian C: Experiment Tracking & Perbandingan Run](#bagian-c-experiment-tracking--perbandingan-run)
4. [Bagian D: Model Versioning & Local Model Registry](#bagian-d-model-versioning--local-model-registry)
5. [Bagian E: Model Packaging & Keamanan Deserialisasi Artefak](#bagian-e-model-packaging--keamanan-deserialisasi-artefak)
6. [Bagian F–G: Model Serving & Demonstrasi Local REST API (`127.0.0.1`)](#bagian-fg-model-serving--demonstrasi-local-rest-api-127001)
7. [Bagian H: Fondasi Monitoring & Observability Sistem ML](#bagian-h-fondasi-monitoring--observability-sistem-ml)
8. [Bagian I: Deteksi Data Drift (PSI & Uji Kolmogorov–Smirnov)](#bagian-i-deteksi-data-drift-psi--uji-kolmogorovsmirnov)
9. [Bagian J: Simulasi Concept Drift vs Pure Covariate Shift](#bagian-j-simulasi-concept-drift-vs-pure-covariate-shift)
10. [Bagian K: Performa Inferensi (Latensi, Persentil P95/P99 & Throughput)](#bagian-k-performa-inferensi-latensi-persentil-p95p99--throughput)
11. [Bagian L: Pembaruan Model, Retraining & Controlled Rollback](#bagian-l-pembaruan-model-retraining--controlled-rollback)
12. [Bagian M: Keamanan & Responsible MLOps](#bagian-m-keamanan--responsible-mlops)
13. [Latihan Bertingkat & Pembahasan Lengkap](#latihan-bertingkat--pembahasan-lengkap)
14. [Panduan Mini Project & Rubrik Asesmen 100 Poin](#panduan-mini-project--rubrik-asesmen-100-poin)
15. [Referensi Akademik & Standar Resmi](#referensi-akademik--standar-resmi)

---

## BAGIAN A: PENGANTAR MLOPS & SIKLUS HIDUP SISTEM ML

### 1. Dari Eksperimen Notebook Menuju Rekayasa Produksi
Banyak proyek Machine Learning terhenti sebagai prototipe rapuh di dalam sel-sel Jupyter Notebook yang hanya bekerja di laptop peneliti pada satu waktu tertentu. **MLOps (*Machine Learning Operations*)** adalah disiplin rekayasa yang menyatukan prinsip *Machine Learning*, *Software Engineering*, dan *Operations* untuk merancang, menguji, menyajikan, dan memantau model ML secara terstruktur, andal, dan dapat direproduksi ulang.

Sebagaimana ditegaskan oleh Sculley et al. (2015), kode algoritma pelatihan model hanyalah porsi sangat kecil dalam ekosistem ML nyata. Bagian terbesar kompleksitas terletak pada rekayasa pipeline data, konfigurasi, verifikasi integritas artefak, layanan inferensi, serta pemantauan pergeseran distribusi (*drift monitoring*).

```
+-----------------------------------------------------------------------+
|                    SIKLUS HIDUP MLOPS LENGKAP (CLOSED-LOOP)           |
+-----------------------------------------------------------------------+
|                                                                       |
|  [1. Data Ingestion & Provenance] ---> [2. Reproducible Training]     |
|                 ^                                    |                |
|                 |                                    v                |
|  [7. Governance & Rollback]          [3. Experiment Tracker (SQLite)] |
|                 ^                                    |                |
|                 |                                    v                |
|  [6. Drift & Latency Monitoring] <--- [4. Local Registry & Packaging] |
|                 ^                                    |                |
|                 |                                    v                |
|                 +------------------- [5. REST API Serving (127.0.0.1)]|
+-----------------------------------------------------------------------+
```

> **Peringatan Akademis:** Menjalankan server REST API lokal di laptop dan menerima respons HTTP 200 **tidak berarti** sebuah model telah *production-ready*. Kesiapan produksi sejati menuntut kontrol reproduksibilitas, verifikasi integritas kriptografis, validasi skema masukan, pemantauan *drift* kontinu, manajemen SLA latensi, serta protokol *rollback* darurat.

---

## BAGIAN B: REPRODUCIBLE ML PIPELINES & PENCEGAHAN KEBOCORAN DATA

Sebuah pipeline pelatihan disebut reproduktif apabila eksekusi ulang dengan konfigurasi yang sama menghasilkan metrik serta artefak yang konsisten lintas waktu. Tiga pilar pengamanannya meliputi:
1. **Dataset Provenance:** Menghitung sidik jari hash SHA-256 atas array fitur dan label masukan (`telemetry_ds_<hash>`).
2. **Zero Data Leakage Guard:** Pemecahan data (`train_test_split(..., stratify=y, random_state=42)`) **wajib dilakukan sebelum** penghitungan statistik penskalaan atau seleksi fitur apa pun. Statistik `StandardScaler` hanya diestimasi (*fit*) pada `X_train`.
3. **Training–Serving Parity:** Membungkus pra-pemrosesan dan pengklasifikasi dalam satu objek terpadu `sklearn.pipeline.Pipeline` mencegah terjadinya *training-serving skew* saat data langsung masuk melalui API.

```
+-----------------------------------------------------------------------+
|             PEMISAHAN KETAT TRAINING VS INFERENCE PIPELINE            |
+-----------------------------------------------------------------------+
| OFFLINE TRAINING PIPELINE            ONLINE SERVING PIPELINE          |
| ==========================           =======================          |
| 1. Load Versioned Data + Hash        1. Load Trusted Verified Artifact|
| 2. Stratified Train/Test Split       2. Terima JSON HTTP POST Request |
| 3. Pipeline([Scaler, Classifier])    3. Validasi Skema Input Ketat    |
| 4. Evaluate on Isolated Test Set     4. pipeline.predict_proba(x)     |
| 5. Save .pkl + Compute SHA-256       5. Return JSON + Measure Latency |
+-----------------------------------------------------------------------+
```

---

## BAGIAN C: EXPERIMENT TRACKING & PERBANDINGAN RUN

Selama pengembangan model, tim ML mencoba puluhan kombinasi arsitektur, fitur, dan hiperparameter. **Experiment Tracking** mencatat setiap sesi eksekusi (**Run**) ke penyimpanan terstruktur agar tidak ada eksperimen yang terlupakan atau hasilnya diklaim tanpa bukti.

Pada implementasi lokal (`code/experiment_tracking.py`), setiap Run dicatat ke database SQLite ringan (`mlops_tracker.db`) memuat:
- `run_id` & `timestamp_utc`
- `experiment_name` & `dataset_version`
- `params_json` (hiperparameter lengkap)
- `metrics_json` (Accuracy, F1-Score, ROC-AUC)
- `artifact_path`, `checksum_sha256`, dan `env_json` (versi Python, NumPy, Scikit-learn).

Berdasarkan eksekusi aktual `code/experiment_comparison.py` pada dataset telemetri 1.200 sampel (`test_size=25%`):

| Peringkat | Run ID | Model & Konfigurasi | Test Accuracy | Test F1-Score | Test ROC-AUC |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | `run_37298947` | **RandomForest (`n_estimators=80`)** | **0.9000** | **0.8438** | **0.9698** |
| **2** | `run_cb173116` | RandomForest (`n_estimators=20`) | 0.8967 | 0.8410 | 0.9696 |
| **3** | `run_a6272802` | LogisticRegression (`C=1.0`) | 0.8067 | 0.6848 | 0.8303 |
| **4** | `run_9a8a086c` | LogisticRegression (`C=0.1`) | 0.8067 | 0.6848 | 0.8299 |

---

## BAGIAN D: MODEL VERSIONING & LOCAL MODEL REGISTRY

### 1. Semantic Versioning untuk Model
Penomoran versi model mengadopsi konvensi `vMAJOR.MINOR.PATCH`:
- **PATCH (`v1.0.1`):** Penyesuaian kecil atau perbaikan kode pembungkus yang kompatibel penuh dengan kontrak input/output lama.
- **MINOR (`v1.1.0`):** Pelatihan ulang bobot model dengan data baru atau hiperparameter berbeda tanpa mengubah skema input fitur.
- **MAJOR (`v2.0.0`):** Perubahan struktur fitur masukan atau format keluaran yang memecah kompatibilitas klien lama (*breaking change*).

### 2. Local Model Registry & Siklus Status Jujur
**Model Registry** (`code/local_model_registry.py`) bertindak sebagai katalog terpusat yang mengikat versi model, silsilah dataset, checksum SHA-256, dan status tata kelola yang jujur:

```
+-----------------------------------------------------------------------+
|             SIKLUS STATUS MODEL DALAM LOCAL MODEL REGISTRY            |
+-----------------------------------------------------------------------+
|                                                                       |
|  [ CANDIDATE ] ---> [ VALIDATED ] ===(Human Approval)===> [ STAGED_LOCAL_SERVING ] |
|    (Baru dilatih)    (Lolos metrik)                         |         |
|                                                             v         |
|                                                        [ ARCHIVED ]   |
|                                               (Disimpan untuk Rollback)|
+-----------------------------------------------------------------------+
```

Setiap promosi menuju `STAGED_LOCAL_SERVING` mewajibkan argumen eksplisit `human_approved=True`. Upaya promosi otomatis tanpa persetujuan langsung diblokir oleh `PermissionError`.

---

## BAGIAN E: MODEL PACKAGING & KEAMANAN DESERIALISASI ARTEFAK

### Risiko Keamanan `pickle` dan `joblib`
Di ekosistem Python ML, berkas model sering disimpan menggunakan modul `pickle` atau `joblib`. Namun secara teknis, protokol `pickle` Python **bukan sekadar penyimpanan data pasif**—ia mengandung *stack virtual machine* yang mampu mengeksekusi kode Python sebarang (*arbitrary code execution* via metode `__reduce__`) saat `pickle.load()` dijalankan!

Oleh karena itu, `code/model_artifact_validation.py` menerapkan pertahanan kriptografis wajib:
1. Hitung dan simpan hash **SHA-256** pada saat artefak sah selesai dibuat di mesin pelatihan.
2. Sebelum `pickle.load()` dipanggil oleh server penyajian, hitung ulang SHA-256 berkas di disk dan bandingkan dengan nilai di Model Registry.
3. Jika berbeda walau satu bit saja, hentikan eksekusi dengan `ArtifactSecurityError`. Jangan pernah memuat berkas `.pkl` yang diunggah oleh pengguna eksternal tidak tepercaya.

---

## BAGIAN F–G: MODEL SERVING & DEMONSTRASI LOCAL REST API (`127.0.0.1`)

### 1. Pola Batch Inference vs Online Serving
- **Batch Inference:** Menghitung prediksi pada ratusan ribu baris sekaligus secara periodik (cocok untuk skor kredit bulanan atau segmentasi pelanggan).
- **Online Inference:** Menjawab satu permintaan tunggal secara langsung melalui antarmuka jaringan **REST API** dalam hitungan milidetik (cocok untuk deteksi anomali real-time).

### 2. Desain Endpoint & Pembatasan Keamanan `127.0.0.1`
Server REST API edukasional pada `code/model_serving_api.py` dibangun menggunakan `ThreadingHTTPServer` pustaka standar Python dan **wajib mengikat secara eksklusif pada `127.0.0.1`** (antarmuka loopback lokal—**tidak pernah `0.0.0.0`**) demi menjaga keamanan host pengembang:

| Endpoint | Metode | Fungsi & Kontrak Respon |
| :--- | :---: | :--- |
| `/health` | `GET` | Status kesiapan layanan (`healthy`), host `127.0.0.1`, dan versi model terpasang. |
| `/model/info` | `GET` | Metadata model, algoritma, versi dataset, dimensi fitur yang diharapkan (`6`), serta hash SHA-256. |
| `/predict` | `POST` | Memvalidasi payload JSON `{"features": [f0..f5]}` $\to$ mengembalikan `prediction`, `probability_positive`, `model_version`, dan `latency_ms`. |
| `/metrics` | `GET` | Statistik telemetri: total permintaan, jumlah penolakan galat, rata-rata dan P95 latensi (ms). |

Sebelum menyentuh model, `code/request_validation.py` menolak dengan HTTP `400 Bad Request` apabila:
- Dimensi fitur tidak sama dengan `6`.
- Terdapat elemen non-numerik (string, `None`), nilai `boolean`, `NaN`, atau `Inf`.
- Nilai sensor melampaui batas fisik wajar ($|x| > 10^6$).

---

## BAGIAN H: FONDASI MONITORING & OBSERVABILITY SISTEM ML

Model yang sudah disajikan bukan berarti selesai dikelola. Tim rekayasa membedakan dua sumbu pemantauan utama:
1. **System Observability:** Apakah server berjalan normal? Diukur melalui *P95/P99 inference latency*, *throughput (req/s)*, persentase galat HTTP 4xx/5xx, dan konsumsi CPU/RAM.
2. **ML Statistical & Performance Health:** Apakah distribusi data masukan, distribusi prediksi, dan akurasi model masih relevan dengan dunia nyata saat ini?

---

## BAGIAN I: DETEKSI DATA DRIFT (PSI & UJI KOLMOGOROV–SMIRNOV)

**Data Drift (*Covariate Shift*)** terjadi ketika distribusi probabilitas masukan saat ini $P_{\text{curr}}(X)$ menyimpang dari distribusi data acuan pelatihan $P_{\text{ref}}(X)$.

### 1. Population Stability Index (PSI)
Membandingkan proporsi populasi di setiap bin kuantil:

$$\text{PSI} = \sum_{i=1}^{B} \left(p_{\text{curr}, i} - p_{\text{ref}, i}\right) \cdot \ln\left(\frac{p_{\text{curr}, i}}{p_{\text{ref}, i}}\right)$$

Sebagai pedoman heuristik umum (bukan aturan mutlak universal karena bergantung pada jumlah bin $B$ dan ukuran sampel $N$): $\text{PSI} < 0.10$ stabil, $0.10 \le \text{PSI} < 0.25$ pergeseran sedang, dan $\text{PSI} \ge 0.25$ pergeseran signifikan.

### 2. Uji Dua Sampel Kolmogorov–Smirnov (KS Test)
Mengukur jarak vertikal maksimum antara fungsi distribusi kumulatif empiris (*Empirical CDF*) data acuan dan data saat ini:

$$D = \sup_{x} \left| F_{\text{ref}}(x) - F_{\text{curr}}(x) \right|$$

Hasil pengujian aktual pada `code/data_drift_detection.py` ($N_{\text{ref}}=500$ vs $N_{\text{curr}}=300$):

| Fitur | Pergeseran Mean | PSI | Statistik KS ($D$) | p-value KS | Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `feat_0` | +0.074 | **0.0394** | 0.0807 | 0.166386 | **NORMAL** |
| `feat_1` | +0.467 | **0.2076** | 0.2027 | 0.000000 | **CRITICAL DRIFT** |
| `feat_2` | +1.872 | **1.8298** | 0.5620 | 0.000000 | **CRITICAL DRIFT** |
| `feat_3` | +0.156 | **0.5579** | 0.1947 | 0.000001 | **CRITICAL DRIFT** |

> **Perhatian Statistik:** Pada ukuran sampel sangat besar ($N > 10.000$), perbedaan praktis yang tidak signifikan sekalipun dapat memicu $p\text{-value} < 0.001$ pada uji KS. Gunakan ukuran efek (*effect size* seperti PSI atau $D$) dan bukti performa, bukan sekadar p-value mentah.

---

## BAGIAN J: SIMULASI CONCEPT DRIFT VS PURE COVARIATE SHIFT

Kesalahan konseptual paling sering terjadi adalah menyamakan *Data Drift* dengan *Concept Drift*:
- **Data Drift:** Perubahan pada $P(X)$.
- **Concept Drift:** Perubahan pada hubungan bersyarat antara input dan target $P(y \mid X)$ (misalnya perilaku penipuan keuangan berubah pola, atau preferensi pembeli bergeser mendadak akibat krisis ekonomi). Jenisnya meliputi *Abrupt Drift* (mendadak), *Gradual Drift* (bertahap), dan *Recurring Drift* (berulang/musiman).

```
+-----------------------------------------------------------------------+
|         PERBEDAAN VISUAL: DATA DRIFT VS CONCEPT DRIFT                 |
+-----------------------------------------------------------------------+
| 1. PURE COVARIATE DATA DRIFT          2. ABRUPT CONCEPT DRIFT         |
|    P(X) bergeser, P(y|X) TETAP!          P(X) tetap, P(y|X) BERBALIK! |
|                                                                       |
|    o o | x x          o o | x x          o o | x x        x x | o o   |
|    o o | x x   ===>         x x x        o o | x x  ===>  x x | o o   |
|    (Latih)               (Baru)          (Latih)           (Baru)     |
|    Garis batas tetap akurat!             Garis batas lama salah total!|
+-----------------------------------------------------------------------+
```

Hasil simulasi aliran data berlabel aktual pada `code/concept_drift_simulation.py` membuktikan hal ini secara tegas:

| Rezim Aliran Data | Input PSI ($x_0$) | Akurasi Model | Diagnosis Rekayasa |
| :--- | :---: | :---: | :--- |
| **1. Stable Baseline** | 0.0383 | **96.33%** | Sehat; data dan konsep stasioner |
| **2. Pure Covariate Data Drift** | **2.6366** | **100.00%** | $P(X)$ bergeser jauh, namun $P(y \mid X)$ tetap akurat! |
| **3. Abrupt Concept Drift** | 0.0302 | **64.67%** | **BAHAYA:** Input tampak normal, tetapi batas $P(y \mid X)$ berbalik! |
| **4. Gradual Concept Drift (50%)** | 0.0170 | **76.00%** | Degradasi bertahap terdeteksi melalui label aktual |

**Kesimpulan Fundamental:** Tanpa label kebenaran (*ground-truth labels* $y_{\text{true}}$) atau bukti performa terukur, pergeseran distribusi prediksi semata tidak cukup untuk menyimpulkan terjadinya Concept Drift.

---

## BAGIAN K: PERFORMA INFERENSI (LATENSI, PERSENTIL P95/P99 & THROUGHPUT)

Rata-rata latensi (*mean latency*) kerap menipu karena 5% permintaan terburuk (*tail latency*) dapat menyebabkan antrean macet (*bottleneck*). Pengukuran performa wajib melaporkan Median (**P50**), persentil ke-95 (**P95**), persentil ke-99 (**P99**), dan **Throughput (req/s)** setelah membuang fase *warm-up* awal.

Hasil pengukuran aktual pada CPU lokal (`127.0.0.1`) melalui `code/inference_latency.py`:

| Mode Pengukuran | Sampel Uji (*Warm-Up Excluded*) | Mean (ms) | Median P50 (ms) | P95 (ms) | P99 (ms) | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Direct In-Memory Predict** | $N=500$ (20 warm-up) | **0.167 ms** | **0.153 ms** | **0.256 ms** | **0.320 ms** | **5.973,6 req/s** |
| **2. Localhost REST API POST** | $N=100$ (10 warm-up) | **4.009 ms** | **2.131 ms** | **19.422 ms** | **25.985 ms** | **249,4 req/s** |

> **Batasan Benchmark:** Pengukuran ini mencerminkan eksekusi loopback single-host pada CPU Windows lokal (`127.0.0.1`) dan tidak dapat digeneralisasi secara langsung ke kapasitas *production server* berskala konkurensi tinggi di jaringan awan.

---

## BAGIAN L: PEMBARUAN MODEL, RETRAINING & CONTROLLED ROLLBACK

Ketika dashboard monitoring mendeteksi pelanggaran SLA (misalnya akurasi merosot di bawah $80\%$), tim MLOps tidak boleh membiarkan skrip otomatis mendorong model baru secara liar tanpa validasi:
1. **Alert & Triage:** Pastikan apakah degradasi disebabkan oleh masalah upstream (sensor rusak/bug skema) atau benar-benar *Concept Drift*.
2. **Controlled Rollback:** Jika versi kandidat baru (`v1.1.0`) justru berperforma lebih buruk dari versi sebelumnya, supervisor manusia mengesahkan *rollback* cepat di `LocalModelRegistry` kembali ke versi stabil tervalidasi (`v1.0.0`) yang masih tersimpan sebagai `ARCHIVED` dan terverifikasi checksum SHA-256.
3. **Retraining & Validasi Regresi:** Latih model kandidat baru pada data terkini, buktikan keunggulannya di *validation split* terisolasi, baru ajukan persetujuan manusia untuk promosi.

---

## BAGIAN M: KEAMANAN & RESPONSIBLE MLOPS

Pengelolaan siklus hidup ML bertanggung jawab menuntut penegakan prinsip berikut *(berkesinambungan dengan tata kelola AI Safety pada Chapter 39)*:
- **Privasi Log:** Jangan mencatat data pribadi sensitif (*PII*) ke dalam log permintaan HTTP mentah.
- **Keamanan Model Serving:** Ikat API ke `127.0.0.1` untuk uji lokal, terapkan *timeout*, pembatasan ukuran payload (`Content-Length <= 64 KB`), dan validasi skema ketat.
- **Integritas Artefak:** Wajib verifikasi SHA-256 sebelum deserialisasi.
- **Pengawasan Manusia & Auditability:** Setiap perubahan status model dalam registry tercatat rapi dalam jejak audit dan tidak pernah melewati *human approval gate*.

---

## LATIHAN BERTINGKAT & PEMBAHASAN LENGKAP

### Tingkat Dasar
1. **MLOps Lifecycle:** Mengapa *training-serving skew* dapat terjadi apabila insinyur ML menuliskan ulang rumus standardisasi `(x - mean) / std` secara manual di API server alih-alih menyimpan objek `StandardScaler` di dalam `Pipeline`?  
   *Pembahasan:* Perbedaan kecil implementasi pembulatan atau kesalahan menyalin parameter `mean`/`std` latih akan menggeser seluruh fitur masukan saat inferensi (*skew*), menurunkan akurasi secara misterius.
2. **Local Model Registry:** Jelaskan mengapa penggunaan status `"PRODUCTION"` pada proyek simulasi lokal tanpa infrastruktur produksi dapat menyesatkan!  
   *Pembahasan:* Label harus merefleksikan keadaan sebenarnya (`STAGED_LOCAL_SERVING` pada `127.0.0.1`), menjaga kejujuran audit rekayasa.

### Tingkat Menengah
3. **API Schema Validation:** Mengapa nilai boolean Python (`True` / `False`) harus secara eksplisit ditolak dalam validator tipe numerik `(int, float)`?  
   *Pembahasan:* Di Python, `bool` merupakan subkelas dari `int` (`isinstance(True, int) == True`), sehingga tanpa pengecekan `isinstance(val, bool)` nilai boolean bisa lolos secara diam-diam sebagai `1` atau `0`.
4. **Inference Latency:** Jelaskan mengapa P95 (`19.422 ms`) jauh lebih informatif dibanding rata-rata (`4.009 ms`) dalam menjamin SLA layanan online!  
   *Pembahasan:* Rata-rata menyembunyikan *tail latency* dari permintaan terburuk; P95 memastikan 95 dari 100 pengguna benar-benar dilayani di bawah batas waktu tersebut.

### Tingkat Mahir
5. **Diagnosis Drift:** Diketahui skor PSI fitur $X_1$ melonjak ke `1.83` (`CRITICAL DRIFT`), sementara setelah ditunggu label aktualnya ternyata Akurasi Model bertahan di `99.2%`. Jelaskan fenomena ini dan keputusan operasional yang tepat!  
   *Pembahasan:* Terjadi **Pure Covariate Data Drift** pada rentang $P(X)$ di mana hubungan aturan keputusan $P(y \mid X)$ tetap berlaku baik. Jangan lakukan *rollback* atau *emergency retraining* buta; perbarui baseline acuan distribusi fitur setelah mengonfirmasi stabilitas metrik bisnis.

---

## PANDUAN MINI PROJECT & RUBRIK ASESMEN 100 POIN

### Judul Proyek
**Building a Local MLOps Pipeline with Model Serving & Drift Monitoring**

### Ringkasan Deliverables
Mahasiswa membangun pipeline MLOps lokal ujung-ke-ujung mencakup pelatihan reproduktif bebas kebocoran data, pelacak eksperimen SQLite, *Local Model Registry* berstatus jujur, verifikasi checksum SHA-256, REST API `127.0.0.1` dengan pengujian HTTP klien, benchmark latensi P50/P95, analisis perbedaan *Data Drift* vs *Concept Drift*, serta simulasi *rollback* disetujui manusia.

### Rubrik Penilaian (100 Poin)
| Kriteria Penilaian | Bobot | Indikator Penilaian Maksimal |
| :--- | :---: | :--- |
| **1. Pipeline & Reproducibility** | **20** | Pemecahan data tanpa *leakage*, `Pipeline` utuh, seed deterministik, serta provenance hash dataset tercatat. |
| **2. Experiment Tracking & Registry** | **20** | SQLite tracker mencatat seluruh run, perbandingan kandidat adil, dan Registry menegakkan persetujuan manusia. |
| **3. Local Model Serving** | **25** | REST API mengikat eksklusif pada `127.0.0.1`, menolak skema cacat dengan HTTP 400, serta mengukur latensi P50/P95 aktual. |
| **4. Monitoring & Drift Analysis** | **25** | Kalkulasi PSI & KS akurat, mampu membedakan *Covariate Shift* vs *Concept Drift* berlabel, dan rollback berjalan aman. |
| **5. Documentation & Security** | **10** | Checksum SHA-256 aktif sebelum `pickle.load()`, batasan benchmark lokal dicatat jujur, dan kode 100% reproduktif di CPU. |

---

## REFERENSI AKADEMIK & STANDAR RESMI

1. **Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Dennison, D. (2015).** *Hidden Technical Debt in Machine Learning Systems.* Advances in Neural Information Processing Systems (NeurIPS 2015), 28.
2. **Zinkevich, M. (2017).** *Rules of Machine Learning: Best Practices for ML Engineering.* Google Research. https://developers.google.com/machine-learning/guides/rules-of-ml
3. **Google Cloud. (2021).** *Practitioners Guide to MLOps: A Framework for Continuous Delivery and Automation of Machine Learning.* https://cloud.google.com/resources/mlops-whitepaper
4. **Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014).** *A Survey on Concept Drift Adaptation.* ACM Computing Surveys (CSUR), 46(4), 1–37. https://doi.org/10.1145/2523813
5. **National Institute of Standards and Technology (NIST). (2023).** *Artificial Intelligence Risk Management Framework (AI RMF 1.0).* NIST AI 100-1. https://doi.org/10.6028/NIST.AI.100-1

