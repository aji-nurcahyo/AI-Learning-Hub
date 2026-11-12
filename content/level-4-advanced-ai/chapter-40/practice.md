# Panduan Praktik Capstone Reference Implementation
**Artificial Intelligence Learning Hub — Level 4 (Chapter 40)**  
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## 1. DESKRIPSI PROYEK REFERENSI CAPSTONE

Proyek referensi end-to-end berbasis CPU lokal ini berjudul **Intelligent Sensor Monitoring with Machine Learning, Evaluation & Responsible AI** (`content/level-4-advanced-ai/chapter-40/code/`). Sistem mendemonstrasikan integrasi kompetensi Level 1–4 secara utuh, reprodusibel (`seed=42`), ringan, dan dilengkapi kontrol keamanan serta dokumentasi Model Card.

### Lingkungan Eksekusi
- **Bahasa:** Python 3.10+ (Teruji pada Python 3.12.10 CPU).
- **Pustaka Utama:** `numpy`, `scikit-learn`, `matplotlib`, dan Python Standard Library (`hashlib`, `pickle`, `unittest`, `json`, `dataclasses`). Tidak membutuhkan GPU, CUDA, maupun layanan cloud eksternal.

---

## 2. PANDUAN EKSEKUSI MODUL & PENGUJIAN (11 SCRIPT)

### Langkah 1: Verifikasi Konfigurasi & Generator Dataset + Kamus Data
```bash
python content/level-4-advanced-ai/chapter-40/code/config.py
python content/level-4-advanced-ai/chapter-40/code/dataset_generator.py
```
- **Tujuan:** Menghasilkan `1,500` sampel telemetri motor sintetis 6-fitur (*fault rate* `26.60%`) dan menyimpan kamus data formal di `data/sensor_data_dictionary.json`.

### Langkah 2: Persiapan Data Bebas Kebocoran (*Zero Data Leakage*)
```bash
python content/level-4-advanced-ai/chapter-40/code/preprocessing.py
```
- **Tujuan:** Memecah *Train (`900`)*, *Validation (`300`)*, dan *Held-Out Test (`300`)* secara berstrata sebelum menerapkan `StandardScaler.fit()` murni pada *Train Set*.

### Langkah 3: Evaluasi Model Baseline & Pembandingan Kandidat
```bash
python content/level-4-advanced-ai/chapter-40/code/baseline_model.py
python content/level-4-advanced-ai/chapter-40/code/candidate_models.py
```
- **Tujuan:** Membandingkan `DummyClassifier(most_frequent)` (`73.33%` Acc, `0.0000` F1) terhadap `LogisticRegression` (`F1=0.9146`, `1.15 KB`), `RandomForest` (`F1=0.8875`, `313.66 KB`), dan `MLP(32,16)` (`F1=0.8970`, `39.22 KB`) pada *Validation Set*, lalu menyimpan artefak pemenang beserta checksum **SHA-256** (`experiments/capstone_model_sha256.txt`).

### Langkah 4: Evaluasi Akhir, Explainability & Analisis Kesalahan
```bash
python content/level-4-advanced-ai/chapter-40/code/evaluation_and_error_analysis.py
```
- **Tujuan:** Memverifikasi kecocokan hash SHA-256 sebelum memuat artefak, mengukur performa sekali saja pada *Held-Out Test Set* (`N=300`: Acc `96.00%`, F1 `0.9231`, ROC-AUC `0.9937`, `12` error), menghitung *Permutation Importance* (`vibration_rms: 0.1298`, `temperature_c: 0.0824`), dan menyimpan grafik `experiments/capstone_evaluation_chart.png`.

### Langkah 5: Inferensi Lokal Tervalidasi & Simulasi Monitoring Drift
```bash
python content/level-4-advanced-ai/chapter-40/code/local_inference.py
python content/level-4-advanced-ai/chapter-40/code/monitoring_and_drift.py
```
- **Tujuan:** Menguji penolakan payload cacat (`NaN` $\to$ `REJECTED_INVALID_INPUT`, `human_escalation=True`) serta menyimulasikan pergeseran telemetri (`+9.5C` & `+1.1 mm/s` $\implies$ `MaxPSI = 2.0395 > 0.25`, `Alert=True`) dengan grafik `experiments/capstone_drift_monitoring.png`.

### Langkah 6: Risk Register, Model Card, Pipeline Terpadu & Unit Tests
```bash
python content/level-4-advanced-ai/chapter-40/code/responsible_ai_assessment.py
python content/level-4-advanced-ai/chapter-40/code/run_capstone_pipeline.py
python content/level-4-advanced-ai/chapter-40/code/test_capstone_suite.py
```
- **Tujuan:** Menjalankan keseluruhan orkestrator end-to-end (`1.67s`), menyimpan `experiments/capstone_report_metrics.json` & `experiments/capstone_model_card.json`, dan memverifikasi **4/4 unit tests** (`unittest`).

---

## 3. COMPANION NOTEBOOK & LATIHAN REFLEKSI CAPSTONE

Buka companion notebook terstruktur di: `content/level-4-advanced-ai/chapter-40/notebooks/chapter_40_companion.ipynb`.

### Latihan Integrasi & Pertahanan Teknis (Dengan Pembahasan)
1. **Pemilihan Model:** Mengapa `LogisticRegression` dipilih pada proyek referensi mengungguli `MLP(32,16)` dan `RandomForest`?
   - **Pembahasan:** Karena struktur batas anomali pada telemetri motor sintetis dominan aditif-linear di ruang fitur standar, `LogisticRegression` meraih skor *Validation F1* tertinggi (`0.9146`), ukuran artefak jauh paling ramping (`1.15 KB` vs `313.66 KB`), latensi rata-rata rendah (`0.0848 ms`), dan interpretabilitas koefisien langsung.
2. **Keamanan Artefak & Drift:** Mengapa `load_verified_bundle` mewajibkan perbandingan checksum **SHA-256** sebelum memanggil `pickle.loads()`, dan mengapa pemantauan **PSI** tetap dijalankan meskipun akurasi sesaat di aliran drift (`94.33%`) belum jatuh drastis?
   - **Pembahasan:** Berkas `pickle` Python dapat mengeksekusi bytecode sewenang-wenang bila dimodifikasi penyerang, sehingga SHA-256 menjamin integritas sebelum deserialisasi. Sementara itu, kenaikan *Population Stability Index* (`MaxPSI = 2.0395`) memberi peringatan dini bahwa distribusi sensor telah keluar dari wilayah kalibrasi pelatihan sebelum kegagalan sistem parah terjadi.
3. **Kesesuaian Metrik Lintas Tema:** Mengapa proyek *Reinforcement Learning* (Opsi 8) atau *RAG Assistant* (Opsi 2) dilarang dinilai menggunakan `accuracy` tunggal?
   - **Pembahasan:** Sistem RL tidak memiliki label benar tetap per langkah melainkan dinilai lewat *discounted return*, *success rate* lintas episode, dan stabilitas kebijakan; sistem RAG dinilai melalui metrik peringkat retrieval (`Recall@k`, `MRR`) dan ketepatan sitasi (*groundedness* / *faithfulness*).

---

## 4. TUGAS AKHIR: ADVANCED AI CAPSTONE PROJECT & RUBRIK 100 POIN

Susun proyek Capstone pilihan Anda (dari **8 Opsi Tema** di Modul `index.md` atau topik khusus disetujui pengajar) melampaui **8 Tahap Pengembangan & 7 Milestone** serta memenuhi **14 Persyaratan Ilmiah Wajib**.

### Rubrik Penilaian Capstone (Total 100 Poin — Adil Lintas Semua Domain AI)

| Dimensi Penilaian | Bobot | Full Credit (85–100%) | Partial Credit (50–84%) | Insufficient Evidence (<50%) |
| :--- | :---: | :--- | :--- | :--- |
| **1. Problem Formulation & Requirements** | **15** | Masalah, *intended/prohibited use*, kebutuhan fungsional/nonfungsional, dan kriteria sukses terukur jelas. | Masalah ada, namun kebutuhan nonfungsional atau batasan skop masih kabur. | Rumusan masalah umum tanpa ukuran keberhasilan. |
| **2. Dataset & System Architecture** | **15** | Provenance, lisensi, *Data Dictionary*, pencegahan leakage, dan diagram arsitektur modular jelas. | Pipeline berjalan, namun data dictionary atau pemisahan fold kurang tegas. | Sumber data tak jelas atau terjadi *data leakage*. |
| **3. Model / Algorithm Implementation** | **25** | Kode modular deterministik, konfigurasi rapi, uji unit lolos, pilihan algoritma tepat sesuai masalah. | Implementasi berjalan, namun kurang terstruktur atau tidak ada unit test. | Skrip ad-hoc gagal direproduksi atau overengineered tanpa alasan. |
| **4. Evaluation & Experimental Analysis** | **20** | Baseline adil, metrik tepat tugas (ML/DL/RAG/Agent/RL/Edge), *error analysis*, dan benchmark sumber daya. | Ada evaluasi utama, namun baseline lemah atau minus analisis kegagalan/latensi. | Hanya angka akurasi tunggal tanpa konteks perbandingan. |
| **5. Security & Responsible AI** | **15** | Kontrol input/izin, Risk Register, Model Card jujur, & pembahasan keterbatasan/etika kontekstual. | Penilaian risiko dan batasan hanya disebutkan singkat di penutup. | Tidak ada kontrol keamanan, Risk Register, atau Model Card. |
| **6. Documentation & Presentation** | **10** | Dokumentasi teknis lengkap, README reproduksi, visualisasi bersih, serta pertahanan argumen kuat. | Dokumentasi cukup, namun alur presentasi atau kejelasan visual kurang rapi. | Dokumentasi minim dan tidak mendukung reproduksi ulang. |

---

## 5. REFERENSI AKADEMIK & REKAYASA

1. Sculley, D., Holt, G., Golovin, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS 2015. https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
2. Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*. IEEE Big Data. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
3. NIST (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1*. https://doi.org/10.6028/NIST.AI.100-1
4. Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). *Model Cards for Model Reporting*. ACM FAT* 2019. https://arxiv.org/abs/1810.03993

