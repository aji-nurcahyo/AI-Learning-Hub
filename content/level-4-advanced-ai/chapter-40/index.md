# Chapter 40: Advanced AI Capstone Project
**Penutup Resmi Master Curriculum (40 Chapters) — Artificial Intelligence Learning Hub**  
*Created & Developed by Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## PART A — TUJUAN CAPSTONE & INTEGRASI MASTER CURRICULUM

**Chapter 40** merupakan kulminasi resmi seluruh **40 chapter** kurikulum terbuka **Artificial Intelligence Learning Hub** dari Level 1 (*AI Foundations*), Level 2 (*Machine Learning*), Level 3 (*Deep Learning & Generative AI*), hingga Level 4 (*Applied & Advanced AI Systems*).

```text
  LEVEL 1 (Ch 01–10)       LEVEL 2 (Ch 11–20)       LEVEL 3 (Ch 21–30)       LEVEL 4 (Ch 31–39)
  Fondasi AI, Search,  ->  ML Klasik, Evaluasi  ->  PyTorch, CNN, YOLO,  ->  RAG, Agents, RL, MLOps,
  Probabilitas & Etika     & Explainability         NLP, Transformer & LLM   Edge, Multimodal & Safety
                                       \                     /
                                        v                   v
                             [ CHAPTER 40: ADVANCED AI CAPSTONE PROJECT ]
```

**Filosofi Utama Rekayasa Capstone:**
- **Problem-First Architecture:** Mahasiswa tidak diwajibkan memaksakan semua teknik ke dalam satu aplikasi; pilihlah pendekatan algoritmik paling sesuai dan paling sederhana yang efektif menjawab masalah.
- **CPU-First & Local-First:** Seluruh eksperimen wajib berjalan reproduktif di CPU lokal tanpa mewajibkan GPU, kunci API berbayar, atau penyebaran infrastruktur cloud publik.
- **Scientific Rigor:** Setiap keputusan desain harus ditopang perbandingan *baseline*, pencegahan *data leakage*, analisis kesalahan, pengukuran sumber daya, serta asesmen keselamatan dan tata kelola **Responsible AI**.

---

## PART B — DELAPAN OPSI TEMA CAPSTONE PROJECT

Mahasiswa memilih **satu** jalur tema berikut (atau mengajukan topik alternatif yang relevan dengan persetujuan pengajar):

1. **Option 1 — Edge AI Intelligent Monitoring:** Prototipe model ringan pemantauan sensor lokal dengan kompresi (kuantisasi/pruning/distilasi) dan profiling memori–latensi tanpa mewajibkan perangkat keras fisik.
2. **Option 2 — RAG-Based Academic Assistant:** Asisten tanya-jawab berbasiskan korpus akademik dengan penelusuran vektor/hibrida, validasi sitasi, dan isolasi *indirect prompt injection* (memakai *extractive/deterministic baseline* luring bila LLM lokal tak tersedia—**dilarang** menyebut baseline ekstraktif sebagai model LLM generatif).
3. **Option 3 — Supervised Tool-Using AI Agent:** Agen AI dengan skema alat tervalidasi, kontrol izin RBAC, dan **Human Approval Gate** wajib sebelum mengeksekusi aksi eksternal apa pun.
4. **Option 4 — Multimodal Sensor–Image Intelligence:** Penggabungan sinyal citra CNN dan telemetri/teks melalui *Early*, *Late*, atau *Cross-Attention Fusion* beserta uji ketahanan kehilangan modalitas di atas dataset sintetis/berlisensi.
5. **Option 5 — Computer Vision for Smart Agriculture:** Deteksi atau klasifikasi kondisi visual tanaman menggunakan CNN/YOLO teroptimasi beserta analisis error (dengan penafian tegas bahwa deteksi laboratorium **tidak** mengklaim keberhasilan lapangan nyata tanpa validasi lapangan).
6. **Option 6 — Intelligent Wireless Network Analytics:** Analisis deret waktu kualitas jaringan atau deteksi gangguan telemetri memakai metrik jaringan yang tepat dan tersedia (wajib membedakan tegas antara **SNR** dan **SINR**).
7. **Option 7 — Responsible AI Evaluation Platform:** Alat audit otomatis untuk *Fairness*, *Risk Register*, *Privacy Minimization*, dan pembuatan **Model Card** (tanpa klaim sertifikasi kepatuhan regulasi formal).
8. **Option 8 — Reinforcement Learning for Resource Allocation:** Simulasi Markov Decision Process (*Q-Learning* / *Policy Gradient*) untuk penjadwalan atau alokasi sumber daya terisolasi tanpa kontrol jaringan nyata tanpa otorisasi.

---

## PART C — DESAIN ARSITEKTUR & PERSYARATAN SISTEM

Sebuah sistem AI Capstone yang baik dibangun dengan pemisahan tanggung jawab (*modular architecture*) yang ringan, jelas kontrak datanya, dan terhindar dari arsitektur terdistribusi (*microservices/cloud*) yang tidak diperlukan pada skala prototipe ilmiah:

```text
+--------------------+     +-----------------------+     +------------------------+
| Data Generator /   | --> | Zero-Leakage Splitter | --> | Baseline & Candidate   |
| Provenance & Dict  |     | (Train / Val / Test)  |     | Selection (Val F1)     |
+--------------------+     +-----------------------+     +------------------------+
                                                                     |
                                                          Save .pkl + SHA-256 Hash
                                                                     v
+--------------------+     +-----------------------+     +------------------------+
| Risk Register &    | <-- | PSI / KS Drift        | <-- | Validated Local        |
| Model Card         |     | Monitoring Simulation |     | Inference + Escalation |
+--------------------+     +-----------------------+     +------------------------+
```

Setiap desain arsitektur wajib mendefinisikan:
1. **Functional Requirements:** Tugas inferensi utama, kontrak skema input/output, format hasil, dan mekanisme penolakan/eskalasi.
2. **Nonfunctional Requirements:** Target reprodusibilitas (`seed`), anggaran ukuran artefak & latensi P95 CPU, serta integritas checksum.
3. **Security Controls:** Validasi batas tipe/nilai, pembatas *trust boundary*, serta verifikasi integritas artefak sebelum deserialisasi.

---

## PART D — DELAPAN TAHAP PENGEMBANGAN & TUJUH MILESTONE

### Delapan Tahap Rekayasa Capstone
1. **Stage 1 — Problem Formulation:** *Problem statement*, latar belakang, pertanyaan penelitian, *intended use*, dan metrik sukses.
2. **Stage 2 — Dataset & Requirements:** Asal data (*provenance*), lisensi, *Data Dictionary*, audit kualitas, dan kebutuhan fungsional/nonfungsional.
3. **Stage 3 — Architecture:** Diagram blok sistem, pemilihan kandidat algoritma, justifikasi, dan protokol evaluasi.
4. **Stage 4 — Implementation:** Kode Python modular, konfigurasi terpusat, integrasi pipeline, dan suite unit test.
5. **Stage 5 — Experimentation:** Hasil baseline pembanding, validasi lintas kandidat, serta pengukuran ukuran & latensi.
6. **Stage 6 — System Evaluation:** Evaluasi akhir *held-out*, contoh kasus gagal (*error analysis*), ketahanan, dan keterbatasan.
7. **Stage 7 — Responsible AI:** *Risk Register*, *Threat Assessment*, privasi, mekanisme *Human Oversight*, dan *Model Card*.
8. **Stage 8 — Documentation & Presentation:** Laporan teknis lengkap, panduan reprodusibilitas, demo langsung, dan kesimpulan jujur.

### Tujuh Milestone & Kriteria Penerimaan (*Acceptance Criteria*)

| Milestone | Fokus Serahan | Kriteria Penerimaan Minimal |
| :--- | :--- | :--- |
| **M1. Proposal** | Rumusan masalah & opsi | *Intended use*, risiko awal, & metrik sukses relevan disetujui pengajar. |
| **M2. Dataset & Architecture** | Data & rancangan sistem | Data berlisensi/sintetis + *Data Dictionary* + diagram arsitektur lengkap. |
| **M3. Initial Prototype** | Baseline & data split | Pipeline pemecahan bebas *leakage* & hasil *baseline* dapat dijalankan ulang. |
| **M4. System Integration** | Model & antarmuka lokal | Pipeline terhubung end-to-end dengan penjagaan validasi input & checksum. |
| **M5. Testing & Evaluation** | Pengujian & analisis | Unit test lolos, metrik akhir tercatat, *error analysis* & benchmark CPU selesai. |
| **M6. Final Documentation** | Risk, Model Card & README | Risk Register, Model Card lengkap, serta notebook berjalan bersih tanpa error. |
| **M7. Final Presentation** | Sidang / demo akademik | Demo live reprodusibel serta pertanggungjawaban jujur atas keterbatasan sistem. |

---

## PART E — PROYEK REFERENSI END-TO-END (CPU VERIFIED)

Sebagai acuan baku standar kualitas, direktori `content/level-4-advanced-ai/chapter-40/code/` mengimplementasikan sistem **Intelligent Sensor Monitoring with Machine Learning, Evaluation & Responsible AI** secara terintegrasi dengan utilitas Chapter 19, 36, 37, dan 39.

### 1. Data Provenance, Data Dictionary & Zero-Leakage Splits
- **Dataset:** `1,500` sampel telemetri motor sintetis (`seed=42`, rasio *Fault Warning* `26.60%`) dengan `6` fitur terstandar di [`sensor_data_dictionary.json`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-40/data/sensor_data_dictionary.json): `temperature_c`, `vibration_rms`, `current_amp`, `acoustic_db`, `rpm_deviation`, dan `voltage_ripple_v`.
- **Pencegahan Kebocoran (`preprocessing.py`):** Data dipecah berstrata ke `Train (900)`, `Validation (300)`, dan `Held-Out Test (300)`. `StandardScaler` di-*fit* **hanya** pada `Train Set`.

### 2. Perbandingan Baseline, Kandidat & Profil Sumber Daya CPU
Seleksi model dilakukan murni pada **Validation Set (`N=300`)** dalam [`candidate_models.py`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-40/code/candidate_models.py):

| Model Kandidat | Val Accuracy | Val F1-Score | Val ROC-AUC | Ukuran Bundel `.pkl` | Mean Latency (P95) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (most_frequent)` | `73.33%` | `0.0000` | `0.5000` | — | — |
| **`LogisticRegression` (Terpilih)** | **`95.33%`** | **`0.9146`** | **`0.9897`** | **`1.15 KB`** | `0.0848 ms` (`0.1503 ms`) |
| `MLPClassifier (32, 16)` | `94.33%` | `0.8970` | `0.9883` | `39.22 KB` | `0.0747 ms` (`0.1232 ms`) |
| `RandomForestClassifier (50 trees)` | `94.00%` | `0.8875` | `0.9854` | `313.66 KB` | `1.8562 ms` (`2.2855 ms`) |

Artefak pemenang (`LogisticRegression`) disimpan disertai tanda integritas kriptografis **SHA-256** (`2e6ebc8046ef3835...`) agar tidak dapat dideserialisasi apabila berkas termodifikasi.

### 3. Evaluasi Akhir, Explainability & Analisis Kesalahan (`evaluation_and_error_analysis.py`)
- **Held-Out Test Set (`N=300`):** Accuracy `96.00%`, Precision `0.9474`, Recall `0.9000`, F1-Score `0.9231`, ROC-AUC `0.9937`.
- **Matriks Kebingungan:** `[[216, 4], [8, 72]]` (total `12` kesalahan dari `300` sampel uji: `4` *False Positives* dan `8` *False Negatives* di daerah batas noise).
- **Explainability (*Permutation Importance*):** Pendorong prediksi terkuat adalah `vibration_rms` (`0.1298`), diikuti `temperature_c` (`0.0824`), dan `acoustic_db` (`0.0769`), sejalan dengan model fisik kerusakan bearing motor pada generator telemetri. Grafik tersimpan pada `experiments/capstone_evaluation_chart.png`.

### 4. Inferensi Tervalidasi, Pemantauan Drift & Responsible AI
- **Penjagaan Skema (`local_inference.py`):** Memblokir muatan `NaN`, `Inf`, tipe boolean tersamar, atau angka di luar Data Dictionary (`REJECTED_INVALID_INPUT`, otomatis memicu `human_escalation=True`), serta menaikkan eskalasi bila `confidence < 0.65`.
- **Simulasi Drift (`monitoring_and_drift.py`):**
  - *Nominal Stream:* Acc `96.00%` | `Max PSI = 0.0844` | `KS(Temp) = 0.0522` | `Alert = False`.
  - *Drifted Stream (`+9.5C` & `+1.1 mm/s`):* Acc `94.33%` | `Max PSI = 2.0395` (`vibration_rms`) | `KS(Temp) = 0.4478` | **`Alert = True`**.
- **Risk Register & Model Card:** Mencatat `4` risiko spesifik (`CAP-R1..CAP-R4`) ke NIST AI RMF, menerbitkan [`capstone_model_card.json`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-40/experiments/capstone_model_card.json) dengan daftar eksplisit *Prohibited Use*, serta memverifikasi **4/4 unit tests** (`test_capstone_suite.py`) dan total waktu pipeline terpadu **`1.67s`** pada CPU lokal.

---

## PART F — 14 PERSYARATAN ILMIAH & RUBRIK PENILAIAN 100 POIN

### Empat Belas Persyaratan Ilmiah Wajib
Setiap proyek Capstone wajib memuat: (1) definisi masalah jernih, (2) *data provenance* terlacak, (3) pemilihan model terjustifikasi, (4) pembandingan *baseline*, (5) pencegahan *data leakage*, (6) reprodusibilitas berbasis `seed`, (7) **metrik evaluasi relevan sesuai jenis tugas** (*jangan memaksakan akurasi untuk RL, klasterisasi, atau RAG/generatif*), (8) *error analysis* konkret, (9) analisis sumber daya/latensi, (10) asesmen keamanan, (11) dokumentasi *Responsible AI* (*Risk Register* & *Model Card*), (12) pengungkapan keterbatasan secara jujur, (13) referensi akurat, dan (14) kesimpulan bebas klaim berlebih.

### Latihan & Panduan Praktikum
Instruksi eksekusi lengkap proyek referensi, soal **latihan refleksi bertingkat beserta pembahasannya**, rincian lengkap **Rubrik Penilaian 100 Poin** (*Problem Formulation 15 + Dataset/Architecture 15 + Implementation 25 + Evaluation/Analysis 20 + Security/Responsible AI 15 + Documentation/Presentation 10 = 100*) yang setara bagi seluruh domain proyek tanpa memihak ukuran model, tersedia pada [`practice.md`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-40/practice.md) dan [`chapter_40_companion.ipynb`](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-4-advanced-ai/chapter-40/notebooks/chapter_40_companion.ipynb).

### Referensi Akademik Capstone
1. Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS 2015. https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
2. Breck, E., et al. (2017). *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*. IEEE Big Data. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
3. NIST (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. https://doi.org/10.6028/NIST.AI.100-1
4. Mitchell, M., et al. (2019). *Model Cards for Model Reporting*. ACM FAT* 2019. https://arxiv.org/abs/1810.03993

