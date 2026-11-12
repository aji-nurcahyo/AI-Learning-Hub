# Panduan Praktik Laboratorium: Deep Learning Capstone Project

Selamat datang di panduan proyek akhir **Level 3 Deep Learning Capstone** (Bab 30). Modul ini memandu Anda merancang, mengimplementasikan, memvalidasi, dan mengomunikasikan proyek rekayasa kecerdasan buatan komprehensif yang mengintegrasikan seluruh materi Level 3 (Bab 21 hingga Bab 29) berbasis **Python**, **NumPy**, **Pandas**, **Scikit-learn**, dan **PyTorch** pada jalur komputasi CPU: mulai dari perumusan hipotesis dan kriteria sukses terukur, pemuatan data dan partisi 3-way (*Train/Val/Test*) bebas kebocoran (*zero data leakage*), penetapan baseline acuan performa minimum, perancangan arsitektur deep learning modular dengan koneksi residu, siklus pelatihan profesional dengan gradient clipping dan early stopping, evaluasi diagnostik independen pada data uji terisolasi, diseksi kesalahan (*error analysis*), audit etika keadilan subgrup (*Responsible AI*), hingga penyusunan laporan Model Card dan pemulihan checkpoint teruji secara deterministik.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- **Python**: Versi 3.10+ (Direkomendasikan Python 3.12).
- **Pustaka Utama**: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `scikit-learn>=1.3.0` (teruji 1.9.1), `pandas>=2.0.0` (teruji 2.2.3), `numpy>=2.0.0` (teruji 2.5.4).
- **Akselerator Perangkat Keras**: Bersifat opsional. Seluruh 8 berkas skrip proyek acuan (*Reference Implementation*) dirancang khusus agar dapat dieksekusi secara instan dan deterministik pada CPU komputer standar dalam waktu **< 5 detik** tanpa mewajibkan GPU, tanpa model unduhan raksasa, dan tanpa API berbayar.

### Struktur Direktori Bab 30:
```
content/level-3-deep-learning/chapter-30/
├── code/
│   ├── README.md
│   ├── dataset_loader.py
│   ├── baseline_model.py
│   ├── deep_learning_model.py
│   ├── training_engine.py
│   ├── evaluation_diagnostics.py
│   ├── checkpoint_manager.py
│   ├── responsible_ai_audit.py
│   └── capstone_pipeline_runner.py
├── experiments/
│   ├── results_ch30.json
│   ├── capstone_best_model.pt
│   └── capstone_model_checkpoint.pt
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_30_companion.ipynb
├── slides/
│   ├── build_slides.py
│   └── deck.json
├── index.md
├── practice.md
├── metadata.json
└── audit.md
```

---

## 2. Delapan Tahapan Siklus Hidup Proyek Capstone

### Tahap 1: Perumusan Masalah & Hipotesis Rekayasa
- Tentukan domain masalah nyata dengan batasan ruang lingkup yang jelas.
- Rumuskan hipotesis ilmiah (*e.g.*, *"Apakah arsitektur Deep MLP dengan koneksi residu mampu mengungguli model regresi logistik sebesar $\ge 3\%$ akurasi pada data telemetri getaran dan suhu sensor industri?"*).
- Tetapkan metrik keberhasilan kuantitatif: Akurasi $> 90\%$, Macro F1 $> 0.90$, dan latensi inferensi CPU $< 10$ ms per batch.

### Tahap 2: Dataset Provenance, Etika & Eksplorasi
- Jalankan skrip pemuatan dan partisi data:
```bash
python content/level-3-deep-learning/chapter-30/code/dataset_loader.py
```
- **Fokus Pengamatan**:
  - Dokumentasikan asal-usul data (*provenance*), format fitur, dan lisensi penggunaan.
  - Terapkan partisi 3-way bertingkat (*Stratified Split*): Training 70%, Validation 15%, Test 15%.
  - Buktikan independensi himpunan data uji (*disjoint index sets*) untuk menjamin ketiadaan kebocoran data (*zero data leakage*).

### Tahap 3: Model Acuan Pembanding (Baseline Benchmark)
- Bangun patokan performa minimum sebelum melatih model deep learning yang kompleks:
```bash
python content/level-3-deep-learning/chapter-30/code/baseline_model.py
```
- **Fokus Pengamatan**:
  - Penskalaan fitur (`StandardScaler`) wajib di-*fit* **hanya pada Training Set**, lalu men-transformasi Validation dan Test Set.
  - Latih model regresi logistik terstandarisasi.
  - Catat metrik acuan: Akurasi Test **88.67%** dan Macro F1 **0.8841**. Model deep learning wajib melampaui angka ini!

### Tahap 4: Arsitektur Model Deep Learning PyTorch
- Desain arsitektur jaringan saraf mendalam berorientasi objek:
```bash
python content/level-3-deep-learning/chapter-30/code/deep_learning_model.py
```
- **Fokus Pengamatan**:
  - Periksa integrasi tumpukan blok residu (*residual skip connections*) untuk memperlancar aliran gradien.
  - Pastikan keluaran akhir berupa logits mentah tanpa aktivasi Softmax prematur (agar kompatibel dengan `nn.CrossEntropyLoss`).
  - Verifikasi kesesuaian dimensi tensor masukan `(Batch, 4)` ke keluaran `(Batch, 3)`.

### Tahap 5: Siklus Pelatihan Profesional & Early Stopping
- Latih model dengan optimasi modern dan pencegahan overfitting:
```bash
python content/level-3-deep-learning/chapter-30/code/training_engine.py
```
- **Fokus Pengamatan**:
  - Amati optimasi AdamW dengan pembatasan norma gradien (*gradient clipping* $\|g\| \le 1.0$).
  - Perhatikan fase validasi yang terisolasi penuh (`model.eval()` dan `torch.no_grad()`).
  - Amati kerja *Early Stopping*: model memulihkan bobot terbaik (*best checkpoint*) saat Validation Loss terendah tercapai.

### Tahap 6: Evaluasi Akhir Data Uji & Error Analysis
- Lakukan pengujian akhir satu kali pada Test Set yang selama ini disembunyikan:
```bash
python content/level-3-deep-learning/chapter-30/code/evaluation_diagnostics.py
```
- **Fokus Pengamatan**:
  - Catat hasil evaluasi: Akurasi Test **93.33%** dan Macro F1 **0.9328** (mengungguli baseline sebesar $+4.66\%$).
  - Periksa matriks konfusi antarkelas (Normal vs Degraded vs Kritis).
  - Bedah diseksi galat (*error analysis*): telusuri sampel yang salah diprediksi untuk memahami kelemahan batas keputusan model (*boundary cases*).

### Tahap 7: Manajemen Checkpoint & Reproducibility
- Simpan dan pulihkan model secara aman:
```bash
python content/level-3-deep-learning/chapter-30/code/checkpoint_manager.py
```
- **Fokus Pengamatan**:
  - Simpan berkas checkpoint `.pt` yang memuat bobot model (`state_dict`), parameter scaler, dan metadata skema.
  - Muat kembali checkpoint dan buktikan bahwa hasil prediksi model reloaded memiliki deviasi $< 1.0 \times 10^{-7}$ (identik bitwise).

### Tahap 8: Audit Responsible AI & Keadilan Subgrup
- Verifikasi kepatuhan etika dan ketahanan operasional:
```bash
python content/level-3-deep-learning/chapter-30/code/responsible_ai_audit.py
```
- **Fokus Pengamatan**:
  - Audit disparitas performa di antara subgrup unit mesin (UNIT-01 s/d UNIT-10): disparitas Macro F1 terbukti bernilai $0.1624 \le 0.30$ (**Lolos Uji Keadilan Subgrup**).
  - Uji ketahanan derau sensor (*noise robustness*): suntikkan derau Gauss $\mathcal{N}(0, \sigma^2)$ dan amati penurunan performa yang bertahap (*graceful degradation*).

---

## 3. Enam Pilihan Jalur Topik Proyek Capstone

Mahasiswa dapat memilih salah satu dari 6 jalur topik resmi yang selaras dengan materi Bab 21–29:

1. **Option 1 — CNN-Based Image Classification**:
   - Membangun pipeline klasifikasi citra terstruktur (misal dataset tekstil, citra sel, atau daun tanaman).
   - Menggunakan arsitektur CNN modular dengan normalisasi spasial, data augmentation, evaluasi matriks konfusi, dan visualisasi feature maps.
2. **Option 2 — YOLO Object Detection Application**:
   - Mengembangkan aplikasi deteksi objek pada citra atau frame video berlisensi terbuka.
   - Mengonfigurasi representasi bounding box `xyxy`/`xywh`, penapisan Non-Maximum Suppression (NMS), evaluasi mAP@0.5, dan analisis trade-off confidence threshold.
3. **Option 3 — Sensor Sequence Classification**:
   - Memodelkan data deret waktu telemetri sensor getaran/arus menggunakan PyTorch Simple RNN, LSTM, atau GRU.
   - Menerapkan time-series splitting (tanpa pengacakan masa depan), visualisasi aktivasi gerbang, dan perbandingan parameter model sekuensial.
4. **Option 4 — Text Classification with Neural Networks**:
   - Membangun sistem klasifikasi teks ulasan dokumen menggunakan embedding padat dan BiLSTM atau modul atensi.
   - Menerapkan tokenisasi subword, pemeliharaan kata negasi, benchmark vs baseline TF-IDF, dan analisis galat semantik.
5. **Option 5 — Small Transformer Text Generation**:
   - Mengimplementasikan model Decoder-Only Transformer mini (TinyGPT) untuk tugas generasi teks domain spesifik.
   - Melatih model pada korpus teks edukasi, membandingkan decoding Greedy vs Temperature vs Top-p, dan mengaudit halusinasi luaran.
6. **Option 6 — Applied Computer Vision Monitoring**:
   - Merancang pipeline pemrosesan video cerdas untuk penghitungan objek melintasi garis maya (*Virtual Tripwire*).
   - Menggabungkan deteksi citra, pelacakan berbasis Centroid Tracker, mitigasi penghitungan ganda, dan pengukuran throughput FPS pada CPU.

---

## 4. Enam Tonggak Kerja Proyek (Milestones)

Setiap mahasiswa wajib mematuhi 6 tonggak penyerahan berkala:

| Milestone | Target Waktu | Deliverable Utama | Kriteria Keberhasilan (Acceptance Criteria) |
|---|:---:|---|---|
| **M1: Proposal** | Minggu 1 | Dokumen proposal (1–2 halaman) | Rumusan masalah lugas, hipotesis ilmiah jelas, metrik terukur, stakeholders teridentifikasi. |
| **M2: Data & Design** | Minggu 2 | Skrip pemuatan data & EDA | Provenance dan lisensi legal, partisi 3-way bebas kebocoran, data dictionary lengkap. |
| **M3: Initial Model** | Minggu 3 | Skrip baseline & arsitektur model | Model baseline acuan berjalan, arsitektur deep learning PyTorch terdefinisi tanpa galat tensor. |
| **M4: Validation** | Minggu 4 | Training engine & kurva konvergensi | Pelatihan konvergen stabil, gradient clipping aktif, early stopping bekerja menahan overfitting. |
| **M5: Final Evaluation** | Minggu 5 | Evaluasi Test Set & Audit AI | Pengujian Test Set sekali pakai, error analysis diseksi sampel galat, audit keadilan subgrup. |
| **M6: Final Presentation** | Minggu 6 | Laporan ilmiah, Model Card & Slide | Seluruh kode 100% reproduktif pada CPU, slide presentasi siap uji sidang, laporan lengkap. |

---

## 5. Rubrik Penilaian Capstone (Total: 100 Poin)

| Kriteria Penilaian | Bobot | Indikator Skor Penuh (Full Credit) | Indikator Skor Sebagian (Partial) | Indikator Kurang (Insufficient) |
|---|:---:|---|---|---|
| **1. Problem Formulation & Research Design** | 15 Poin | Perumusan masalah lugas, hipotesis ilmiah jelas, kriteria sukses kuantitatif terukur, dan pemilihan metode selaras dengan karakteristik masalah. | Masalah terdefinisi namun hipotesis atau metrik keberhasilan kurang terukur secara spesifik. | Masalah kabur, tidak ada hipotesis ilmiah, metrik keberhasilan tidak terdefinisi. |
| **2. Dataset Preparation & Responsible Data Usage** | 15 Poin | Provenance dan lisensi data transparan, data dictionary lengkap, partisi 3-way (*Train/Val/Test*) terbukti bebas kebocoran data (*zero leakage*). | Data terdokumentasi namun partisi data tidak terisolasi sempurna atau berpotensi kebocoran fitur. | Sumber data tidak jelas, lisensi abu-abu, terjadi kebocoran data (*data leakage*) fatal. |
| **3. Model Architecture & Implementation** | 25 Poin | Implementasi PyTorch modular berorientasi objek, penggunaan blok residu/normalisasi tepat, baseline acuan tersedia, kode bersih (*clean code*). | Model berjalan namun implementasi kaku, tidak modular, atau baseline acuan tidak sebanding. | Model gagal dijalankan, terjadi galat dimensi tensor, atau plagiasi kode tanpa pemahaman. |
| **4. Training, Validation & Evaluation** | 25 Poin | Pelatihan konvergen stabil, gradient clipping & early stopping aktif, evaluasi independen pada Test Set terisolasi, unggul vs baseline. | Pelatihan konvergen namun evaluasi akhir tercampur dengan data validasi atau tuning pada Test Set. | Model mengalami overfitting parah, divergensi loss, atau evaluasi fabrikasi metrik. |
| **5. Error Analysis & Responsible AI** | 10 Poin | Diseksi mendalam terhadap sampel salah prediksi (*error analysis*), audit disparitas keadilan subgrup, dan uji ketahanan derau sinyal. | Error analysis ada namun bersifat deskriptif dangkal tanpa telaah penyebab, audit keadilan terbatas. | Tidak ada error analysis, mengabaikan kegagalan model, tidak ada audit etika Responsible AI. |
| **6. Documentation, Model Card & Presentation** | 10 Poin | Dokumentasi replikasi 100% deterministik pada CPU, Model Card lengkap berstandar tata kelola AI, paparan presentasi terstruktur lugas. | Laporan lengkap namun reproduksibilitas membutuhkan perbaikan manual, slide presentasi padat teks. | Laporan tidak lengkap, kode tidak dapat direproduksi oleh penguji, presentasi tidak siap. |

---

## 6. Checklist Kesiapan Penyerahan Capstone

- [ ] Seluruh 8 skrip Python di direktori `content/level-3-deep-learning/chapter-30/code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Partisi 3-way data terbukti bebas kebocoran indeks (*disjoint sets*) pada `dataset_loader.py`.
- [ ] Model baseline (Logistic Regression) menetapkan batas bawah performa acuan pada `baseline_model.py`.
- [ ] Arsitektur jaringan saraf mendalam PyTorch modular dengan blok residu tervalidasi pada `deep_learning_model.py`.
- [ ] Siklus pelatihan konvergen dengan gradient clipping dan early stopping teruji pada `training_engine.py`.
- [ ] Evaluasi Test Set terisolasi menghasilkan akurasi di atas baseline dengan diseksi galat pada `evaluation_diagnostics.py`.
- [ ] Penyimpanan dan pemulihan checkpoint `.pt` terbukti mereproduksi luaran bitwise identik pada `checkpoint_manager.py`.
- [ ] Audit keadilan subgrup ($\Delta \text{F1} \le 0.30$) dan uji ketahanan derau sensor lulus pada `responsible_ai_audit.py`.
- [ ] Orkestrator `capstone_pipeline_runner.py` berjalan penuh dan menyimpan ringkasan ke `results_ch30.json`.
- [ ] Notebook pendamping `chapter_30_companion.ipynb` tervalidasi bersih dalam format v4.
- [ ] Slide Reveal.js `deck.json` memuat 22 slide semantik terstruktur rapi.
- [ ] Model Card dan laporan ilmiah telah memuat analisis limitasi dan pertimbangan etika kecerdasan buatan.

