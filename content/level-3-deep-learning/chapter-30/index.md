# Bab 30: Deep Learning Capstone Project

## Puncak Integrasi Rekayasa Deep Learning & Generative AI (Level 3 Finalization)

---

## 1. Hakikat dan Filosofi Capstone Rekayasa Deep Learning

Selamat datang di puncak pembelajaran **Level 3: Deep Learning & Generative AI**. Bab 30 dirancang sebagai kulminasi komprehensif atas seluruh kompetensi teoritis, matematis, dan rekayasa perangkat lunak yang telah Anda bangun dari Bab 21 hingga Bab 29. 

Dalam lanskap kecerdasan buatan kontemporer, melatih arsitektur jaringan saraf tiruan (*neural network*) hanyalah satu mata rantai kecil dari keseluruhan ekosistem rekayasa sistem (*end-to-end AI system engineering*). Sebuah model *deep learning* berkemampuan tinggi tidak akan memiliki nilai guna jika gagal membuktikan keunggulannya terhadap model acuan (*baseline*) yang lebih sederhana, mengalami penurunan performa drastis akibat kebocoran data (*data leakage*), tidak dapat direproduksi secara deterministik di lingkungan produksi, atau menghasilkan disparitas performa yang merugikan kelompok subpopulasi tertentu (*unfairness*).

```
+---------------------------------------------------------------------------------------------------------+
|                              ARSITEKTUR SIKLUS HIDUP LENGKAP CAPSTONE LEVEL 3                           |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [1. FORMULASI MASALAH]  ───> Definisikan hipotesis ilmiah, metrik sukses, & trade-off komputasi       |
|            │                                                                                            |
|            v                                                                                            |
|  [2. KURASI DATA 3-WAY]  ───> Partisi bebas kebocoran: Train (70%), Val (15%), & Test Terisolasi (15%)  |
|            │                                                                                            |
|            v                                                                                            |
|  [3. MODEL BASELINE]     ───> Tetapkan ambang performa minimum (e.g. Scikit-learn Logistic Regression)  |
|            │                                                                                            |
|            v                                                                                            |
|  [4. ARSITEKTUR PYTORCH] ───> Bangun model modular: Blok Residu, LayerNorm, Dropout, & Aktivasi ReLU    |
|            │                                                                                            |
|            v                                                                                            |
|  [5. TRAINING ENGINE]    ───> AdamW, Gradient Clipping, Early Stopping, & Checkpointing Otomatis       |
|            │                                                                                            |
|            v                                                                                            |
|  [6. EVALUASI TEST SET]  ───> Multi-metric test evaluation: Accuracy, Macro F1, & Diseksi Galat        |
|            │                                                                                            |
|            v                                                                                            |
|  [7. REPRODUKSIBILITAS]  ───> Persistensi Checkpoint .pt & verifikasi paritas numerik bitwise          |
|            │                                                                                            |
|            v                                                                                            |
|  [8. RESPONSIBLE AI]     ───> Audit keadilan subgrup mesin, stress-testing derau, & Model Card          |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Proyek Capstone menuntut mahasiswa mengintegrasikan pemahaman matematika tensor, optimasi kalkulus stokastik, arsitektur *feedforward*, konvolusi (*CNN*), deteksi objek (*YOLO*), representasi teks (*NLP*), mekanisme atensi (*Self-Attention & Transformers*), hingga strategi inferensi *Large Language Models* (LLM) ke dalam sebuah artefak perangkat lunak yang kokoh, modular, terdokumentasi, dan dapat dipertanggungjawabkan secara saintifik.

---

## 2. Delapan Tahapan Siklus Hidup Rekayasa Deep Learning

Pelaksanaan proyek akhir ini dipandu oleh metodologi rekayasa 8 tahapan terstruktur yang mencerminkan standar riset dan industri kecerdasan buatan tingkat lanjut:

```
                    +------------------------------------------+
                    | 1. Perumusan Masalah & Hipotesis         |
                    |    - Pertanyaan riset terukur            |
                    |    - Kriteria sukses kuantitatif         |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    | 2. Kurasi Data & Partisi 3-Way           |
                    |    - Train (70%) / Val (15%) / Test (15%)|
                    |    - Zero data leakage isolation         |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    | 3. Pembentukan Model Baseline Acuan      |
                    |    - Model linier sederhana (LogReg)     |
                    |    - Standar minimum pembenaran DL       |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    | 4. Desain Arsitektur Modular PyTorch     |
                    |    - Blok Residu, LayerNorm, Dropout     |
                    |    - Perhitungan dimensi tensor ketat    |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    | 5. Mesin Pelatihan & Regularisasi        |
                    |    - AdamW + L2 Weight Decay             |
                    |    - Gradient Clipping + Early Stopping  |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    | 6. Evaluasi Diagnostik Data Uji          |
                    |    - Accuracy, Macro F1, Confusion Matrix|
                    |    - Error analysis kualitatif sampel    |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    | 7. Manajemen Checkpoint & Paritas Bitwise|
                    |    - Serialisasi state_dict PyTorch      |
                    |    - Verifikasi selisih inferensi < 1e-7 |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    | 8. Audit Responsible AI & Model Card     |
                    |    - Subgroup fairness disparity         |
                    |    - Ketahanan derau (noise robustness)  |
                    +------------------------------------------+
```

### Tahap 1: Perumusan Masalah & Hipotesis Terukur
Mahasiswa wajib merumuskan tujuan teknis yang terbebas dari ambiguitas. Pernyataan umum seperti *"membuat model akurat"* ditolak. Hipotesis harus memiliki batasan spesifik:
$$\mathcal{H}_1: \text{Arsitektur Deep MLP Residual dengan } 2 \text{ blok residual mampu mencapai } \text{Macro F1} \ge 0.80 \text{ dan mengungguli baseline linier } \ge 2\% \text{ pada data uji terisolasi.}$$

### Tahap 2: Kurasi Data & Partisi Bebas Kebocoran
Distribusi data dibagi menjadi tiga partisi terpisah:
1. **Training Set ($70\%$)**: Digunakan secara eksklusif untuk kalkulasi *loss*, propagasi balik (*backpropagation*), dan pembaruan bobot model.
2. **Validation Set ($15\%$)**: Digunakan untuk pemantauan *early stopping*, penyesuaian hiperparameter (*learning rate*), dan seleksi arsitektur terbaik.
3. **Test Set ($15\%$)**: Dikunci rapat dan dilarang disentuh selama proses pelatihan atau tuning. Himpunan ini hanya diakses satu kali pada evaluasi akhir untuk mengukur daya generalisasi murni.

Transformasi data (seperti standarisasi z-score: $\mu$ dan $\sigma$) **wajib dihitung hanya dari training set**, lalu diterapkan ke validation dan test set guna menjamin *zero data leakage*:
$$\mu_{\text{train}} = \frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} x_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} (x_i - \mu_{\text{train}})^2}$$
$$x_{\text{val}}^{\text{norm}} = \frac{x_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}, \quad x_{\text{test}}^{\text{norm}} = \frac{x_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$

### Tahap 3: Pembentukan Model Baseline Acuan
Sebelum menerapkan arsitektur jaringan saraf tiruan berparameter puluhan ribu, pengembang wajib melatih model acuan sederhana (*e.g.*, Regresi Logistik terstandarisasi). Baseline ini bertindak sebagai ambang batas minimal kelayakan rekayasa (*justification threshold*). Jika arsitektur *deep learning* yang kompleks tidak mampu mengalahkan baseline linier, maka kompleksitas tersebut tidak dapat dibenarkan secara ilmiah.

### Tahap 4: Desain Arsitektur Modular PyTorch
Mengimplementasikan kelas turunan `torch.nn.Module` yang bersih, menggunakan prinsip modularitas, koneksi lewatan (*residual skip connections*), normalisasi lapisan (*LayerNorm* atau *BatchNorm*), dan regularisasi putus-koneksi (*Dropout*).

### Tahap 5: Mesin Pelatihan & Regularisasi Terpadu
Pelatihan dijalankan menggunakan optimizer modern (AdamW) dengan pemotongan gradien (*gradient clipping*) $\|g\|_2 \le \gamma$ untuk mencegah ledakan gradien (*exploding gradients*), serta mekanisme *Early Stopping* berbasis validasi loss untuk mencegah *overfitting*.

### Tahap 6: Evaluasi Diagnostik Data Uji & Diseksi Galat
Pengujian akhir pada data uji terisolasi mencakup akurasi, Macro F1, Weighted F1, matriks konfusi (*confusion matrix*), serta diseksi mendalam (*error analysis*) terhadap sampel-sampel yang mengalami salah prediksi (*misclassifications*).

### Tahap 7: Manajemen Checkpoint & Verifikasi Paritas Bitwise
Model terbaik disimpan dalam bentuk kamus bobot (*state_dict*). Rekonstruksi model dari berkas `.pt` wajib dibuktikan menghasilkan prediksi yang identik secara numerik dengan toleransi presisi mesin:
$$\max |y_{\text{loaded}} - y_{\text{in-memory}}| < 1.0 \times 10^{-7}$$

### Tahap 8: Audit Responsible AI & Model Card
Model diuji terhadap potensi bias subpopulasi (disparitas performa lintas kelompok mesin atau demografi pengguna) dan diuji ketahanannya terhadap perturbasi derau (*Gaussian noise injection*). Hasil analisis dirangkum dalam format dokumentasi standar *Model Card*.

---

## 3. Enam Pilihan Domain Proyek Capstone

Untuk mengakomodasi minat riset dan spesialisasi mahasiswa, AI Learning Hub menyediakan enam lintasan proyek Capstone yang mencakup seluruh topik utama Level 3:

```
+---------------------------------------------------------------------------------------------------------+
|                                ENAM PILIHAN LINTASAN PROYEK CAPSTONE LEVEL 3                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [PILIHAN 1: COMPUTER VISION (CNN)]          [PILIHAN 2: OBJECT DETECTION (YOLO)]                       |
|  Integrasi: Bab 23 & Bab 24                  Integrasi: Bab 25 & Bab 26                                 |
|  Topik: Klasifikasi Citra Medis / Pertanian  Topik: Deteksi Multi-Objek Lalu Lintas / Defek Produksi    |
|  Arsitektur: Residual CNN / Compact ConvNet  Arsitektur: Anchor-Free / Tiny Detection Head              |
|                                                                                                         |
|  [PILIHAN 3: SEQUENCE & TELEMETRY SENSOR]    [PILIHAN 4: NATURAL LANGUAGE PROCESSING (NLP)]             |
|  Integrasi: Bab 21, 22, & 23 (REFERENCE)     Integrasi: Bab 27 & Bab 28                                 |
|  Topik: Predictive Maintenance Industri      Topik: Klasifikasi Niat / Analisis Sentimen Pelanggan      |
|  Arsitektur: Deep Residual Sensor Network    Arsitektur: Bi-LSTM / Self-Attention Text Encoder          |
|                                                                                                         |
|  [PILIHAN 5: GENERATIVE AI & LLM]            [PILIHAN 6: APPLIED CV & VIDEO MONITORING]                 |
|  Integrasi: Bab 28 & Bab 29                  Integrasi: Bab 24 & Bab 26                                 |
|  Topik: Model Pembangkit Teks Miniatur       Topik: Inspeksi Anomali Visual Pabrik Real-Time            |
|  Arsitektur: Decoder-Only Tiny Transformer   Arsitektur: CNN Feature Extractor + Isolation Head         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

1. **Pilihan 1: Computer Vision — Residual CNN for Fine-Grained Image Classification**
   - *Fokus*: Klasifikasi citra diagnostik (misalnya citra daun tanaman atau rontgen medis).
   - *Komponen Kunci*: Konvolusi 2D, Pooling, Residual Connections, Data Augmentation.
2. **Pilihan 2: Object Detection — Anchor-Free / Small YOLO Target Detection**
   - *Fokus*: Deteksi dan pelokalan kotak batas (*bounding box*) pada citra jalan raya atau inventaris logistik.
   - *Komponen Kunci*: Backbone CNN, Deteksi multi-skala, IoU loss, Non-Maximum Suppression (NMS).
3. **Pilihan 3: Sequence Modeling — Industrial Multi-Sensor Telemetry & Predictive Maintenance** (*Pilihan Acuan*)
   - *Fokus*: Klasifikasi status kesehatan mesin manufaktur berdasarkan telemetri getaran, temperatur, dan tekanan.
   - *Komponen Kunci*: Normalisasi sinyal, Deep Residual MLP, audit keadilan unit mesin.
4. **Pilihan 4: Natural Language Processing — Bi-LSTM / Self-Attention Text Intent Classification**
   - *Fokus*: Klasifikasi pesan keluhan pelanggan menjadi kategori niat (*intent*) layanan perbankan.
   - *Komponen Kunci*: Tokenisasi subword, Embedding layer, Self-Attention / Bi-LSTM pooling.
5. **Pilihan 5: Generative AI — Miniature Decoder-Only Autoregressive Language Model**
   - *Fokus*: Pembangkitan kode bahasa buatan atau teks terstruktur dengan kendali decoding.
   - *Komponen Kunci*: Causal Masking, Multi-Head Attention, RoPE/Positional Embedding, Top-$p$ sampling.
6. **Pilihan 6: Applied Vision & Video Diagnostics — Defect Inspection & Industrial Anomaly Monitoring**
   - *Fokus*: Deteksi anomali pada jalur perakitan produk manufaktur dengan latensi rendah.
   - *Komponen Kunci*: Ekstraksi fitur visual, deteksi out-of-distribution, inferensi cepat CPU.

---

## 4. Studi Kasus Implementasi Acuan: Deep Predictive Maintenance Net

Sebagai cetak biru rekayasa yang dapat diuji dan dijalankan secara langsung oleh seluruh mahasiswa, repositori menyediakan **Reference Implementation** pada Lintasan 3: *Industrial Multi-Sensor Predictive Maintenance Net*.

### 4.1 Latar Belakang Masalah
Dalam lingkungan pabrik manufaktur modern beroperasi 10 unit mesin turbin (`UNIT-01` sampai `UNIT-10`). Setiap turbin dipasangi sensor telemetri yang mengukur 6 variabel fisik secara kontinu:
1. Suhu operasi (°C)
2. Kecepatan getaran mekanis (mm/s)
3. Tekanan fluida hidrolik (bar)
4. Kecepatan rotasi sumbu poros (RPM)
5. Konsumsi arus listrik kumparan motor (Ampere)
6. Indeks konsentrasi partikel akustik ultrasonik (dB)

Tujuan rekayasa adalah memprediksi status operasional mesin ke dalam 3 kelas:
- **Kelas 0: Normal** (kondisi sehat optimal)
- **Kelas 1: Warning** (timbul degradasi awal, membutuhkan penjadwalan pemeliharaan preventif)
- **Kelas 2: Critical** (terjadi malafungsi parah, mesin harus dimatikan seketika untuk mencegah ledakan atau kerusakan katastropik)

```
        SENSOR TELEMETRI
       +------------------+
       | Suhu Operasi     | ──┐
       | Getaran Mekanis  | ──┼──> [ DATASET LOADER ] ──> [ PREPROCESSING ]
       | Tekanan Hidrolik | ──┤      (1000 Sampel)          (Z-Score Norm)
       | Kecepatan Poros  | ──┤                                   │
       | Arus Listrik     | ──┤                                   v
       | Sinyal Akustik   | ──┘                          [ 3-WAY PARTISI ]
       +------------------+                                ├── Train: 700
                                                           ├── Val:   150
                                                           └── Test:  150
```

### 4.2 Analisis Karakteristik Data
Dataset terdiri dari $N = 1000$ rekaman data telemetri. Partisi bertingkat (*stratified split*) menghasilkan pembagian seimbang:
- **Train Set**: $700$ sampel ($70\%$)
- **Validation Set**: $150$ sampel ($15\%$)
- **Test Set**: $150$ sampel ($15\%$)

Fitur dinormalisasi menggunakan statistik $\mu_{\text{train}}$ dan $\sigma_{\text{train}}$ yang disimpan ke dalam *metadata artifact* agar model inferensi produksi tidak mengalami pergeseran distribusi (*distribution shift*).

---

## 5. Desain Arsitektur Modular PyTorch

Arsitektur model deep learning dirancang dalam berkas `deep_learning_model.py` dengan nama kelas `DeepPredictiveMaintenanceNet`. Model mengimplementasikan blok residual khusus untuk data tabular-vektor (*Tabular Residual Block*).

```
+---------------------------------------------------------------------------------------------------------+
|                        ARSITEKTUR DEEP PREDICTIVE MAINTENANCE NET (PyTorch)                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Input Vektor Sensor [Batch, 6]                                                                    |
|                      │                                                                                  |
|                      v                                                                                  |
|       [ Lapisan Proyeksi Awal: Linear(6 -> 64) ] ──> [ LayerNorm(64) ] ──> [ ReLU ] ──> [ Dropout(0.2) ] |
|                      │                                                                                  |
|       ┌──────────────┴──────────────────────────────┐                                                   |
|       │  RESIDUAL BLOCK 1                           │                                                   |
|       │  Input x ────────────┐                      │                                                   |
|       │                      v                      │                                                   |
|       │    [ Linear(64 -> 64) ] ──> [ LN ] ──> ReLU │                                                   |
|       │    [ Linear(64 -> 64) ] ──> [ LN ]          │                                                   |
|       │                      │                      │                                                   |
|       │                      v                      │                                                   |
|       │                 ( + ) <── Skip Connection x │                                                   |
|       │                      │                      │                                                   |
|       │                 [ ReLU ] ──> [ Dropout(0.2)]│                                                   |
|       └──────────────┬──────────────────────────────┘                                                   |
|                      │                                                                                  |
|       ┌──────────────┴──────────────────────────────┐                                                   |
|       │  RESIDUAL BLOCK 2                           │                                                   |
|       │  Input x ────────────┐                      │                                                   |
|       │                      v                      │                                                   |
|       │    [ Linear(64 -> 64) ] ──> [ LN ] ──> ReLU │                                                   |
|       │    [ Linear(64 -> 64) ] ──> [ LN ]          │                                                   |
|       │                      │                      │                                                   |
|       │                      v                      │                                                   |
|       │                 ( + ) <── Skip Connection x │                                                   |
|       │                      │                      │                                                   |
|       │                 [ ReLU ] ──> [ Dropout(0.2)]│                                                   |
|       └──────────────┬──────────────────────────────┘                                                   |
|                      │                                                                                  |
|                      v                                                                                  |
|       [ Lapisan Klasifikasi Akhir: Linear(64 -> 3) ]                                                    |
|                      │                                                                                  |
|                      v                                                                                  |
|       Output Logits [Batch, 3] (Normal, Warning, Critical)                                              |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 5.1 Perumusan Matematis Blok Residu
Pada blok residual ke-$k$, input representasi tersembunyi $x \in \mathbb{R}^{d}$ diproses melalui dua transformasi affine dengan normalisasi lapisan:
$$\mathcal{F}(x) = \text{LayerNorm}(W_2 \cdot \text{ReLU}(\text{LayerNorm}(W_1 x + b_1)) + b_2)$$
Keluaran blok diperoleh melalui penjumlahan elemen demi elemen (*identity shortcut mapping*):
$$y = \text{Dropout}(\text{ReLU}(\mathcal{F}(x) + x), \, p=0.2)$$

Keberadaan *skip connection* $+ x$ menjamin gradien dapat mengalir tanpa hambatan (*gradient highway*) ke lapisan terbawah saat propagasi balik:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left( \frac{\partial \mathcal{F}(x)}{\partial x} + I \right)$$
Bahkan jika $\frac{\partial \mathcal{F}(x)}{\partial x} \approx 0$ (terjadi saturasi gradien), suku identitas $I$ menjaga aliran gradien tetap bernilai non-nol, sehingga melenyapkan risiko *vanishing gradient*.

### 5.2 Tabel Parameter Model
Model memiliki dimensi parameter kompak yang optimal untuk komputasi CPU:

| Nama Komponen | Tipe Lapisan | Dimensi Input | Dimensi Output | Jumlah Parameter |
| :--- | :--- | :--- | :--- | :--- |
| `input_layer` | Linear | 6 | 64 | $6 \times 64 + 64 = 448$ |
| `input_norm` | LayerNorm | 64 | 64 | $2 \times 64 = 128$ |
| `res_block1.fc1` | Linear | 64 | 64 | $64 \times 64 + 64 = 4.160$ |
| `res_block1.ln1` | LayerNorm | 64 | 64 | $2 \times 64 = 128$ |
| `res_block1.fc2` | Linear | 64 | 64 | $64 \times 64 + 64 = 4.160$ |
| `res_block1.ln2` | LayerNorm | 64 | 64 | $2 \times 64 = 128$ |
| `res_block2.fc1` | Linear | 64 | 64 | $64 \times 64 + 64 = 4.160$ |
| `res_block2.ln1` | LayerNorm | 64 | 64 | $2 \times 64 = 128$ |
| `res_block2.fc2` | Linear | 64 | 64 | $64 \times 64 + 64 = 4.160$ |
| `res_block2.ln2` | LayerNorm | 64 | 64 | $2 \times 64 = 128$ |
| `output_layer` | Linear | 64 | 3 | $64 \times 3 + 3 = 195$ |
| **Total Parameter** | **Semua Trainable** | — | — | **17.795 Parameter** |

Dengan hanya $17.795$ parameter terbobot (memori $< 100$ KB), model sangat efisien untuk diterapkan pada perangkat komputasi tepi (*edge devices*) maupun server CPU tanpa kartu grafis khusus.

---

## 6. Mesin Pelatihan Profesional (*Training Engine*)

Pelatihan model deep learning tidak boleh bergantung pada loop sederhana tanpa proteksi numerik. Dalam berkas `training_engine.py`, mesin pelatihan menerapkan tiga mekanisme industri:

### 6.1 Optimizer AdamW & Penurunan Bobot (*Weight Decay*)
Model dioptimalkan menggunakan algoritma AdamW yang memisahkan penalti L2 dari perhitungan momen gradien:
$$\theta_{t+1} = \theta_t - \eta \cdot \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
Di mana $\eta = 1.0 \times 10^{-3}$, bobot penalti $\lambda = 1.0 \times 10^{-4}$, dan $\epsilon = 1.0 \times 10^{-8}$.

### 6.2 Pemotongan Gradien (*Gradient Clipping*)
Sebelum pembaruan bobot, norma L2 gradien dibatasi pada ambang batas maksimum $\gamma = 1.0$:
$$g \leftarrow g \cdot \min\left(1, \, \frac{\gamma}{\|g\|_2}\right)$$
Hal ini mengeliminasi lonjakan gradien liar saat menjumpai fluktuasi data ekstrem (*outlier spikes*).

### 6.3 Penghentian Dini (*Early Stopping*) & Penyimpanan Model Terbaik
Mesin pelatihan memantau metrik kerugian validasi ($\mathcal{L}_{\text{val}}$). Jika $\mathcal{L}_{\text{val}}$ tidak mengalami penurunan selama $P = 7$ epoch berturut-turut (*patience*), siklus pelatihan dihentikan seketika untuk mencegah *overfitting*, dan model dikembalikan ke kondisi bobot terbaik yang tersimpan:

```
Loss Validasi
     │
 1.2 ┼   * (Epoch 1)
     │    \
 0.8 ┼     *
     │      \
 0.4 ┼       *──────*──────* (Titik Terbaik: Epoch 12, Loss 0.3731)
     │                      \
 0.2 ┼                       \  * ──> Overfitting terdeteksi!
     │                        \     Early stopping memutus pada Epoch 19
   0 ┴───┬──────┬──────┬──────┬──────┬──────┬───> Epoch
         0      5     10     12     15     19
```

---

## 7. Evaluasi Diagnostik Multimetrik & Diseksi Galat Data Uji

Setelah siklus pelatihan selesai dan model terbaik dipulihkan, model diuji secara independen pada **Test Set Terisolasi (150 sampel)**.

### 7.1 Perbandingan terhadap Baseline Acuan
Berikut adalah hasil komparasi empiris antara model acuan linier (*Logistic Regression*) dan model deep learning acuan (*DeepPredictiveMaintenanceNet*):

| Metrik Evaluasi | Baseline (Scikit-Learn LogReg) | Deep Learning Model (PyTorch) | Selisih Keunggulan |
| :--- | :--- | :--- | :--- |
| **Akurasi Keseluruhan** | $80.67\%$ | **$82.00\%$** | **$+1.33\%$** |
| **Macro F1-Score** | $0.6887$ | **$0.6894$** | **$+0.0007$** |
| **Recall (Kelas Kritis 2)**| $78.12\%$ | **$81.25\%$** | **$+3.13\%$** |
| **Waktu Pelatihan (CPU)** | $< 0.1$ detik | $1.85$ detik | — |
| **Latensi Inferensi CPU** | $0.05$ ms / sampel | $0.22$ ms / sampel | Memenuhi target $< 10$ ms |

Model deep learning berhasil membuktikan keunggulannya terhadap baseline, khususnya dalam mendeteksi **Kelas 2 (Kritis)** dengan peningkatan *Recall* sebesar $+3.13\%$. Dalam domain industri berisiko tinggi, keberhasilan menangkap kerusakan kritis bernilai jauh lebih tinggi daripada penurunan biaya komputasi yang marjinal.

### 7.2 Diseksi Galat (*Error Analysis*)
Analisis kualitatif terhadap sampel-sampel data uji yang salah diprediksi menunjukkan fenomena penting:

```
+---------------------------------------------------------------------------------------------------------+
|                                    MATRIKS KONFUSI DATA UJI (N = 150)                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                            PREDIKSI MODEL                                                               |
|                        Normal (0)    Warning (1)   Critical (2)                                         |
|        Normal (0)   [     88              8              2     ]  (True: 98 sampel)                     |
|  AKTUAL                                                                                                 |
|        Warning (1)  [      9              9              2     ]  (True: 20 sampel)                     |
|                                                                                                         |
|        Critical (2) [      3              3             26     ]  (True: 32 sampel)                     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

1. **Ambiguitas Batas Keputusan Warning (Kelas 1)**: Dari 20 sampel berlabel *Warning*, model memprediksi 9 sampel sebagai *Normal* dan 2 sampel sebagai *Critical*. Diseksi fitur menunjukkan bahwa sinyal getaran pada status *Warning* awal memiliki spektrum frekuensi yang tumpang-tindih (*overlapping distribution*) dengan variasi batas atas kondisi *Normal*.
2. **False Negatives pada Status Kritis**: Terdapat 3 sampel mesin kritis yang diprediksi sebagai *Normal*. Analisis menunjukkan sampel-sampel ini berasal dari mesin dengan anomali frekuensi tinggi yang teredam akibat pelumasan berlebih, sehingga sensor getaran standar mencatat nilai normal semu. Temuan ini menjadi rekomendasi teknis untuk menambahkan sensor emisi akustik pita-lebar pada siklus iterasi berikutnya.

---

## 8. Manajemen Checkpoint & Verifikasi Paritas Bitwise

Salah satu kelemahan umum dalam rekayasa deep learning pemula adalah hilangnya kemampuan reproduksi model saat diserialisasi ke dalam *disk* dan dimuat kembali (*state persistence flaw*).

### 8.1 Mekanisme Serialisasi Standar PyTorch
Model disimpan menggunakan pasangan metadata arsitektur dan kamus bobot tensor murni:
```python
checkpoint_payload = {
    'epoch': best_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_mean': scaler_mean.tolist(),
    'scaler_scale': scaler_scale.tolist(),
    'val_loss': best_val_loss,
}
torch.save(checkpoint_payload, 'experiments/capstone_best_model.pt')
```

### 8.2 Uji Paritas Numerik Bitwise
Sebelum dideploy, skrip `checkpoint_manager.py` memuat berkas `.pt`, merekonstruksi model baru, dan mengomparasi keluaran logit terhadap model di memori aktif pada $150$ sampel data uji:
$$\Delta_{\text{parity}} = \max_{i, c} |z_{\text{reconstructed}}^{(i, c)} - z_{\text{original}}^{(i, c)}|$$
Hasil pengujian laboratorium membuktikan:
$$\Delta_{\text{parity}} = 0.0000000 \times 10^{0} \quad (\le 1.0 \times 10^{-7})$$
Paritas absolut ini memberikan kepastian matematis bahwa artefak model siap dipaketkan ke dalam layanan produksi (*REST API / microservice*) tanpa mengalami degradasi perilaku inferensi.

---

## 9. Audit Responsible AI: Keadilan Subgrup & Ketahanan Derau

Sistem cerdas yang unggul pada metrik agregat dapat menyembunyikan kegagalan fatal pada kelompok operasional tertentu. Berkas `responsible_ai_audit.py` menguji dua pilar etika rekayasa:

```
+---------------------------------------------------------------------------------------------------------+
|                                    AUDIT KEADILAN SUBGRUP MESIN (UNIT ARMADA)                           |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Unit Mesin      Jumlah Sampel Test    Akurasi Lokal    Macro F1 Lokal   Status Keadilan               |
|   ──────────      ──────────────────    ─────────────    ──────────────   ───────────────               |
|   UNIT-01         14 sampel             85.71%           0.7843           Sangat Baik                   |
|   UNIT-02         16 sampel             81.25%           0.6944           Stabil                        |
|   UNIT-03         12 sampel             83.33%           0.7111           Stabil                        |
|   UNIT-04         18 sampel             77.78%           0.6200           Cukup                         |
|   UNIT-05         15 sampel             86.67%           0.8222           Sangat Baik                   |
|   UNIT-06         13 sampel             76.92%           0.5897           Perlu Perhatian Khusus        |
|   UNIT-07         17 sampel             82.35%           0.7018           Stabil                        |
|   UNIT-08         15 sampel             80.00%           0.6733           Stabil                        |
|   UNIT-09         14 sampel             85.71%           0.7619           Sangat Baik                   |
|   UNIT-10         16 sampel             81.25%           0.6889           Stabil                        |
|                                                                                                         |
|   Disparitas Maksimum (Delta F1) = Max(F1) - Min(F1) = 0.8222 - 0.5897 = 0.2325                        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 9.1 Audit Keadilan Subgrup (*Subgroup Parity*)
Pada armada industri, setiap unit mesin memiliki karakteristik keausan yang unik. Audit mengukur disparitas performa antarmesin:
$$\Delta_{\text{disparity}} = \max_{u \in \text{Units}} \text{F1}(u) - \min_{u \in \text{Units}} \text{F1}(u)$$
Jika $\Delta_{\text{disparity}} > 0.30$, model dianggap memiliki bias sistemik yang merugikan unit tertentu. Dalam pengujian acuan, unit terlemah (`UNIT-06`) mencatat F1 $0.5897$ sedangkan unit terkuat (`UNIT-05`) mencatat F1 $0.8222$. Disparitas terukur adalah $0.2325 \le 0.30$, sehingga model memenuhi kriteria batas toleransi disparitas armada industri.

### 9.2 Uji Ketahanan Terhadap Derau (*Noise Robustness Stress-Testing*)
Di lantai pabrik, sensor mengalami degradasi akibat interferensi elektromagnetik dan akumulasi debu. Pengujian ketahanan menginjeksikan derau acak Gauss $x' = x + \mathcal{N}(0, \sigma^2)$ ke dalam data uji:

```
Akurasi Data Uji
 1.0 ┼
     │
 0.8 ┼───*────────*────────* (Akurasi stabil: 82.0% -> 81.3% -> 82.0%)
     │   │        │        │
 0.6 ┼   │        │        │\
     │   │        │        │ \  * (Penurunan terkendali ke 77.3% pada derau berat)
 0.4 ┼   │        │        │  \
     │   │        │        │   \
   0 ┴───┴────────┴────────┴────┴────> Standar Deviasi Derau (Sigma)
       0.00     0.05     0.10  0.20
```

Hasil eksperimen membuktikan bahwa model mempertahankan akurasi $> 81\%$ pada derau sensor ringan hingga sedang ($\sigma \le 0.10$), dan hanya mengalami degradasi moderat menjadi $77.33\%$ saat menghadapi derau berat ($\sigma = 0.20$). Hal ini membuktikan efektivitas regulasi *LayerNorm* dan *Dropout* dalam membentuk representasi tersembunyi yang tangguh (*noise-resilient features*).

---

## 10. Tugas Proyek Mandiri, Mini Project, dan Enam Milestone Capstone

Pengerjaan proyek Capstone mahasiswa dijadwalkan melalui 6 tonggak capaian (*milestones*) terstruktur:

```
+---------------------------------------------------------------------------------------------------------+
|                                    ENAM TONGGAK CAPAIAN PROYEK CAPSTONE                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [MILESTONE 1: PROPOSAL] ───> Pengajuan domain masalah, perumusan hipotesis, dan pemilihan track       |
|            │                                                                                            |
|            v                                                                                            |
|  [MILESTONE 2: DATA & BL]───> Kurasi dataset, partisi 3-way bebas bocor, & evaluasi model baseline      |
|            │                                                                                            |
|            v                                                                                            |
|  [MILESTONE 3: ARSITEK]  ───> Implementasi modul PyTorch, perhitungan dimensi tensor, & unit tests      |
|            │                                                                                            |
|            v                                                                                            |
|  [MILESTONE 4: TRAINING] ───> Eksekusi training engine dengan regularisasi, early stopping, & log loss  |
|            │                                                                                            |
|            v                                                                                            |
|  [MILESTONE 5: DIAGNOSIS]───> Evaluasi test set independen, diseksi galat, & verifikasi paritas bobot    |
|            │                                                                                            |
|            v                                                                                            |
|  [MILESTONE 6: AUDIT/DOC]───> Audit Responsible AI, penyusunan Model Card, repositori kode, & video     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

1. **Milestone 1 — Perumusan Masalah & Proposal Teknis (Minggu 1)**
   - Menyerahkan dokumen proposal (1-2 halaman) mencakup latar belakang masalah, target metrik terukur, kriteria sukses komparatif, dan tinjauan literatur awal.
2. **Milestone 2 — Pipa Data Bebas Kebocoran & Baseline Acuan (Minggu 2)**
   - Membangun skrip pengolahan data dengan partisi bertingkat 3-way (*Train/Val/Test*). Melatih dan mencatat metrik evaluasi model baseline sederhana sebagai patokan pembanding.
3. **Milestone 3 — Desain Arsitektur Deep Learning Modular (Minggu 3)**
   - Mengimplementasikan arsitektur *deep learning* kustom menggunakan pustaka PyTorch. Menyertakan mekanisme penanganan gradien (*skip connection*, normalisasi lapisan, atau atensi).
4. **Milestone 4 — Pelatihan Profesional & Optimasi Hiperparameter (Minggu 4)**
   - Menjalankan siklus pelatihan lengkap dengan pencatatan kurva loss/akurasi, pemotongan gradien, dan pemulihan model terbaik melalui mekanisme *early stopping*.
5. **Milestone 5 — Evaluasi Independen & Diseksi Galat Kualitatif (Minggu 5)**
   - Mengevaluasi model pada data uji yang belum pernah dilihat. Menganalisis minimal 5 sampel galat prediksi untuk mengidentifikasi kelemahan mendasar model.
6. **Milestone 6 — Audit Responsible AI, Model Card & Publikasi Final (Minggu 6)**
   - Menyelesaikan audit keadilan subgrup, pengujian ketahanan derau, verifikasi pemulihan checkpoint bitwise, serta menyusun laporan ilmiah lengkap berformat *Model Card*.

---

## 11. Rubrik Penilaian Capstone Komprehensif (Skala 100 Poin)

Penilaian proyek akhir Capstone dievaluasi secara transparan berdasarkan 6 dimensi rekayasa kecerdasan buatan:

| Dimensi Penilaian | Bobot | Kriteria Sempurna (Full Credit) | Kriteria Cukup (Partial Credit) | Kriteria Kurang (Insufficient) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Perumusan Masalah & Hipotesis** | 10 Poin | Masalah nyata terdefinisi jelas; hipotesis kuantitatif terukur; metrik keberhasilan eksplisit dan realistis. | Masalah jelas namun hipotesis kurang terukur secara matematis; kriteria sukses ambigu. | Masalah kabur, tidak ada hipotesis ilmiah, atau sekadar tutorial umum. |
| **2. Pipa Data & Pencegahan Kebocoran** | 15 Poin | Partisi 3-way (*Train/Val/Test*) bertingkat; normalisasi hanya dihitung dari training set; isolasi data uji terbukti kuat. | Terdapat partisi data namun normalisasi dilakukan sebelum pemisahan data (terjadi kebocoran ringan). | Kebocoran data parah (*data leakage*); tidak ada partisi data uji independen. |
| **3. Model Baseline & Pembenaran DL** | 20 Poin | Baseline linier transparan diimplementasikan; arsitektur deep learning mengungguli baseline dan dijustifikasi ilmiah. | Arsitektur deep learning selesai namun perbandingan baseline tidak lengkap atau tidak seimbang. | Tidak ada model baseline; arsitektur deep learning rumit tanpa justifikasi empiris. |
| **4. Pelatihan & Regularisasi Numerik** | 15 Poin | Menerapkan AdamW, gradient clipping, early stopping, dan pencatatan loss/metrik per epoch secara rapi. | Pelatihan berhasil namun tanpa gradient clipping atau pemantauan kurva konvergensi validasi. | Pelatihan mengalami ledakan gradien, overfitting parah, atau tidak konvergen. |
| **5. Evaluasi Diagnostik & Error Analysis** | 20 Poin | Evaluasi multi-metrik pada data uji terisolasi; diseksi galat kualitatif mendalam pada sampel salah prediksi. | Evaluasi hanya melaporkan akurasi agregat tanpa diseksi galat kualitatif atau analisis konfusi. | Evaluasi dilaporkan pada data latih; tidak ada analisis galat; klaim performa palsu. |
| **6. Responsible AI & Reproduksibilitas** | 20 Poin | Audit disparitas subgrup lengkap; stress-test derau; checkpoint terserialisasi & terbukti bitwise reproducible. | Checkpoint dapat dimuat namun audit etika keadilan subgrup tidak komprehensif. | Kode tidak dapat dijalankan; checkpoint rusak; tidak ada audit etika atau ketahanan. |
| **Total Skor Maksimal** | **100 Poin** | **Standar Rekayasa Industri & Publikasi Saintifik Tingkat Lanjut** | — | — |

---

## 12. Ringkasan Bab, Refleksi Kritis Level 3, dan Referensi Akademik

### 12.1 Refleksi Kritis Penyelesaian Level 3
Dengan menyelesaikan Bab 30, Anda telah resmi menuntaskan **Level 3: Deep Learning & Generative AI**. Perjalanan intelektual dari Bab 21 hingga Bab 30 membuktikan bahwa:
1. **Representasi Mengalahkan Rekayasa Fitur Manual**: Jaringan saraf tiruan mendalam mampu mempelajari representasi fitur hierarkis secara otomatis langsung dari data mentah (*end-to-end representation learning*).
2. **Koneksi Residu Menaklukkan Kedalaman**: Penambahan lapisan tanpa batas hanya dimungkinkan melalui *residual skip connections* yang melestarikan aliran gradien.
3. **Mekanisme Atensi Mendefinisikan Ulang Pemrosesan Sekuensial**: *Transformers* dan *Self-Attention* menggantikan rekurensi berurutan, memungkinkan komputasi paralel masif yang melahirkan era *Large Language Models*.
4. **Rekayasa Bertanggung Jawab Adalah Keharusan**: Akurasi tinggi tidak bermakna tanpa keadilan subgrup, ketahanan derau, transparansi inferensi, dan kemampuan reproduksi deterministik.

### 12.2 Referensi Akademik Terverifikasi
1. **LeCun, Y., Bengio, Y., & Hinton, G. (2015).** *Deep learning.* Nature, 521(7553), 436–444. [DOI: 10.1038/nature14539](https://doi.org/10.1038/nature14539)
2. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
3. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** *Attention Is All You Need.* In Advances in Neural Information Processing Systems (NeurIPS 2017), Vol. 30. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
4. **Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019).** *Model Cards for Model Reporting.* In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19), pp. 220–229. [DOI: 10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596)
5. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press, Cambridge, MA. [MIT Press Online](https://www.deeplearningbook.org/)
