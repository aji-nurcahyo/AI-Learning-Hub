# Panduan Praktik Laboratorium: Transfer Learning, Fine-Tuning & LoRA

Selamat datang di panduan laboratorium **Chapter 31: Transfer Learning & Fine-Tuning**. Modul ini memandu Anda mempraktikkan secara langsung teknik adaptasi model pembelajaran mendalam modern berbasis **Python** dan **PyTorch** pada jalur komputasi CPU: mulai dari ekstraksi fitur (*Feature Extraction*), pembekuan bobot (*Layer Freezing*), *Linear Probing*, *Partial Fine-Tuning* dengan *differential learning rates*, mitigasi *Catastrophic Forgetting* pada *Full Fine-Tuning*, simulasi *Domain Shift* dan *Negative Transfer*, hingga perancangan adaptor rank rendah *Low-Rank Adaptation* (LoRA) dari dasar matematis sampai implementasi modular produksi.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- **Python**: Versi 3.10+ (Teruji pada Python 3.12.10).
- **Pustaka Utama**: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `numpy>=2.0.0`, `scikit-learn>=1.3.0`.
- **Akselerator Perangkat Keras**: Bersifat opsional. Seluruh 12 berkas skrip praktik dirancang khusus agar dapat dieksekusi secara mandiri, deterministik, dan instan pada prosesor CPU komputer standar tanpa ketergantungan GPU, tanpa model unduhan raksasa, dan tanpa API berbayar.

### Struktur Direktori Bab 31:
```
content/level-4-advanced-ai/chapter-31/
├── code/
│   ├── README.md
│   ├── transfer_learning_basics.py
│   ├── pretrained_feature_extraction.py
│   ├── freeze_unfreeze_layers.py
│   ├── linear_probing.py
│   ├── partial_fine_tuning.py
│   ├── full_fine_tuning.py
│   ├── domain_shift_demo.py
│   ├── negative_transfer_demo.py
│   ├── lora_from_scratch.py
│   ├── lora_parameter_count.py
│   ├── lora_pytorch_demo.py
│   └── fine_tuning_comparison.py
├── experiments/
│   └── results_ch31.json
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_31_companion.ipynb
├── slides/
│   ├── build_slides.py
│   └── deck.json
├── index.md
├── practice.md
├── metadata.json
└── audit.md
```

---

## 2. Dua Belas Tahapan Praktik Terpadu

### Langkah 1: Eksplorasi Konsep Dasar Transfer Learning
Jalankan skrip dasar perbandingan pelatihan from scratch vs transfer learning:
```bash
python content/level-4-advanced-ai/chapter-31/code/transfer_learning_basics.py
```
**Fokus Pengamatan**:
- Perhatikan bagaimana model yang telah melalui pra-pelatihan (*pretraining*) pada tugas sumber (Source Task) memiliki representasi awal yang unggul saat berhadapan dengan data target yang terbatas (low-data regime).
- Amati selisih akurasi validasi (*delta accuracy*) yang membuktikan manfaat *feature reuse*.

### Langkah 2: Ekstraksi Fitur & Perilaku Runtime Mode
Jalankan skrip inspeksi pembekuan parameter dan pengujian mode runtime:
```bash
python content/level-4-advanced-ai/chapter-31/code/pretrained_feature_extraction.py
```
**Fokus Pengamatan**:
- Pahami cara membekukan bobot via `param.requires_grad = False`.
- Buktikan bahwa model dengan bobot beku tetap menghasilkan variansi output jika berada dalam mode `model.train()` akibat keaktifan lapisan Dropout dan pembaruan running statistics pada BatchNorm.
- Verifikasi bahwa memanggil `model.eval()` adalah keharusan mutlak untuk memastikan inferensi deterministik.

### Langkah 3: Pembekuan & Pencairan Lapisan Progresif
Jalankan skrip inspeksi flag gradien bertingkat:
```bash
python content/level-4-advanced-ai/chapter-31/code/freeze_unfreeze_layers.py
```
**Fokus Pengamatan**:
- Analisis tabel parameter yang dihasilkan: amati bagaimana persentase parameter terlatih (*trainable percentage*) berubah dari 4.4% pada Linear Probing hingga 100% pada inisialisasi penuh.
- Pahami teknik navigasi hirarki modul PyTorch melalui `named_parameters()`.

### Langkah 4: Pelatihan Linear Probing
Jalankan eksperimen pelatihan kepala linier murni:
```bash
python content/level-4-advanced-ai/chapter-31/code/linear_probing.py
```
**Fokus Pengamatan**:
- Amati konvergensi loss yang sangat cepat pada kepala klasifikasi baru (hanya 99 parameter).
- Catat efisiensi waktu komputasi saat representasi fitur laten sudah terpisah secara memadai di ruang semantik.

### Langkah 5: Partial Fine-Tuning dengan Differential Learning Rates
Jalankan eksperimen adaptasi lapisan atas dengan laju pembelajaran bertingkat:
```bash
python content/level-4-advanced-ai/chapter-31/code/partial_fine_tuning.py
```
**Fokus Pengamatan**:
- Perhatikan sintaks konfigurasi optimizer parameter groups di mana lapisan atas backbone dilatih dengan LR kecil ($\eta = 10^{-4}$), sedangkan kepala baru dilatih dengan LR cepat ($\eta = 10^{-2}$).
- Buktikan bahwa gradien pada lapisan bawah yang dibekukan bernilai `None`.

### Langkah 6: Mengamati Bahaya Catastrophic Forgetting pada Full Fine-Tuning
Jalankan eksperimen diseksi retensi tugas sumber:
```bash
python content/level-4-advanced-ai/chapter-31/code/full_fine_tuning.py
```
**Fokus Pengamatan**:
- Bandingkan performa sebelum dan sesudah fine-tuning agresif: akurasi tugas sumber runtuh hingga puluhan persen (*catastrophic forgetting*).
- Amati bagaimana pendekatan konservatif dengan laju pembelajaran kecil dan *weight decay* mampu memitigasi tingkat kelupaan representasi.

### Langkah 7: Simulasi Pergeseran Domain (Domain Shift)
Jalankan eksperimen injeksi pergeseran kovariat sensor:
```bash
python content/level-4-advanced-ai/chapter-31/code/domain_shift_demo.py
```
**Fokus Pengamatan**:
- Ukur degradasi performa model saat masukan bergeser dari kondisi normal ke kondisi lingkungan ekstrem.
- Buktikan bahwa adaptasi domain menggunakan sedikit sampel target terlabel mampu memulihkan batas keputusan model secara signifikan.

### Langkah 8: Mengidentifikasi Transfer Negatif (Negative Transfer)
Jalankan skrip pengujian kondisi kegagalan transfer:
```bash
python content/level-4-advanced-ai/chapter-31/code/negative_transfer_demo.py
```
**Fokus Pengamatan**:
- Amati kasus di mana transfer learning menghasilkan akurasi yang lebih rendah dibanding pelatihan dari nol (*Scratch*).
- Pelajari kriteria evaluasi diagnostik untuk memutuskan kapan transfer learning layak ditinggalkan.

### Langkah 9: Pembuktian Matematis LoRA dari Dasar
Jalankan pembuktian aljabar dekomposisi matriks LoRA:
```bash
python content/level-4-advanced-ai/chapter-31/code/lora_from_scratch.py
```
**Fokus Pengamatan**:
- Buktikan secara numerik bahwa inisialisasi $B=0$ menjamin $\Delta W = 0$ saat langkah awal, sehingga model adaptor memiliki luaran yang identik 100% dengan model dasar.
- Verifikasi bahwa rank matriks pembaruan $B \cdot A$ terbukti $\le r$.

### Langkah 10: Analisis Efisiensi Parameter & Memori LoRA
Jalankan skrip kalkulasi penghematan memori dan parameter:
```bash
python content/level-4-advanced-ai/chapter-31/code/lora_parameter_count.py
```
**Fokus Pengamatan**:
- Pelajari tabel penghematan parameter lintas dimensi ($d=256$ hingga $d=4096$) dan rank ($r=1$ hingga $r=64$).
- Pahami mengapa penghematan memori VRAM/RAM mencapai $>95\%$ akibat ketiadaan status optimizer Adam pada bobot beku.

### Langkah 11: Implementasi Modul Kustom LoRALinear di PyTorch
Jalankan pengujian kelas kustom `LoRALinear`:
```bash
python content/level-4-advanced-ai/chapter-31/code/lora_pytorch_demo.py
```
**Fokus Pengamatan**:
- Verifikasi bahwa saat propagasi balik, gradien hanya mengalir ke `lora_A` dan `lora_B`, sementara bobot dasar `base_layer.weight` tetap bersih dari gradien.
- Uji fitur penggabungan bobot (*weight merging*) untuk inferensi produksi dengan latensi nol.

### Langkah 12: Komparasi Empiris 5 Paradigma Adaptasi
Jalankan perbandingan menyeluruh pada data uji terisolasi:
```bash
python content/level-4-advanced-ai/chapter-31/code/fine_tuning_comparison.py
```
**Fokus Pengamatan**:
- Evaluasi tabel ringkasan perbandingan: parameter latih, persentase trainable, waktu latih, dan akurasi data uji akhir antara Scratch, Linear Probing, Partial FT, Full FT, dan LoRA.

---

## 3. Latihan Bertingkat (3-Tier Exercises)

### Latihan Tingkat 1: Pemahaman Dasar (Konseptual & Perhitungan)
1. **Soal 1.1**: Sebuah lapisan linier proyeksi pada model Transformer memiliki dimensi masukan $d_{\text{in}} = 1024$ dan keluaran $d_{\text{out}} = 1024$. Jika lapisan ini diadaptasi menggunakan LoRA dengan rank $r=8$, hitunglah:
   a. Jumlah parameter pada matriks bobot dasar $W_0$.
   b. Jumlah parameter pada matriks LoRA $A$ dan matriks LoRA $B$.
   c. Persentase pengurangan parameter terlatih yang dihasilkan LoRA.
2. **Soal 1.2**: Jelaskan mengapa membekukan parameter via `param.requires_grad = False` pada PyTorch tidak otomatis menonaktifkan sifat acak dari lapisan `nn.Dropout`. Apa pemanggilan metode yang wajib dilakukan untuk menjamin determinisme evaluasi?

*Kunci Jawaban*:
- **1.1**:
  a. Parameter $W_0 = 1024 \times 1024 = 1.048.576$ parameter.
  b. Matriks $A \in \mathbb{R}^{8 \times 1024} = 8.192$ elemen. Matriks $B \in \mathbb{R}^{1024 \times 8} = 8.192$ elemen. Total LoRA $= 8.192 + 8.192 = 16.384$ parameter.
  c. Persentase pengurangan $= \left(1 - \frac{16.384}{1.048.576}\right) \times 100\% = 98.44\%$.
- **1.2**: Flag `requires_grad = False` hanya menginstruksikan autograd engine untuk tidak menghitung turunan parsial terhadap tensor bobot tersebut saat *backward pass*. Operasi forward pass pada lapisan `nn.Dropout` dipandu oleh atribut internal `self.training`. Jika model tetap berada dalam mode `train()`, dropout akan terus menjatuhkan aktivasi acak. Solusinya adalah memanggil `model.eval()`.

### Latihan Tingkat 2: Implementasi Menengah
**Soal 2.1**: Tuliskan cuplikan kode PyTorch untuk membuat optimizer AdamW dengan dua parameter group berbeda:
- Lapisan `model.backbone.layer4` dilatih dengan learning rate $5 \times 10^{-5}$ dan weight decay $10^{-4}$.
- Lapisan `model.classifier` dilatih dengan learning rate $1 \times 10^{-3}$ dan weight decay $10^{-3}$.
Semua lapisan lain di backbone harus dibekukan (`requires_grad = False`).

*Kunci Jawaban*:
```python
# 1. Bekukan seluruh parameter model
for param in model.parameters():
    param.requires_grad = False

# 2. Buka lapisan spesifik
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

# 3. Definisikan parameter groups
optimizer = torch.optim.AdamW([
    {'params': model.backbone.layer4.parameters(), 'lr': 5e-5, 'weight_decay': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3}
])
```

### Latihan Tingkat 3: Analisis Lanjut & Pemecahan Masalah
**Soal 3.1**: Anda bertugas mengadaptasi model klasifikasi citra medis untuk mendeteksi jenis lesi kulit langka (hanya tersedia 60 sampel citra berlabel). Tim Anda mengusulkan Full Fine-Tuning pada arsitektur ResNet-50.
a. Evaluasi risiko rekayasa apa saja yang dihadapi pendekatan ini.
b. Rancang strategi adaptasi bertahap alternatif yang meminimalkan risiko overfitting dan keterbatasan memori komputasi.

*Kunci Jawaban*:
a. **Risiko Rekayasa**:
   - *Severe Overfitting*: Melatih 23+ juta parameter ResNet-50 pada 60 sampel akan membuat model menghafal derau data latih murni, menghasilkan generalisasi buruk pada data uji.
   - *Catastrophic Forgetting*: Bobot filter konvolusi umum yang telah dipelajari akan rusak seketika.
b. **Strategi Alternatif Rekayasa**:
   - **Tahap 1**: Lakukan *Data Augmentation* ketat (rotasi acak, flip, pergeseran warna) untuk memperkaya variasi sampel.
   - **Tahap 2**: Terapkan *Linear Probing* sebagai patokan minimum: bekukan seluruh backbone ResNet-50 (`eval()` mode), latih hanya lapisan kepala klasifikasi baru dengan regularisasi L2 yang kuat.
   - **Tahap 3**: Jika batas pemisahan kelas membutuhkan penyesuaian fitur tingkat tinggi, terapkan *LoRA* ($r=4, \alpha=8$) hanya pada matriks proyeksi 1x1 di blok residual teratas, atau *Partial Fine-Tuning* dengan laju pembelajaran sangat kecil ($\le 10^{-5}$) dan early stopping berbasis validasi loss.

---

## 4. Mini Project: Efficient Model Adaptation Using Transfer Learning & LoRA

### 4.1 Deskripsi Masalah & Sasaran
Mahasiswa diminta merancang eksperimen terukur untuk mengadaptasi sebuah model representasi terhadap tugas target baru dengan batasan komputasi CPU dan jumlah data latih terbatas:
1. Memilih domain masalah target (*misalnya klasifikasi teks keluhan pelanggan, telemetri sensor getaran mesin, atau sinyal IoT*).
2. Membangun representasi tugas sumber (*pre-trained backbone*) atau menggunakan model rujukan yang telah dipelajari.
3. Mengimplementasikan minimal tiga varian adaptasi:
   - Baseline: Pelatihan dari awal (*Training from Scratch*).
   - Varian 1: *Feature Extraction* (*Linear Probing* dengan backbone beku).
   - Varian 2: *Partial Fine-Tuning* ATAU *LoRA Adaptation*.
4. Mengukur dan membandingkan secara kuantitatif:
   - Jumlah parameter terlatih (*Trainable Parameter Count* dan persentasenya).
   - Waktu pelatihan (*Training Latency* per epoch pada CPU).
   - Metrik akurasi dan Macro F1 pada Test Set terisolasi.
5. Menyusun laporan analisis komparatif, evaluasi kegagalan (*error analysis*), dan pertimbangan etika kecerdasan buatan (*Responsible AI*).

---

## 5. Rubrik Penilaian Proyek Mini (Skala 100 Poin)

| Dimensi Evaluasi | Bobot | Kriteria Sempurna (Full Credit) | Kriteria Cukup (Partial Credit) | Kriteria Kurang (Insufficient) |
|---|:---:|---|---|---|
| **1. Pemahaman Masalah & Partisi Data** | 15 Poin | Perumusan masalah jelas; dataset terdokumentasi; pembagian 3-way (Train/Val/Test) bebas kebocoran data (*zero data leakage*). | Masalah terdefinisi namun partisi data uji kurang terisolasi atau terjadi kebocoran skala ringan. | Tidak ada isolasi data uji; dataset ambigu; kebocoran data parah. |
| **2. Implementasi Transfer Learning & Freezing** | 25 Poin | Backbone dibekukan secara benar (`requires_grad=False`); mode `eval()` diposisikan tepat; Linear Probing berhasil dieksekusi. | Backbone dibekukan namun mode runtime `train()` vs `eval()` diabaikan. | Pembekuan parameter gagal; kode runtime error. |
| **3. Implementasi Fine-Tuning / LoRA** | 25 Poin | Menerapkan Partial FT dengan differential LRs atau LoRA dengan dekomposisi rank rendah; pembuktian gradien dan paritas inisialisasi tuntas. | Menerapkan adaptasi namun hiperparameter laju pembelajaran tidak terkontrol atau pembuktian gradien tidak ada. | Implementasi salah aljabar; parameter tidak terupdate; tidak ada LoRA/FT. |
| **4. Evaluasi & Analisis Sumber Daya** | 25 Poin | Tabel komparasi lengkap (param count, waktu latih, akurasi, F1); analisis trade-off performa vs biaya memori komputasi mendalam. | Komparasi hanya melaporkan akurasi tanpa analisis waktu komputasi atau parameter latih. | Klaim performa tanpa tabel evaluasi; metrik dilaporkan pada data latih. |
| **5. Dokumentasi & Referensi Akademik** | 10 Poin | Kode bersih, modular, terdokumentasi rapi; Model Card ringkas tersedia; menyertakan $\ge 2$ referensi akademik terverifikasi. | Kode dapat dijalankan namun dokumentasi minim; referensi kurang lengkap. | Tidak ada dokumentasi; referensi fiktif atau tidak relevan. |
| **Total Skor Maksimal** | **100 Poin** | **Standar Rekayasa Industri & Akademik Teruji** | — | — |

