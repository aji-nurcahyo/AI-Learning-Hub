# Panduan Praktik Laboratorium: Deep Learning with PyTorch

Selamat datang di modul panduan praktik laboratorium **Deep Learning with PyTorch** (Bab 23). Pada laboratorium ini, Anda akan mempraktikkan alur kerja pengembangan jaringan saraf tiruan modern menggunakan pustaka **PyTorch** pada jalur komputasi CPU: mulai dari manipulasi tensor, kalkulus graf diferensiasi otomatis (*autograd*), perancangan arsitektur modular dengan `torch.nn.Module`, pengelolaan aliran data efisien dengan `Dataset` dan `DataLoader`, siklus pelatihan dan validasi terstandar industri, hingga serialisasi parameter aman melalui `state_dict`.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- Python: Versi 3.10+ (Direkomendasikan Python 3.12).
- Pustaka Inti: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `numpy>=2.0.0`, `matplotlib>=3.8.0`.
- GPU/CUDA: Bersifat opsional. Seluruh skrip dan eksperimen bab ini dirancang agar dapat dieksekusi secara instan pada CPU komputer standar tanpa memerlukan akselerator khusus.

### Struktur Repositori Praktik Bab 23:
```
content/level-3-deep-learning/chapter-23/
├── code/
│   ├── README.md
│   ├── tensor_fundamentals.py
│   ├── autograd_mechanics.py
│   ├── dataset_dataloader.py
│   ├── neural_network_module.py
│   ├── training_validation_pipeline.py
│   └── checkpoint_inference.py
├── experiments/
│   └── results_ch23.json
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_23_companion.ipynb
├── slides/
│   ├── build_slides.py
│   └── deck.json
├── index.md
├── practice.md
├── metadata.json
└── audit.md
```

---

## 2. Panduan Langkah-demi-Langkah Praktik Laboratorium

### Langkah 1: Eksplorasi Fondasi Tensor & NumPy Bridge
Jalankan skrip operasi tensor dasar untuk memahami dimensi, tipe data, dan alokasi memori bersama:
```bash
python content/level-3-deep-learning/chapter-23/code/tensor_fundamentals.py
```
- **Fokus Pengamatan**:
  - Perhatikan bagaimana fungsi `view(3, 4)` mengubah tata letak data tanpa menyalin memori fisik.
  - Buktikan jembatan zero-copy dengan memodifikasi elemen pertama tensor PyTorch dan perhatikan bahwa elemen array NumPy asal ikut berubah secara instan.
  - Amati mekanisme *broadcasting* saat menjumlahkan tensor shape $(3, 1)$ dan $(1, 4)$ menjadi $(3, 4)$.

### Langkah 2: Mekanisme Autograd & Bahaya Akumulasi Gradien
Uji cara kerja mesin graf komputasi dinamis dan investigasi jebakan akumulasi gradien bawaan:
```bash
python content/level-3-deep-learning/chapter-23/code/autograd_mechanics.py
```
- **Fokus Pengamatan**:
  - Bandingkan nilai gradien parsial yang dihitung otomatis melalui `L.backward()` dengan turunan analitis diferensiasi rantai. Keduanya harus cocok hingga presisi $10^{-5}$.
  - Amati apa yang terjadi ketika `backward()` dipanggil untuk kedua kalinya tanpa memanggil `optimizer.zero_grad()`: gradien akan bertambah dua kali lipat secara akumulatif.
  - Pelajari penghematan memori graf menggunakan konteks `with torch.no_grad():` dan pemutusan rantai melalui `.detach()`.

### Langkah 3: Abstraksi Custom Dataset & DataLoader 3-Way
Bangun pipa aliran data modular dengan partisi data yang terisolasi secara ketat:
```bash
python content/level-3-deep-learning/chapter-23/code/dataset_dataloader.py
```
- **Fokus Pengamatan**:
  - Amati dataset sintetis 1000 sampel yang dibagi menjadi Train (700), Validation (150), dan Test (150). Verifikasi bahwa irisan indeks antar-partisi bernilai kosong (0% tumpang tindih).
  - Pastikan bahwa normalisasi fitur (rata-rata dan standar deviasi) dihitung **HANYA** dari data Train guna mencegah *data leakage*.
  - Periksa parameter `DataLoader`: `shuffle=True` hanya diaktifkan pada data latih, sedangkan data validasi dan uji mempertahankan urutan deterministik (`shuffle=False`).

### Langkah 4: Desain Arsitektur Modular dengan nn.Module
Definisikan arsitektur jaringan saraf lapis banyak (*Multilayer Perceptron*) dan model acuan linier:
```bash
python content/level-3-deep-learning/chapter-23/code/neural_network_module.py
```
- **Fokus Pengamatan**:
  - Telusuri penghitungan parameter terbobot: pastikan model `TabularMLP` memiliki total tepat 266 parameter *trainable*.
  - Pahami alasan mendasar mengapa model mengeluarkan **logits mentah** tanpa fungsi `Softmax`: PyTorch `nn.CrossEntropyLoss` secara internal telah mengintegrasikan fungsi log-softmax yang distabilkan dengan trik *subtract-max*.

### Langkah 5: Eksekusi Pipeline Pelatihan & Evaluasi Lengkap
Latih model `TabularMLP` dan `SimpleBaseline` pada kondisi eksperimen yang identik:
```bash
python content/level-3-deep-learning/chapter-23/code/training_validation_pipeline.py
```
- **Fokus Pengamatan**:
  - Amati peran `model.train()` sebelum loop pelatihan dan `model.eval()` sebelum evaluasi validasi.
  - Perhatikan konvergensi nilai rugi (*loss*) dan akurasi per epoch.
  - Evaluasi performa akhir pada test set terisolasi: bandingkan akurasi `TabularMLP` (sekitar 88.67%) terhadap `SimpleBaseline` (sekitar 83.33%). Buktikan bahwa kapasitas non-linier MLP memberikan keunggulan terukur (+5.33%).
  - Periksa berkas hasil eksperimen terstruktur yang tersimpan di `experiments/results_ch23.json`.

### Langkah 6: Checkpointing Aman & Inferensi Produksi
Simpan parameter bobot model ke dalam berkas checkpoint dan uji konsistensi inferensi saat dimuat kembali:
```bash
python content/level-3-deep-learning/chapter-23/code/checkpoint_inference.py
```
- **Fokus Pengamatan**:
  - Simpan hanya `model.state_dict()`, bukan seluruh objek model.
  - Muat kembali bobot ke model baru yang masih kosong menggunakan `weights_only=True` guna menghindari kerentanan eksekusi kode arbitrer.
  - Buktikan bahwa selisih absolut logits sebelum disimpan dan sesudah dimuat bernilai $0.00 \times 10^0$ (100% identik).

---

## 3. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Analisis Bentuk Tensor**: Diberikan tensor masukan gambar berdimensi $(B, C, H, W) = (16, 3, 32, 32)$. Tuliskan perintah PyTorch untuk mengubah bentuk tensor tersebut menjadi representasi 2D $(16, 3072)$ sebelum diumpankan ke lapisan `nn.Linear(3072, 128)`!
   - *Jawaban Singkat*: Gunakan `x = x.view(x.size(0), -1)` atau `x = torch.flatten(x, start_dim=1)`. Operasi ini meratakan dimensi channel, tinggi, dan lebar ($3 \times 32 \times 32 = 3072$) sambil mempertahankan ukuran batch 16.
2. **Tipe Data Target Klasifikasi**: Mengapa saat melatih model klasifikasi multi-kelas dengan `nn.CrossEntropyLoss`, tensor label target $y$ harus bertipe `torch.long` (int64) dan berdimensi 1D $(N)$, bukan one-hot encoded berdimensi 2D $(N, C)$ bertipe `torch.float32`?
   - *Jawaban Singkat*: Fungsi `nn.CrossEntropyLoss` di PyTorch dirancang efisien dengan menerima indeks kelas skalar $0 \le y_i < C$ bertipe integer 64-bit (`torch.long`). Hal ini menghemat alokasi memori graf dan mempercepat pengindeksan tensor gradien di level kernel C++ tanpa memerlukan matriks jarang *one-hot*.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Bug Pelatihan**: Seorang mahasiswa mendapati bahwa modelnya menghasilkan nilai akurasi uji yang sangat tinggi (99%), namun ketika dideploy ke data baru performanya anjlok menjadi 50%. Setelah memeriksa kodenya, ditemukan potongan baris berikut:
   ```python
   scaler = StandardScaler()
   X_all = scaler.fit_transform(X_raw)  # Sebelum train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X_all, y_raw)
   ```
   Jelaskan secara tepat bentuk kesalahan metodologis pada kode di atas dan berikan koreksi perbaikannya!
   - *Jawaban Singkat*: Terjadi **Data Leakage** (Kebocoran Data). Melakukan `fit_transform` pada seluruh data sebelum partisi menyebabkan informasi statistik (rata-rata dan variansi) dari data uji bocor ke data latih. Kode yang benar adalah:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw)
     scaler = StandardScaler()
     X_train = scaler.fit_transform(X_train)  # Fit HANYA pada data latih
     X_test = scaler.transform(X_test)        # Transform data uji dengan statistik data latih
     ```
2. **Perbedaan `model.eval()` dan `torch.no_grad()`**: Jelaskan mengapa memanggil `model.eval()` saja tanpa membungkus kode dalam `with torch.no_grad():` tetap akan memakan banyak memori RAM/GPU saat menjalankan inferensi batch besar!
   - *Jawaban Singkat*: `model.eval()` hanya mengubah *mode flag* internal lapisan tertentu (seperti mematikan pengacakan pada Dropout dan membekukan running statistics pada BatchNorm), tetapi **TIDAK** menonaktifkan mesin pelacak graf komputasi *autograd*. Akibatnya, PyTorch tetap mengalokasikan memori untuk menyimpan seluruh operasi forward pass. Agar alokasi graf dihentikan total, kode evaluasi wajib dibungkus di dalam blok `with torch.no_grad():`.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Kalkulus Gradien Akumulatif**: Pada model besar di mana ukuran batch ideal adalah 128 tetapi VRAM GPU hanya mampu menampung batch berukuran 32, tuliskan cuplikan kode PyTorch yang mengimplementasikan teknik *Gradient Accumulation* selama 4 sub-batch sehingga pembaruan bobot optimizer ekuivalen dengan batch size 128!
   - *Jawaban Singkat*:
     ```python
     accumulation_steps = 4
     optimizer.zero_grad()
     for i, (inputs, targets) in enumerate(train_loader):
         outputs = model(inputs)
         loss = criterion(outputs, targets) / accumulation_steps  # Skalakan loss
         loss.backward()
         if (i + 1) % accumulation_steps == 0:
             optimizer.step()
             optimizer.zero_grad()
     ```

---

## 4. Mini Project Chapter 23

### Judul Tugas Proyek:
**"Building an End-to-End Classification Pipeline Using PyTorch"**

### Deskripsi Tugas:
Rancang dan bangun pipeline klasifikasi terpadu dari awal menggunakan **PyTorch**. Anda diminta untuk:
1. Membangkitkan dataset klasifikasi sintetis dengan seed tetap, membaginya ke dalam partisi 3-way (Train, Validation, Test) secara terisolasi tanpa tumpang tindih, dan menormalisasi fitur bebas data leakage.
2. Membangun kelas arsitektur jaringan saraf lapis banyak `nn.Module` yang memiliki minimal 2 lapisan tersembunyi non-linier.
3. Mengimplementasikan siklus pelatihan (*training loop*) dan validasi (*validation loop*) lengkap dengan pelacakan metrik loss dan akurasi per epoch.
4. Membandingkan performa model jaringan saraf terhadap model acuan linier sederhana (*Simple Baseline*) di bawah protokol evaluasi yang sama.
5. Menyimpan bobot terbaik menggunakan `state_dict`, memuatnya kembali dengan `weights_only=True`, dan memvalidasi konsistensi prediksi inferensi.
6. Menyusun analisis kesalahan (*error analysis*) singkat melalui matriks konfusi dan mendokumentasikan keterbatasan model.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Data & Protokol Split** | 20 Poin | Pembangkitan data terdefinisi (`seed=42`), pemisahan partisi 3-way (Train, Val, Test) terbukti tidak tumpang tindih, dan normalisasi fitur terisolasi hanya pada statistik data latih (zero data leakage). |
| **2. Model & Tensor Correctness** | 20 Poin | Subclassing `nn.Module` tepat, penentuan dimensi lapisan dan jumlah parameter akurat, serta luaran model berupa logits mentah tanpa Softmax prematur yang cocok untuk `nn.CrossEntropyLoss`. |
| **3. Training & Validation Loop** | 25 Poin | Implementasi loop pelatihan baku (`optimizer.zero_grad()`, `backward()`, `step()`), penggunaan tepat `model.train()` dan `model.eval()`, serta pembungkusan evaluasi di dalam `with torch.no_grad()`. |
| **4. Evaluation & Reproducibility** | 20 Poin | Evaluasi akhir pada test set terisolasi, perbandingan performa objektif terhadap baseline linier, matriks konfusi 2x2, serta penyimpanan dan pemuatan aman checkpoint `state_dict` (`weights_only=True`). |
| **5. Penjelasan Hasil & Keterbatasan** | 15 Poin | Analisis keunggulan representasi non-linier, dokumentasi batas reproduksibilitas floating-point antar-perangkat keras, dan pembahasan keterbatasan data sintetis secara jujur. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 6 skrip Python di direktori `code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Gradien autograd terbukti cocok dengan perhitungan diferensiasi analitis.
- [ ] Partisi data 3-way membuktikan isolasi total tanpa kebocoran data (*zero leakage*).
- [ ] Luaran model mengeluarkan logits mentah untuk `nn.CrossEntropyLoss`.
- [ ] Model dievaluasi menggunakan `model.eval()` dan `torch.no_grad()`.
- [ ] Checkpoint model disimpan dan dimuat ulang menggunakan `weights_only=True` dengan konsistensi 100%.
- [ ] Berkas hasil eksperimen `experiments/results_ch23.json` terisi dengan metrik valid.
- [ ] Notebook pendamping `chapter_23_companion.ipynb` tervalidasi bersih dalam format v4.

