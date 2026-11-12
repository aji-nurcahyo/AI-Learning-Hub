# Katalog Skrip Praktik: Chapter 24 — Convolutional Neural Networks

Direktori ini berisi implementasi kode Python modular dan runnable untuk **Bab 24: Convolutional Neural Networks**. Seluruh skrip menggunakan framework PyTorch pada jalur komputasi CPU dan dataset resmi **Fashion-MNIST** (Zalando Research, MIT License).

---

## 1. Daftar Skrip dan Deskripsi Komponen

| No | Nama Berkas | Deskripsi Fungsional |
|---|---|---|
| 1 | [`manual_convolution.py`](manual_convolution.py) | Operasi 2D Cross-Correlation manual dari awal (*scratch* loop), perbandingan dengan `torch.nn.functional.conv2d`, serta pembuktian kesetaraan numerik 100% pada single-channel dan multi-channel tensor. |
| 2 | [`conv_shape_calculator.py`](conv_shape_calculator.py) | Kalkulator dimensi spasial output $H_{out} = \lfloor \frac{H_{in} + 2P - D(K - 1) - 1}{S} + 1 \rfloor$ dan formula jumlah parameter, diverifikasi terhadap `nn.Conv2d` (Valid, Same, Strided, Dilated) dan `nn.MaxPool2d`. |
| 3 | [`cnn_architecture.py`](cnn_architecture.py) | Arsitektur model `FashionCNN` modular (Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, Linear) dibandingkan dengan model acuan `FlattenMLPBaseline`, serta pembuktian efisiensi parameter sharing. |
| 4 | [`data_loader_fashion_mnist.py`](data_loader_fashion_mnist.py) | Pipa pemuatan dataset Fashion-MNIST (Zalando Research, MIT), isolasi augmentasi data pada train set saja, normalisasi mean=0.2860 std=0.3530, dan fallback otomatis ke `SyntheticFashionDataset` bila offline. |
| 5 | [`train_cnn_vs_baseline.py`](train_cnn_vs_baseline.py) | Pelatihan komparatif terpadu CNN vs FlattenMLP pada protokol evaluasi identik, opsi `--smoke-test`, pelacakan waktu nyata CPU, evaluasi akurasi per-kelas, dan penyimpanan hasil ke `experiments/results_ch24.json`. |
| 6 | [`error_analysis_feature_maps.py`](error_analysis_feature_maps.py) | Diseksi kesalahan prediksi (top confused pairs: T-shirt vs Shirt, Pullover vs Coat), ekstraksi feature maps lapisan konvolusi pertama, render visualisasi ASCII, dan diskusi batas interpretasi manusia. |

---

## 2. Cara Menjalankan Skrip

Pastikan virtual environment telah aktif:
```bash
# Windows PowerShell
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-24/code/manual_convolution.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-24/code/conv_shape_calculator.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-24/code/cnn_architecture.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-24/code/data_loader_fashion_mnist.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-24/code/train_cnn_vs_baseline.py
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-24/code/error_analysis_feature_maps.py
```

Untuk verifikasi smoke test cepat:
```bash
.venv\Scripts\python.exe content/level-3-deep-learning/chapter-24/code/train_cnn_vs_baseline.py --smoke-test
```

---

## 3. Hasil Pengujian Aktual (Terverifikasi)
- **Dataset**: Fashion-MNIST Asli (Zalando Research, lisensi MIT, 10 kelas produk).
- **Verifikasi Konvolusi**: Selisih maksimum manual vs `F.conv2d` = $0.00 \times 10^0$ (Single-channel) dan $1.91 \times 10^{-6}$ (Multi-channel NCHW).
- **Perbandingan Model (Test Set N=2000 terisolasi)**:
  - `FashionCNN` (105,962 parameter): Akurasi **86.30%**, Test Loss **0.3755** (Waktu latih CPU: 6.6s).
  - `FlattenMLPBaseline` (109,386 parameter): Akurasi **84.15%**, Test Loss **0.4319** (Waktu latih CPU: 4.8s).
  - Keunggulan CNN: **+2.15%** secara agregat, dan pada kelas pakaian rumit seperti Shirt: **75.63% vs 54.31% (+21.32%)**.
- **Analisis Kesalahan**: Pasangan paling sering tertukar adalah kategori pakaian berstruktur mirip (T-shirt/top vs Shirt: 226 salah, Pullover vs Coat: 129 salah).

