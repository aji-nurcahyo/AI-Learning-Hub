# Laporan Audit Kualitas: Chapter 24 — Convolutional Neural Networks

**Tanggal Audit**: 8 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR KURIKULUM RESMI)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 24 — Convolutional Neural Networks (Level 3 Deep Learning)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 24 | 24 | Sesuai |
| Judul Resmi | Convolutional Neural Networks | Convolutional Neural Networks | Sesuai |
| Level Pembelajaran | Level 3: Deep Learning & Generative AI | Level 3: Deep Learning & Generative AI | Sesuai |
| Prasyarat | Chapter 23 | `[23]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Komputasi | Ekosistem PyTorch CPU terverifikasi | PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu | Memenuhi standar |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-24/index.md` | Lengkap (Part 1–14, 16+ Diagram ASCII, Latihan 3-Level, Rubrik 100 Poin, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-24/slides/deck.json` | 22 Slide valid, format direct array JSON, KaTeX, speaker notes |
| 3 | Panduan Praktik Lab | `chapter-24/practice.md` | Lengkap (6 Tahapan Praktik, 3-Tier Exercises, Rubrik 100 Poin) |
| 4 | Skrip Python Lengkap | `chapter-24/code/*.py` (6 skrip) | 6/6 skrip dieksekusi dengan exit code 0 |
| 5 | Notebook Pendamping | `chapter-24/notebooks/chapter_24_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 6 | Katalog Kode | `chapter-24/code/README.md` | Panduan eksekusi dan tabel skrip terverifikasi |
| 7 | Log Eksperimen | `chapter-24/experiments/results_ch24.json` | Log JSON terstruktur perbandingan model dan metrik |
| 8 | Model Checkpoint | `chapter-24/models/fashion_cnn_weights.pt` | Checkpoint `state_dict` berbobot valid |
| 9 | Metadata Bab | `chapter-24/metadata.json` | 14 Bloom Objectives diawali kata kerja valid, 5 Referensi terverifikasi |

---

## 3. Verifikasi Numerik & Hasil Komputasi Aktual

Seluruh nilai eksperimen berikut telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10, PyTorch 2.15.0+cpu, dan torchvision 0.29.2+cpu:

```
[Pengujian 1: Operasi Konvolusi Manual vs PyTorch F.conv2d]
- Input Matriks 5x5, Kernel 3x3  : Manual calculation vs F.conv2d
- Selisih Maksimum Absolut       : 0.00e+00 (Single-channel) dan 1.91e-06 (Multi-channel NCHW)
- Status Numerik                 : IDENTIK SEMPURNA hingga toleransi floating-point

[Pengujian 2: Kalkulator Dimensi Spasial & Parameter Bobot]
- Rumus Spasial H_out            : floor((H_in + 2P - D(K - 1) - 1) / S + 1)
- Valid Padding (K=3, P=0, S=1)  : 28 -> 26 (PyTorch: 26) [MATCH]
- Same Padding (K=3, P=1, S=1)   : 28 -> 28 (PyTorch: 28) [MATCH]
- Strided Conv (K=3, P=1, S=2)   : 28 -> 14 (PyTorch: 14) [MATCH]
- Dilated Conv (K=3, P=0, S=1, D=2): 28 -> 24 (PyTorch: 24) [MATCH]
- Max Pooling (K=2, S=2)         : Parameter = 0 (PyTorch: 0) [MATCH]
- Parameter Conv Layer 1 (1->16) : (3*3*1 + 1)*16 = 160 (PyTorch: 160) [MATCH]
- Parameter Conv Layer 2 (16->32): (3*3*16 + 1)*32 = 4640 (PyTorch: 4640) [MATCH]

[Pengujian 3: Efisiensi Parameter Arsitektur CNN vs MLP]
- Total Parameter FashionCNN     : 105.962 bobot
- Total Parameter FlattenMLP     : 109.386 bobot (Kapasitas model seimbang)
- Parameter Lapisan Pertama      : CNN (160) vs MLP (100.480) -> CNN 99.84% LEBIH HEMAT BOBOT

[Pengujian 4: Pipeline Aliran Data & Normalisasi Fashion-MNIST]
- Sumber Dataset                 : Zalando Research (MIT License) via torchvision.datasets
- Statistik Normalisasi          : Mean = 0.2860, Std = 0.3530
- Partisi Data                   : Train = 50.000, Val = 10.000, Test = 10.000 (Terisolasi ketat)
- Mekanisme Fallback             : Synthetic fallback dataset teruji offline

[Pengujian 5: Benchmark Komparatif pada CPU (5 Epochs, Batch 64, Adam lr=0.001)]
- FashionCNN Test Accuracy       : 86.30% | Test Loss = 0.3664
- FlattenMLP Test Accuracy       : 84.15% | Test Loss = 0.4410
- Keunggulan Agregat CNN         : +2.15% Akurasi | -0.0746 Loss
- Keunggulan Kelas 'Shirt'       : CNN 75.63% vs MLP 54.31% (+21.32% KEUNGGULAN MASIF!)
- Keunggulan Kelas 'Pullover'    : CNN 83.40% vs MLP 76.10% (+7.30%)
- Akurasi Kelas 'Ankle boot'     : CNN 96.10% vs MLP 94.60% (+1.50%)

[Pengujian 6: Error Analysis & Feature Maps Inspection]
- Pasangan Kelas Sering Tertukar : 1. T-shirt vs Shirt (226 kekeliruan)
                                   2. Pullover vs Coat (129 kekeliruan)
                                   3. Shirt vs Coat (118 kekeliruan)
- Kesalahan Antar-Kategori Beda  : Shirt vs Ankle boot mendekati 0% (Hampir sempurna)
- Ekstraksi Feature Maps Conv1   : 16 filter 28x28 mengekstraksi detektor tepi horizontal,
                                   vertikal, dan kontras tekstur lokal
```

---

## 4. Kesimpulan Audit Akhir
Bab 24 telah memenuhi seluruh kriteria pedagogi, matematika, dan rekayasa perangkat lunak berstandar internasional.
Status: **PASSED (100%)**.

