# Laporan Audit Kualitas: Chapter 21 — Neural Network Fundamentals

**Tanggal Audit**: 5 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR KURIKULUM RESMI)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 21 — Neural Network Fundamentals (Level 3 Kickoff)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 21 | 21 | Sesuai |
| Judul Resmi | Neural Network Fundamentals | Neural Network Fundamentals | Sesuai |
| Level Pembelajaran | Level 3: Deep Learning & Generative AI | Level 3: Deep Learning & Generative AI | Sesuai |
| Prasyarat | Chapter 20 | `[20]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Komputasi | Fondasi komputasi NumPy murni | NumPy (Bebas GPU/PyTorch wajib) | Memenuhi standar |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-21/index.md` | Lengkap (Part A–G, 16 Diagram ASCII, Worked Example 2-3-1, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-21/slides/deck.json` | 30 Slide valid, format direct array JSON, KaTeX, speaker notes |
| 3 | Panduan Praktik Lab | `chapter-21/practice.md` | Lengkap (10 Tahapan Praktik, 3-Tier Exercises, Rubrik 100 Poin) |
| 4 | Skrip Python Lengkap | `chapter-21/code/*.py` (10 skrip) | 10/10 skrip dieksekusi dengan exit code 0 |
| 5 | Notebook Pendamping | `chapter-21/notebooks/chapter_21_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 6 | Katalog Kode | `chapter-21/code/README.md` | Panduan eksekusi dan tabel skrip terverifikasi |
| 7 | Metadata Bab | `chapter-21/metadata.json` | 13 Bloom Objectives, 5 Referensi terverifikasi |

---

## 3. Verifikasi Numerik & Hasil Komputasi

Seluruh nilai eksperimen berikut telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10:

```
[Pengujian 1: Model Neuron Matematis]
- Persamaan Weighted Sum        : z = 2.5*x1 + 1.8*x2 + 1.2*x3 - 3.0
- Kasus Profil Resiko Tinggi    : z = -3.00 -> Aktivasi Step a = 0 (Ditolak)
- Kasus Profil Skor Baik & Gaji : z = +1.30 -> Aktivasi Step a = 1 (Disetujui)

[Pengujian 2: Gerbang Logika Linier Perceptron]
- Gerbang AND (w=[1,1], b=-1.5) : (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1 (100% Benar)
- Gerbang OR  (w=[1,1], b=-0.5) : (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1 (100% Benar)
- Gerbang NAND(w=[-1,-1],b=1.5) : (0,0)->1, (0,1)->1, (1,0)->1, (1,1)->0 (100% Benar)

[Pengujian 3: Masalah XOR & Solusi 2-Layer]
- Bukti Kontradiksi Aljabar     : w1 + w2 + b > 0 vs w1 + w2 + b < 0 (Terbukti Kontradiksi)
- Solusi 2-Layer MLP            : Hidden OR & NAND -> Output AND -> XOR 100% Benar

[Pengujian 4: Fungsi Aktivasi & Dead ReLU]
- Sigmoid Derivative Max di z=0 : f'(0) = 0.2500 (Menyusut ke 0 saat |z| > 4)
- Tanh Derivative Max di z=0    : f'(0) = 1.0000 (Zero-centered, luaran [-1, 1])
- Dead ReLU Verification        : Untuk input [-5.0, -3.2, -1.8], gradien = [0, 0, 0] permanen
- Leaky ReLU Verification       : Mempertahankan gradien alpha = 0.01 saat z < 0

[Pengujian 5: Stabilitas Numerik Softmax]
- Naive Softmax [1000,1001,1002]: Overflow exp(1000) = inf -> Gagal (NaN/Warning)
- Stable Softmax (Subtract Max) : Probs = [0.0900, 0.2447, 0.6652], sum = 1.0000 (Sukses)

[Pengujian 6: Worked Numerical Example (2 -> 3 -> 1)]
- Masukan x = [1.5, 2.0]
- Z^[1] Hidden Sum              : [1.6500, 0.8000, 0.4000]
- A^[1] ReLU Activations        : [1.6500, 0.8000, 0.4000]
- Z^[2] Output Sum              : 0.3700 (Kalkulasi manual: 0.99 - 0.64 + 0.12 - 0.1 = 0.370)
- A^[2] Sigmoid Output          : 0.5915 (Kalkulasi manual: 1 / (1 + exp(-0.37)) = 0.59146)
- Kesesuaian Manual vs Matrix   : 100% Identik

[Pengujian 7: Evaluasi Fungsi Rugi]
- Mean Squared Error (MSE)      : Prediksi dekat = 0.1350 vs Prediksi jauh = 49.3000
- Binary Cross-Entropy (BCE)    : Prediksi yakin & benar = 0.0790 vs Salah percaya diri = 2.6469
- Epsilon Clipping Protection   : Menghilangkan risiko -log(0) = inf
```

---

## 4. Kesimpulan Audit Akhir
Bab 21 telah memenuhi seluruh kriteria pedagogi, matematika, dan rekayasa perangkat lunak berstandar internasional.
Status: **PASSED (100%)**.

