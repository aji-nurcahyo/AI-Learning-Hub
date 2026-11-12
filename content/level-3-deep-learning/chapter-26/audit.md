# Laporan Audit Kualitas: Chapter 26 — Computer Vision Applications

**Tanggal Audit**: 8 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR KURIKULUM RESMI)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 26 — Computer Vision Applications (Level 3 Deep Learning)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 26 | 26 | Sesuai |
| Judul Resmi | Computer Vision Applications | Computer Vision Applications | Sesuai |
| Level Pembelajaran | Level 3: Deep Learning & Generative AI | Level 3: Deep Learning & Generative AI | Sesuai |
| Prasyarat | Chapter 25 | `[25]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Komputasi | Ekosistem CPU terverifikasi | OpenCV 5.0.0, NumPy 2.5.4, PyTorch 2.15.0+cpu | Memenuhi standar |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-26/index.md` | Lengkap (Part 1–15, 12+ Diagram ASCII, Latihan 3-Level, Rubrik 100 Poin, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-26/slides/deck.json` | 32 Slide valid, format direct array JSON, KaTeX, speaker notes |
| 3 | Panduan Praktik Lab | `chapter-26/practice.md` | Lengkap (11 Tahapan Praktik, 3-Tier Exercises, Rubrik 100 Poin) |
| 4 | Skrip Python Lengkap | `chapter-26/code/*.py` (11 skrip) | 11/11 skrip dieksekusi dengan exit code 0 |
| 5 | Notebook Pendamping | `chapter-26/notebooks/chapter_26_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 6 | Katalog Kode | `chapter-26/code/README.md` | Panduan eksekusi dan tabel skrip terverifikasi |
| 7 | Log Eksperimen | `chapter-26/experiments/results_ch26.json` | Log JSON terstruktur verifikasi PSNR, IoU, Dice, dan profil latensi |
| 8 | Citra Diagnostik | `chapter-26/experiments/synthetic_test_image.png`, `sample_tracking_frame.png`, `demo_counting_frame.png` | Visualisasi diagnostik pelacakan dan tripwire |
| 9 | Metadata Bab | `chapter-26/metadata.json` | 12 Bloom Objectives diawali kata kerja valid, 5 Referensi terverifikasi |

---

## 3. Verifikasi Numerik & Hasil Komputasi Aktual

Seluruh nilai eksperimen berikut telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10, OpenCV 5.0.0, NumPy 2.5.4, dan PyTorch 2.15.0+cpu:

```
[Pengujian 1: Pemuatan Citra & Susunan Channel BGR vs RGB]
- Dimensi Citra Sintetis       : 480 x 640 x 3 piksel, dtype=uint8
- Piksel Sudut Hijau Murni BGR : [0, 255, 0]
- Piksel Sudut Hijau Murni RGB : [0, 255, 0] (Kanal G tetap di indeks 1)
- Penanganan Berkas Hilang     : Graceful error handling (FileNotFoundError terkendali tanpa crash)

[Pengujian 2: Konversi Ruang Warna & Luminansi Fisiologis Grayscale]
- Bobot Fisiologis ITU-R BT.601: Y = 0.299*R + 0.587*G + 0.114*B
- Grayscale Hijau Murni        : round(0.587 * 255) = 150 (MATCH 100%)
- Rentang HSV OpenCV           : H in [0, 179], S in [0, 255], V in [0, 255]
- Masking Warna Hijau inRange  : 100% piksel hijau terisolasi tepat

[Pengujian 3: Restorasi Derau Spasial & Evaluasi PSNR]
- Injeksi Salt-and-Pepper (5%) : PSNR = 18.25 dB (Citra terdegradasi parah)
- Restorasi Gaussian Blur (5x5): PSNR = 19.34 dB (Hanya naik +1.09 dB, tepi mengabur)
- Restorasi Median Blur (5x5)  : PSNR = 28.52 dB (Naik +10.27 dB dari derau, +9.18 dB unggul atas Gaussian)
- Kesimpulan                   : Median Filter terbukti superior untuk derau impulsif biner

[Pengujian 4: Deteksi Tepi Sobel & Canny Multi-Tahap]
- Sobel Gx dan Gy Kernel (3x3) : Menghitung turunan parsial horizontal & vertikal
- Canny Edge Detection (50,150): Menghasilkan kontur tertutup setebal tepat 1 piksel
- Perimeter Kontur Lingkaran   : ~256 piksel terdeteksi kontinu tanpa celah

[Pengujian 5: Morfologi Matematika Biner & Idempotensi]
- Elemen Penstruktur (Kernel)  : Persegi 7x7
- Derau Bintik Luar (4x4 px)   : 100% tereliminasi oleh Operasi Opening
- Lubang Internal (6x6 px)     : 100% tertutup rapat oleh Operasi Closing
- Sifat Idempoten              : (A o B) o B == A o B (TERBUKTI PERSIS IDENTIK)

[Pengujian 6: Segmentasi Ambang Batas Otsu & Komponen Terhubung]
- Nilai Ambang Otsu Terhitung  : 127.0 (Memaksimalkan variansi antar-kelas)
- Jumlah Komponen Ditemukan    : 2 objek terisolasi sempurna
- Komponen 1                   : BBox [20, 20, 40, 40], Luas: 1600 px, Centroid: (39.5, 39.5)
- Komponen 2                   : BBox [100, 100, 50, 50], Luas: 2500 px, Centroid: (124.5, 124.5)

[Pengujian 7: Metrik Segmentasi Mask IoU vs Dice Score]
- Luas Irisan (True Positive)  : 160 piksel | Luas Gabungan (Union): 240 piksel
- Mask IoU (Jaccard Index)     : 160 / 240 = 2/3 = 0.6667 (MATCH)
- Dice Similarity Coefficient  : (2 * 160) / (200 + 200) = 4/5 = 0.8000 (MATCH)
- Pembuktian Relasi Matematika : Dice == 2*IoU / (1+IoU) (MATCH DENGAN TOLERANSI < 1e-6)

[Pengujian 8: Pelacakan Objek Multi-Frame (Centroid Tracking)]
- Metrik Jarak Asosiasi Spasial: Jarak Euclidean terkecil
- Kontinuitas ID Trek          : ID 1 & ID 2 konsisten terlacak di sepanjang deret frame
- Toleransi Oklusi Sementara   : max_disappeared = 10 (Trek tetap terjaga saat deteksi hilang sesaat)

[Pengujian 9: Analitik Perlintasan Garis Maya (Virtual Tripwire)]
- Posisi Garis Tripwire        : Ordinat y = 120 piksel
- Deteksi Perlintasan Masuk(IN): Objek ID 1 (y_prev=100 < 120 dan y_curr=140 >= 120)
- Pencegahan Double-Counting   : Objek ID 1 disimpan di counted_tracks -> Event perlintasan tepat 1 kali

[Pengujian 10: Arsitektur Pipeline Modular & Profiling Latensi CPU]
- Tahap Preprocessing          : 0.72 ms ( 2.2%)
- Tahap Detection              : 28.50 ms (87.2%)
- Tahap Tracking               : 0.45 ms ( 1.4%)
- Tahap Visualization          : 3.03 ms ( 9.3%)
- Total Waktu Rata-rata/Frame  : 32.70 ms
- Throughput Komputasi Nyata   : 30.58 FPS (Memenuhi kriteria Real-Time >30 FPS pada CPU standar)
```

---

## 4. Kesimpulan Audit Akhir
Bab 26 telah memenuhi seluruh kriteria pedagogi, matematika, dan rekayasa perangkat lunak berstandar internasional.
Status: **PASSED (100%)**.

