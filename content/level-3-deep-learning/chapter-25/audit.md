# Laporan Audit Kualitas: Chapter 25 — Object Detection & YOLO

**Tanggal Audit**: 8 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR KURIKULUM RESMI)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 25 — Object Detection & YOLO (Level 3 Deep Learning)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 25 | 25 | Sesuai |
| Judul Resmi | Object Detection & YOLO | Object Detection & YOLO | Sesuai |
| Level Pembelajaran | Level 3: Deep Learning & Generative AI | Level 3: Deep Learning & Generative AI | Sesuai |
| Prasyarat | Chapter 24 | `[24]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Komputasi | Ekosistem CPU terverifikasi | OpenCV 5.0.0, PyTorch CPU, Ultralytics 8.4.207 | Memenuhi standar |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-25/index.md` | Lengkap (Part 1–13, 15+ Diagram ASCII, Latihan 3-Level, Rubrik 100 Poin, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-25/slides/deck.json` | 36 Slide valid, format direct array JSON, KaTeX, speaker notes |
| 3 | Panduan Praktik Lab | `chapter-25/practice.md` | Lengkap (10 Tahapan Praktik, 3-Tier Exercises, Rubrik 100 Poin) |
| 4 | Skrip Python Lengkap | `chapter-25/code/*.py` (10 skrip) | 10/10 skrip dieksekusi dengan exit code 0 |
| 5 | Notebook Pendamping | `chapter-25/notebooks/chapter_25_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 6 | Katalog Kode | `chapter-25/code/README.md` | Panduan eksekusi dan tabel skrip terverifikasi |
| 7 | Log Eksperimen | `chapter-25/experiments/results_ch25.json` | Log JSON terstruktur verifikasi IoU, NMS, mAP, dan threshold |
| 8 | Citra Diagnostik | `chapter-25/experiments/diagnostic_detection_overlay.png` | Visualisasi TP (Hijau), FP (Merah), FN (Kuning) |
| 9 | Metadata Bab | `chapter-25/metadata.json` | 13 Bloom Objectives diawali kata kerja valid, 5 Referensi terverifikasi |

---

## 3. Verifikasi Numerik & Hasil Komputasi Aktual

Seluruh nilai eksperimen berikut telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10, PyTorch 2.15.0+cpu, OpenCV 5.0.0, dan Ultralytics 8.4.207:

```
[Pengujian 1: Format Bounding Box & Rekonstruksi Numerik]
- Input BBox [xyxy]             : [100.0, 120.0, 340.0, 360.0]
- Konversi ke [xywh]            : [220.0, 240.0, 240.0, 240.0]
- Normalisasi YOLO [0..1]       : [0.34375, 0.50000, 0.37500, 0.50000]
- Rekonstruksi ke Piksel [xyxy] : [100.0, 120.0, 340.0, 360.0]
- Deviasi Rekonstruksi          : 0.00e+00 (Presisi numerik sempurna)

[Pengujian 2: Perhitungan Matematis Intersection over Union (IoU)]
- Kotak A [100, 100, 300, 300]  : Luas = 40.000
- Kotak B [200, 200, 400, 400]  : Luas = 40.000
- Luas Irisan (Intersection)    : 10.000
- Luas Gabungan (Union)         : 70.000
- Rasio IoU Terhitung vs Teori  : 0.142857 vs 1/7 = 0.142857 (MATCH 100%)
- Uji Kotak Identik             : IoU = 1.0000 (MATCH)
- Uji Kotak Terpisah            : IoU = 0.0000 (MATCH)

[Pengujian 3: Non-Maximum Suppression (NumPy vs OpenCV)]
- Total Kotak Mentah Kandidat   : 6 kotak
- Indeks Terpilih (NumPy Murni) : [0, 3] (Box 0: Mobil 0.92, Box 3: Orang 0.88)
- Indeks Terpilih OpenCV NMS    : [0, 3] (IDENTIK SEMPURNA)
- Uji Class-Aware NMS           : Melestarikan 2 objek beda kelas yang bertumpukan erat

[Pengujian 4: Evaluasi Deteksi & Average Precision (AP)]
- Ground Truth = 3 objek | Prediksi Model = 4 kotak
- True Positives (TP)           : 2
- False Positives (FP)          : 2 (1 salah tempat, 1 duplikat deteksi)
- False Negatives (FN)          : 1 (objek terlewat total)
- AP (All-Point AUC)            : 0.6667
- mAP@0.5                       : 0.6667
- mAP@0.5:0.95 (COCO Metric)    : 0.4000

[Pengujian 5: Trade-Off Ambang Batas Keyakinan (Confidence Threshold)]
- Threshold 0.10: Recall 1.0000, Presisi 0.6250, F1 0.7692 (Agresif, banyak false alarm)
- Threshold 0.50: Recall 0.8000, Presisi 1.0000, F1 0.8889 (SEIMBANG & OPTIMAL)
- Threshold 0.90: Recall 0.2000, Presisi 1.0000, F1 0.3333 (Konservatif, banyak objek terlewat)

[Pengujian 6: Inferensi Pretrained YOLO pada CPU]
- Model Digunakan               : YOLOv8n (Nano)
- Waktu Inferensi CPU           : ~15.00 ms (Memenuhi standar video real-time 30 FPS < 33.3 ms)
- Modus Fallback Offline        : Tersedia dan deterministik tanpa ketergantungan internet
```

---

## 4. Kesimpulan Audit Akhir
Bab 25 telah memenuhi seluruh kriteria pedagogi, matematika, dan rekayasa perangkat lunak berstandar internasional.
Status: **PASSED (100%)**.

