# Track A Capstone Specification: Intelligent Visual Monitoring Using Computer Vision

**Jalur:** Track A — Computer Vision  
**Lingkungan Wajib:** CPU Lokal (PyTorch / NumPy / Scikit-learn — GPU tidak diwajibkan)

---

## 1. Deskripsi Proyek
Rancang dan validasi sistem pemantauan visual cerdas end-to-end menggunakan dataset citra sintetis terkendali atau dataset publik berlisensi terbuka (misalnya pemantauan kelengkapan keselamatan kerja berbasis kategori, deteksi objek sederhana, atau inspeksi cacat permukaan).

## 2. Daftar Deliverables Wajib
1. **Problem Definition:** Pernyataan masalah, skop *Intended Use*, dan *Prohibited Use*.
2. **Dataset & Provenance:** Dokumentasi sumber, lisensi, dan pembagian *Train/Val/Test* bebas kebocoran.
3. **Model & Pipeline Preprocessing:** Implementasi CNN / detektor ringan berbasis PyTorch di CPU.
4. **Inference Pipeline:** Fungsi inferensi lokal dengan penolakan ukuran/tipe input cacat.
5. **Evaluation:** Pelaporan metrik sesuai tugas (Accuracy/Macro-F1 untuk klasifikasi; IoU/mAP untuk deteksi) dibandingkan dengan *baseline* sederhana.
6. **Error Analysis:** Pemeriksaan minimal 3 contoh kasus visual yang gagal diprediksi beserta analisis penyebab.
7. **Latency Profiling:** Pengukuran ukuran model dan latensi CPU berulang (Mean/P95 ms).
8. **Responsible AI Assessment:** Risk Register privasi kamera, bias pencahayaan, dan Model Card.
9. **Documentation:** Panduan eksekusi ulang (`seed` deterministik) dan kesimpulan jujur tanpa klaim validasi lapangan nyata bila hanya diuji di laboratorium.

## 3. Rubrik Penilaian Track A (Total 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Keunggulan |
| :--- | :---: | :--- |
| **Problem Definition & Dataset Hygiene** | **15 poin** | Masalah visual jelas, lisensi valid, serta pemecahan train/val/test bersih tanpa kebocoran. |
| **Model & Visual Inference Implementation** | **30 poin** | Pra-pemrosesan spasial tepat, kode PyTorch modular, berjalan stabil pada CPU lokal. |
| **Evaluation & Error Analysis** | **25 poin** | Perbandingan baseline objektif, metrik visual akurat, dan analisis kasus gagal mendalam. |
| **Latency Profiling & Responsible AI** | **20 poin** | Benchmark latensi jujur, proteksi privasi visual, Risk Register, dan Model Card lengkap. |
| **Documentation & Reproducibility** | **10 poin** | Panduan eksekusi ulang lengkap dengan hasil konsisten. |

