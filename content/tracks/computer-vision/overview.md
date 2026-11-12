# Track A — Computer Vision: Overview, Roadmap & Studi Kasus

**Fokus Jalur:** Pemrosesan citra digital, Convolutional Neural Networks (CNN), Object Detection (YOLO), pengantar segmentasi & tracking, serta inferensi visual ringan pada CPU.

---

## 1. Pengantar Jalur Spesialisasi
Jalur **Computer Vision** bersifat pilihan (opsional) bagi mahasiswa yang ingin memperdalam pemrosesan sinyal visual dua dimensi menggunakan fondasi matematika aljabar matriks hingga model deteksi dan pemantauan visual bertanggung jawab. Seluruh praktikum dirancang agar berjalan di CPU lokal tanpa mewajibkan GPU.

## 2. Roadmap Pembelajaran Terkurasi
1. **Tahap Fondasi:**
   - **Chapter 06** — Probability, Statistics & Linear Algebra for AI (operasi matriks & probabilitas).
   - **Chapter 07** — Python Ecosystem for AI (NumPy & Matplotlib).
   - **Chapter 08** — Data Preparation & Exploratory Analysis (inspeksi dan normalisasi dataset).
2. **Tahap Inti Computer Vision:**
   - **Chapter 21–23** — Neural Network Fundamentals, Backpropagation & PyTorch.
   - **Chapter 24** — Convolutional Neural Networks (konvolusi 2D, pooling, klasifikasi CNN).
   - **Chapter 25** — Object Detection & YOLO (Bounding Box, IoU, NMS, evaluasi mAP).
   - **Chapter 26** — Computer Vision Applications (ruang warna, filter, tepi Canny, segmentasi dasar, centroid tracking).
3. **Tahap Lanjutan & Keselamatan:**
   - **Chapter 31** — Transfer Learning & Fine-Tuning (Linear Probing & adaptasi fitur visual).
   - **Chapter 37** — Edge AI & Model Optimization (kuantisasi & profiling latensi).
   - **Chapter 38** — Multimodal AI Systems (pengaitan fitur citra dan teks).
   - **Chapter 39** — AI Safety, Security & Governance (privasi visual & Model Card).

## 3. Studi Kasus Domain Computer Vision

### Studi Kasus 1: Pemantauan Kualitas Produk Manufaktur Lini Konveyor
- **Konteks:** Mendeteksi cacat goresan atau penyok kemasan komponen pada citra kamera resolusi rendah secara lokal.
- **Pipeline:** Filter peredam derau (`Chapter 26`) $\to$ ekstraksi fitur CNN (`Chapter 24`) / *Transfer Learning* (`Chapter 31`) $\to$ *Confidence Threshold* & eskalasi manual operator (`Chapter 39`).
- **Batasan Penting:** Variasi pencahayaan pabrik menuntut kalibrasi ulang; deteksi uji laboratorium tidak menjamin kesiapan konveyor fisik tanpa uji lapangan.

### Studi Kasus 2: Pemantauan Area Pertanian Cerdas Berbasis Citra
- **Konteks:** Mengklasifikasi pola warna daun sehat vs gejala gangguan klorosis dari citra berlisensi/sintetis.
- **Pipeline:** Segmentasi ruang warna HSV/LAB (`Chapter 26`) + klasifikasi CNN ringan (`cv_track_pipeline_demo.py`).
- **Batasan Etika & Akurasi:** Wajib menyertakan *prohibited use* agar sistem tidak dijadikan penentu tunggal penggunaan pestisida tanpa verifikasi agronomis.

