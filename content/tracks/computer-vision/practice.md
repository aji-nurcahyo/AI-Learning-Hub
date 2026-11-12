# Track A — Computer Vision: Applied Laboratories

Panduan laboratorium terpadu untuk **Track A (Computer Vision)** yang menghubungkan skrip praktik runnable lintas chapter dan demonstrasi jalur CPU.

---

## 1. Lab Terapan Domain Computer Vision (Run Langsung)
```bash
python content/tracks/computer-vision/labs/cv_track_pipeline_demo.py
```
- **Output Terverifikasi:** Melatih `TinyVisionClassifier` di PyTorch CPU pada citra sintetis `24x24` RGB (`>90%` akurasi uji) dan mencatat latensi rata-rata inferensi CPU per gambar.

---

## 2. Daftar Enam Laboratorium Terintegrasi

1. **Lab 1 — Image Preprocessing & Filtering (Chapter 26):**
   - Jalankan `python content/level-3-deep-learning/chapter-26/code/color_space_conversion.py`
   - Jalankan `python content/level-3-deep-learning/chapter-26/code/image_filtering.py`
   - **Fokus:** Konversi RGB/Grayscale/HSV dan reduksi noise Gaussian vs Median berbasis metrik PSNR.
2. **Lab 2 — CNN Classification (Chapter 24):**
   - Jalankan `python content/level-3-deep-learning/chapter-24/code/train_cnn_vs_baseline.py`
   - **Fokus:** Perbandingan efisiensi parameter dan akurasi CNN 2D terhadap baseline linear.
3. **Lab 3 — Object Detection: Bounding Box, IoU & NMS (Chapter 25):**
   - Jalankan `python content/level-3-deep-learning/chapter-25/code/iou_calculation.py`
   - Jalankan `python content/level-3-deep-learning/chapter-25/code/non_maximum_suppression.py`
   - **Fokus:** Penapisan prediksi ganda dengan algoritma *Non-Maximum Suppression*.
4. **Lab 4 — Segmentation Overview (Chapter 26):**
   - Jalankan `python content/level-3-deep-learning/chapter-26/code/morphological_operations.py`
   - **Fokus:** Segmentasi berbasis wilayah warna dan operasi morfologi erosi/dilasi.
5. **Lab 5 — Object Tracking (Chapter 26):**
   - Jalankan `python content/level-3-deep-learning/chapter-26/code/centroid_tracking.py`
   - **Fokus:** Pelacakan lintas frame berbasis jarak centroid objek.
6. **Lab 6 — Lightweight Visual Inference (Chapter 37 & Track A):**
   - Jalankan `python content/level-4-advanced-ai/chapter-37/code/dynamic_quantization.py`
   - **Fokus:** Mengaudit ukuran model terkompresi dan latensi inferensi CPU.

