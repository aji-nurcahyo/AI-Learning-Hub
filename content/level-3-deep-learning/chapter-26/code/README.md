# Laboratorium Pemrograman Python: Chapter 26 — Computer Vision Applications

Katalog modul dan skrip praktik pemrograman Python untuk **Bab 26: Computer Vision Applications**. Seluruh skrip dirancang secara mandiri (*standalone*), dapat dijalankan pada CPU komputer standar tanpa memerlukan akselerator GPU atau webcam fisik, serta diverifikasi kompatibel dengan Python 3.10+ (teruji pada Python 3.12.10, PyTorch 2.15.0+cpu, OpenCV 5.0.0, NumPy 2.5.4, dan Matplotlib 3.11.2).

---

## Daftar Berkas & Deskripsi Modul

| No | Nama Berkas Skrip | Fokus Materi & Algoritma | Keterangan & Luaran |
|:--:|---|---|---|
| 1 | `image_loading.py` | Pemuatan citra OpenCV (`cv2.imread`), inspeksi dimensi (H, W, C), tipe data `uint8`, dan verifikasi inversi kanal BGR vs RGB. | Pembuktian perbedaan kanal BGR vs RGB. |
| 2 | `color_space_conversion.py` | Konversi ruang warna ke Grayscale (luminansi tertimbang ITU-R BT.601), HSV, LAB, serta segmentasi warna berbasis ambang batas HSV. | Ekstraksi mask biner warna hijau terisolasi. |
| 3 | `image_filtering.py` | Penapisan spasial 2D: Box Filter, Gaussian Blur, Median Filter, dan Bilateral Filter (edge-preserving) dengan evaluasi kuantitatif PSNR. | Pembuktian keunggulan Median Filter pada salt-and-pepper noise. |
| 4 | `edge_detection.py` | Operator gradien Sobel X/Y, magnitudo dan arah gradien, turunan kedua Laplacian, serta algoritma multi-tahap Canny Edge Detector. | Deteksi kontur tertipis 1-piksel. |
| 5 | `morphological_operations.py` | Operasi morfologi matematika: Erosi, Dilasi, Opening (pembersihan derau luar), Closing (penutupan lubang dalam), dan Morphological Gradient. | Pembuktian sifat aljabar morfologi pada mask biner. |
| 6 | `threshold_segmentation.py` | Segmentasi citra: Global Thresholding, Otsu's Thresholding (minimisasi variansi intra-kelas), Adaptive Gaussian, dan Connected Components with Stats. | Ekstraksi instans objek, luas area, dan centroid. |
| 7 | `segmentation_metrics.py` | Metrik evaluasi segmentasi tingkat piksel: Mask IoU (Jaccard Index), Dice Similarity Coefficient (F1-Score), dan pembuktian hubungan matematis Dice vs IoU. | Evaluasi worked example analitis ($\text{IoU}=2/3$, $\text{Dice}=0.80$). |
| 8 | `centroid_tracking.py` | Pelacakan objek multi-frame (Tracking-by-Detection) berbasis jarak Euclidean centroid, asosiasi greedy, serta manajemen siklus hidup ID objek. | Preservasi konsistensi ID objek saat terjadi oklusi. |
| 9 | `video_frame_processing.py` | Pemrosesan aliran bingkai video: pembangkitan klip sintetis, iterasi frame-by-frame, pencatatan stempel waktu, dan pemrofilan latensi vs input video FPS. | Pengukuran FPS pemrosesan nyata pada CPU. |
| 10 | `object_counting_demo.py` | Penghitungan objek otomatis melintasi garis batas virtual (Tripwire line crossing) dengan mitigasi penghitungan ganda (double-counting). | Pelacakan arah masuk/keluar kendaraan. |
| 11 | `computer_vision_pipeline.py` | Pipeline visi komputer terpadu ujung-ke-ujung (End-to-End): Preprocessing $\to$ Detection $\to$ Tracking $\to$ Visualization dengan profil latensi setiap tahapan. | Throughput pipeline real-time teruji (> 30 FPS pada CPU). |

---

## Petunjuk Menjalankan Praktik

Pastikan virtual environment telah diaktifkan:
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Eksekusi seluruh skrip secara berurutan:
python content/level-3-deep-learning/chapter-26/code/image_loading.py
python content/level-3-deep-learning/chapter-26/code/color_space_conversion.py
python content/level-3-deep-learning/chapter-26/code/image_filtering.py
python content/level-3-deep-learning/chapter-26/code/edge_detection.py
python content/level-3-deep-learning/chapter-26/code/morphological_operations.py
python content/level-3-deep-learning/chapter-26/code/threshold_segmentation.py
python content/level-3-deep-learning/chapter-26/code/segmentation_metrics.py
python content/level-3-deep-learning/chapter-26/code/centroid_tracking.py
python content/level-3-deep-learning/chapter-26/code/video_frame_processing.py
python content/level-3-deep-learning/chapter-26/code/object_counting_demo.py
python content/level-3-deep-learning/chapter-26/code/computer_vision_pipeline.py
```

