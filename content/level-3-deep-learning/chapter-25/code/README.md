# Laboratorium Pemrograman Python: Chapter 25 — Object Detection & YOLO

Katalog modul dan skrip praktik pemrograman Python untuk **Bab 25: Object Detection & YOLO**. Seluruh skrip dirancang secara mandiri (*standalone*), dapat dijalankan pada CPU komputer standar, serta diverifikasi kompatibel dengan Python 3.10+ (teruji pada Python 3.12.10, PyTorch 2.15.0+cpu, OpenCV 5.0.0, dan Ultralytics 8.4.207).

---

## Daftar Berkas & Deskripsi Modul

| No | Nama Berkas Skrip | Fokus Materi & Algoritma | Keterangan & Luaran |
|:--:|---|---|---|
| 1 | `bounding_box_formats.py` | Konversi koordinat `xyxy` $\leftrightarrow$ `xywh`, normalisasi $[0, 1]$, validasi geometris, dan clipping kanvas citra. | Verifikasi presisi rekonstruksi numerik ($\Delta < 10^{-6}$). |
| 2 | `iou_calculation.py` | Perhitungan matematis luas irisan (*intersection*), luas gabungan (*union*), dan rasio IoU skalar serta matriks batch penyiaran $(N \times M)$. | Pembuktian worked example analitis ($1/7 \approx 0.142857$). |
| 3 | `non_maximum_suppression.py` | Implementasi mandiri algoritma NMS berbasis NumPy (Class-Aware dan Class-Agnostic) dengan perangkingan skor keyakinan dan penapisan IoU. | Validasi kecocokan 100% terhadap `cv2.dnn.NMSBoxes`. |
| 4 | `detection_metrics.py` | Algoritma pencocokan greedy prediksi ke ground truth (TP, FP, FN), kurva Precision-Recall terinterpolasi, Average Precision (AP), serta evaluasi mAP@0.5 dan mAP@0.5:0.95. | Protokol evaluasi standar PASCAL VOC & COCO. |
| 5 | `bounding_box_visualization.py` | Utilitas rendering visual bounding box, label kelas, kartu skor keyakinan, dan palet warna kontras tinggi menggunakan OpenCV. | Pembangkitan kanvas adegan sintetis deterministik. |
| 6 | `yolo_annotation_parser.py` | Parser, validator rentang koordinat, serta konverter format label YOLO (`<class_id> <xc> <yc> <w> <h>`) ke Pascal VOC dan COCO. | Validasi struktur baca/tulis berkas teks `.txt`. |
| 7 | `yolo_dataset_validation.py` | Skrip audit integritas dataset: deteksi berkas gambar/label hilang (*unmatched*), validasi format, distribusi frekuensi kelas, dan deteksi kebocoran data (*data leakage*). | Audit dataset lokal di direktori `images/` dan `labels/`. |
| 8 | `pretrained_yolo_inference.py` | Pipeline inferensi model deteksi objek terlatih (YOLOv8n) pada CPU, pengukuran latensi waktu inferensi (ms), dan mekanisme fallback deterministik jika offline. | Menghasilkan berkas citra deteksi di `experiments/`. |
| 9 | `confidence_threshold_analysis.py` | Eksperimen komparasi dampak variasi ambang batas skor keyakinan ($0.10, 0.25, 0.50, 0.75, 0.90$) terhadap kuantitas prediksi, presisi, dan recall. | Membuktikan kompromi (*trade-off*) false alarm vs missed detections. |
| 10 | `detection_result_visualization.py` | Rendering diagnostik perbandingan prediksi vs ground-truth dengan kode warna: Hijau (TP), Merah (FP), dan Kuning (FN). | Menghasilkan citra overlay diagnostik di `experiments/`. |

---

## Petunjuk Menjalankan Praktik

Pastikan virtual environment telah diaktifkan:
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Eksekusi seluruh skrip secara berurutan:
python content/level-3-deep-learning/chapter-25/code/bounding_box_formats.py
python content/level-3-deep-learning/chapter-25/code/iou_calculation.py
python content/level-3-deep-learning/chapter-25/code/non_maximum_suppression.py
python content/level-3-deep-learning/chapter-25/code/detection_metrics.py
python content/level-3-deep-learning/chapter-25/code/bounding_box_visualization.py
python content/level-3-deep-learning/chapter-25/code/yolo_annotation_parser.py
python content/level-3-deep-learning/chapter-25/code/yolo_dataset_validation.py
python content/level-3-deep-learning/chapter-25/code/pretrained_yolo_inference.py
python content/level-3-deep-learning/chapter-25/code/confidence_threshold_analysis.py
python content/level-3-deep-learning/chapter-25/code/detection_result_visualization.py
```

