# Panduan Praktik Laboratorium: Object Detection & YOLO

Selamat datang di modul panduan praktik laboratorium **Object Detection & YOLO** (Bab 25). Pada laboratorium ini, Anda akan mempraktikkan fondasi matematika, algoritma pemrosesan spasial, format data, evaluasi metrik, serta alur inferensi deteksi objek berbasis arsitektur **YOLO** pada jalur komputasi CPU: mulai dari konversi format koordinat bounding box (`xyxy` vs `xywh`), kalkulasi matematis *Intersection over Union* (IoU), penapisan duplikasi deteksi dengan *Non-Maximum Suppression* (NMS), parsing dan validasi dataset anotasi YOLO, hingga evaluasi kurva Precision-Recall, mAP@0.5, dan visualisasi diagnostik True Positive (TP), False Positive (FP), serta False Negative (FN).

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- Python: Versi 3.10+ (Direkomendasikan Python 3.12).
- Pustaka Inti: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `opencv-python>=4.8.0` (teruji 5.0.0), `ultralytics>=8.0.0` (teruji 8.4.207), `numpy>=2.0.0`, `matplotlib>=3.8.0`.
- Akselerator Perangkat Keras: Bersifat opsional. Seluruh skrip, algoritma NMS, perhitungan IoU, dan pipeline inferensi dirancang agar dapat dieksekusi secara instan pada CPU komputer standar dalam hitungan detik tanpa mewajibkan GPU.

### Struktur Repositori Praktik Bab 25:
```
content/level-3-deep-learning/chapter-25/
├── code/
│   ├── README.md
│   ├── bounding_box_formats.py
│   ├── iou_calculation.py
│   ├── non_maximum_suppression.py
│   ├── detection_metrics.py
│   ├── bounding_box_visualization.py
│   ├── yolo_annotation_parser.py
│   ├── yolo_dataset_validation.py
│   ├── pretrained_yolo_inference.py
│   ├── confidence_threshold_analysis.py
│   └── detection_result_visualization.py
├── experiments/
│   ├── results_ch25.json
│   ├── sample_yolo_detection.png
│   └── diagnostic_detection_overlay.png
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_25_companion.ipynb
├── slides/
│   ├── build_slides.py
│   └── deck.json
├── index.md
├── practice.md
├── metadata.json
└── audit.md
```

---

## 2. Panduan Langkah-demi-Langkah Praktik Laboratorium

### Langkah 1: Format Koordinat & Validasi Bounding Box
Jalankan skrip konversi format bounding box untuk memahami representasi geometris dan validasi batas citra:
```bash
python content/level-3-deep-learning/chapter-25/code/bounding_box_formats.py
```
- **Fokus Pengamatan**:
  - Amati transformasi matematis antara format Pascal VOC `[x1, y1, x2, y2]` dan format YOLO `[xc, yc, w, h]`.
  - Buktikan presisi numerik konversi bolak-balik: deviasi rekonstruksi bernilai $0.00 \times 10^0$.
  - Perhatikan fungsi validasi: kotak dengan $x_2 \le x_1$ atau koordinat di luar batas kanvas akan ditolak atau dipotong (*clipped*).

### Langkah 2: Kalkulasi Matematis Intersection over Union (IoU)
Hitung dan verifikasi derajat tumpang tindih spasial antara bounding box prediksi dan ground truth:
```bash
python content/level-3-deep-learning/chapter-25/code/iou_calculation.py
```
- **Fokus Pengamatan**:
  - Telusuri kasus terhitung manual (*worked example*): Kotak A $[100, 100, 300, 300]$ dan Kotak B $[200, 200, 400, 400]$ menghasilkan luas irisan 10.000, luas gabungan 70.000, dan rasio IoU tepat $1/7 \approx 0.142857$.
  - Uji kasus ekstrem: kotak identik ($\text{IoU} = 1.0$), kotak terpisah total ($\text{IoU} = 0.0$), dan kotak bersarang.
  - Periksa efisiensi matriks batch IoU berbasis penyiaran (*broadcasting*) NumPy $(N \times M)$.

### Langkah 3: Algoritma Non-Maximum Suppression (NMS)
Uji mekanisme penapisan deteksi ganda redundan menggunakan implementasi NumPy mandiri:
```bash
python content/level-3-deep-learning/chapter-25/code/non_maximum_suppression.py
```
- **Fokus Pengamatan**:
  - Perhatikan bagaimana kandidat kotak diurutkan secara menurun berdasarkan skor keyakinan (*Confidence Ranking*).
  - Amati eliminasi kotak: kotak dengan IoU $> 0.5$ terhadap kotak berskor tertinggi akan disingkirkan.
  - Bandingkan hasil NumPy mandiri terhadap fungsi resmi `cv2.dnn.NMSBoxes`: kedua implementasi menghasilkan indeks terpilih yang identik sempurna!
  - Pahami perbedaan *Class-Agnostic NMS* vs *Class-Aware NMS* saat dua objek berbeda kategori saling bertumpukan erat.

### Langkah 4: Evaluasi Metrik Deteksi (Precision, Recall, AP & mAP)
Pelajari alur evaluasi standar deteksi objek melalui pencocokan prediksi ke ground truth:
```bash
python content/level-3-deep-learning/chapter-25/code/detection_metrics.py
```
- **Fokus Pengamatan**:
  - Pahami aturan pencocokan serakah (*greedy matching*): prediksi hanya dapat mencocokkan satu GT dengan IoU tertinggi; duplikat kedua otomatis divonis sebagai False Positive (FP).
  - Hitung kurva Precision-Recall dan amati teknik interpolasi monotonik untuk menghitung Area Under Curve (Average Precision / AP).
  - Bandingkan protokol evaluasi longgar (mAP@0.5) terhadap protokol evaluasi multi-ambang batas ketat COCO (mAP@0.5:0.95).

### Langkah 5: Utilitas Visualisasi Bounding Box
Bangkitkan adegan citra sintetis dan gambarkan bounding box terformat:
```bash
python content/level-3-deep-learning/chapter-25/code/bounding_box_visualization.py
```
- **Fokus Pengamatan**:
  - Perhatikan pembuatan kartu label teks dengan latar belakang kontras agar informasi kelas dan skor keyakinan tetap terbaca jelas di atas berbagai tekstur citra.
  - Amati penggunaan palet warna unik untuk setiap indeks kategori objek.

### Langkah 6: Parser & Validator Format Anotasi YOLO
Eksplorasi struktur berkas teks anotasi YOLO (`.txt`) dan konversinya:
```bash
python content/level-3-deep-learning/chapter-25/code/yolo_annotation_parser.py
```
- **Fokus Pengamatan**:
  - Pelajari struktur baris standar: `<class_id> <x_center> <y_center> <width> <height>` di mana koordinat dinormalisasi ke interval $[0.0, 1.0]$.
  - Uji fungsi konversi otomatis dari format YOLO ke koordinat piksel Pascal VOC dan COCO.

### Langkah 7: Audit Integritas Dataset YOLO
Lakukan pemeriksaan kualitas dataset deteksi objek untuk mencegah kesalahan pelatihan:
```bash
python content/level-3-deep-learning/chapter-25/code/yolo_dataset_validation.py
```
- **Fokus Pengamatan**:
  - Deteksi berkas gambar tanpa label (*unmatched images*) dan label tanpa gambar (*unmatched labels*).
  - Periksa distribusi frekuensi kelas untuk mendeteksi ketimpangan data (*class imbalance*).
  - Audit partisi train/val/test untuk membuktikan ketiadaan kebocoran data (*zero data leakage*).

### Langkah 8: Pipeline Inferensi Pretrained YOLO pada CPU
Jalankan inferensi deteksi objek model terlatih pada perangkat CPU:
```bash
python content/level-3-deep-learning/chapter-25/code/pretrained_yolo_inference.py
```
- **Fokus Pengamatan**:
  - Amati waktu pemrosesan inferensi CPU (sekitar 15–30 ms per citra).
  - Perhatikan isolasi arsitektur: skrip mendukung pemuatan model terlatih Ultralytics dengan fallback otomatis ke inferensi deterministik jika bekerja secara offline bebas ketergantungan internet.
  - Periksa berkas citra hasil deteksi yang tersimpan di `experiments/sample_yolo_detection.png`.

### Langkah 9: Analisis Kompromi Ambang Batas Keyakinan (Confidence Threshold)
Investigasi trade-off antara deteksi agresif dan konservatif:
```bash
python content/level-3-deep-learning/chapter-25/code/confidence_threshold_analysis.py
```
- **Fokus Pengamatan**:
  - Amati data tabular pengaruh ambang batas keyakinan ($0.10, 0.25, 0.50, 0.75, 0.90$).
  - Buktikan bahwa menaikkan threshold meningkatkan presisi (mengurangi false alarm) tetapi menurunkan recall (melewatkan objek riil).
  - Identifikasi titik seimbang optimal F1-score pada threshold $0.25 - 0.50$.

### Langkah 10: Visualisasi Diagnostik Kesalahan Deteksi (TP, FP, FN)
Analisis kesalahan model melalui penguraian visual berkode warna:
```bash
python content/level-3-deep-learning/chapter-25/code/detection_result_visualization.py
```
- **Fokus Pengamatan**:
  - Buka berkas `experiments/diagnostic_detection_overlay.png`.
  - Pelajari kode warna diagnostik: Hijau menandai True Positive, Merah menandai False Positive (halusinasi/duplikasi), dan Kuning menandai False Negative (objek ground-truth yang terlewat).

---

## 3. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Konversi Koordinat**: Sebuah citra beresolusi $1920 \times 1080$ piksel memiliki bounding box objek dengan koordinat absolut $[x_1, y_1, x_2, y_2] = [480, 270, 1440, 810]$. Tuliskan representasi anotasi format YOLO $(<class\_id> <x_c> <y_c> <w> <h>)$ jika ID kelas adalah 0!
   - *Jawaban*:
     - Lebar $w_{\text{abs}} = 1440 - 480 = 960$. Tinggi $h_{\text{abs}} = 810 - 270 = 540$.
     - Titik Pusat $x_{c,\text{abs}} = 480 + 960 / 2 = 960$. $y_{c,\text{abs}} = 270 + 540 / 2 = 540$.
     - Normalisasi: $x_c = 960 / 1920 = 0.50$, $y_c = 540 / 1080 = 0.50$, $w = 960 / 1920 = 0.50$, $h = 540 / 1080 = 0.50$.
     - Baris YOLO: `0 0.500000 0.500000 0.500000 0.500000`.

2. **Perhitungan IoU Kasus Nyata**: Diberikan Bounding Box Ground Truth $G = [50, 50, 150, 150]$ dan Bounding Box Prediksi $P = [100, 50, 200, 150]$. Hitung nilai Intersection over Union (IoU) antara kedua kotak tersebut!
   - *Jawaban*:
     - Luas $G = (150 - 50) \times (150 - 50) = 100 \times 100 = 10.000$.
     - Luas $P = (200 - 100) \times (150 - 50) = 100 \times 100 = 10.000$.
     - Irisan: $x_1 = \max(50, 100) = 100$, $y_1 = \max(50, 50) = 50$, $x_2 = \min(150, 200) = 150$, $y_2 = \min(150, 150) = 150$.
     - Luas Irisan: $(150 - 100) \times (150 - 50) = 50 \times 100 = 5.000$.
     - Luas Gabungan (Union): $10.000 + 10.000 - 5.000 = 15.000$.
     - $\text{IoU} = 5.000 / 15.000 = 1/3 \approx 0.3333$.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Kegagalan NMS pada Objek Berdekatan**: Seorang insinyur menguji sistem deteksi pejalan kaki di trotoar padat dan mendapati bahwa ketika dua orang pejalan kaki berjalan berdampingan rapat, model hanya mendeteksi satu orang saja. Jelaskan penyebab algoritmik masalah ini dan solusinya!
   - *Jawaban*:
     - *Penyebab*: Ambang batas IoU pada tahap NMS (`iou_threshold`) disetel terlalu rendah (misal 0.3 atau 0.4). Ketika dua orang berjalan beriringan rapat, kotak pembatas mereka saling beririsan dengan IoU sekitar 0.45. Algoritma NMS menganggap kotak kedua sebagai duplikat dari orang pertama dan meniadakannya secara keliru.
     - *Solusi*: Naikkan ambang batas IoU NMS ke $0.60 - 0.70$, atau gunakan algoritma *Soft-NMS* yang mendegradasi skor keyakinan kotak bertumpuk secara halus (*decay*) daripada menghapusnya secara biner.

2. **Dilema Evaluasi mAP@0.5 vs mAP@0.5:0.95**: Mengapa sebuah model deteksi objek dapat mencatat nilai mAP@0.5 yang sangat tinggi (misal 92%), namun nilai mAP@0.5:0.95 anjlok menjadi 48%?
   - *Jawaban*:
     - mAP@0.5 hanya mensyaratkan kotak prediksi bertumpang tindih sebesar $50\%$ dengan ground-truth. Jika model berhasil mendeteksi objek secara kasar tetapi koordinat batas tepinya longgar/kurang presisi, model tetap memperoleh nilai sempurna pada ambang batas 0.50.
     - Namun pada ambang batas yang lebih tinggi (seperti 0.85 hingga 0.95), ketidaktepatan batas kotak piksel menyebabkan kotak divonis sebagai False Positive, menurunkan nilai integral kurva PR secara drastis.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Analisis Latensi Video Real-Time**: Sebuah sistem pengawasan lalu lintas memproses video berkecepatan 30 FPS. Tahapan prapemrosesan citra membutuhkan 4 ms, inferensi model membutuhkan 22 ms, dan pascapemrosesan (NMS + visualisasi) membutuhkan 8 ms pada CPU.
   - Berapakah total waktu pemrosesan per frame?
   - Apakah sistem ini dapat beroperasi secara real-time pada video 30 FPS tanpa terjadi penumpukan antrean frame (*frame drop*)?
   - *Jawaban*:
     - Total latensi per frame: $4 + 22 + 8 = 34 \text{ ms}$.
     - Durasi anggaran waktu satu frame pada 30 FPS: $1000 \text{ ms} / 30 \approx 33.33 \text{ ms}$.
     - Karena $34 \text{ ms} > 33.33 \text{ ms}$, sistem sedikit melampaui batas anggaran waktu (*processing FPS* $\approx 29.41$ FPS), sehingga akan terjadi penumpukan antrean frame jika tidak dilakukan optimasi ringan (seperti mengecilkan resolusi visualisasi atau menerapkan frame sampling).

---

## 4. Mini Project Chapter 25

### Judul Tugas Proyek:
**"Object Detection with YOLO on a Small Visual Dataset"**

### Deskripsi Tugas:
Rancang, bangun, dan evaluasi pipeline deteksi objek mandiri menggunakan **YOLO** pada dataset citra kecil yang terverifikasi (misalnya deteksi kendaraan dan pejalan kaki sintetis atau dataset terbuka berlisensi CC/MIT). Anda diminta untuk:
1. Mendefinisikan masalah deteksi objek spesifik, sumber data, dan lisensi penggunaan data secara etis.
2. Memeriksa format anotasi YOLO (`.txt`), melakukan audit integritas dataset, dan memastikan partisi data train/val/test terisolasi tanpa kebocoran data (*zero data leakage*).
3. Mengimplementasikan alur inferensi model deteksi objek teroptimasi CPU (misalnya YOLOv8 Nano) dengan visualisasi bounding box, label kelas, dan skor keyakinan.
4. Mengevaluasi performa model melalui pencocokan terhadap ground-truth beranotasi: menghitung True Positive, False Positive, False Negative, Precision, Recall, serta metrik mAP.
5. Melakukan analisis kesalahan (*error analysis*) dengan membedah kasus kegagalan deteksi (False Positive akibat latar belakang dan False Negative akibat objek kecil/terhalang).
6. Mendokumentasikan pertimbangan etika (Responsible AI), privasi citra, latensi inferensi CPU, dan batas kemampuan model.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Problem & Dataset Understanding** | 15 Poin | Perumusan masalah deteksi objek lugas, kejelasan sumber dataset dan lisensi penggunaan etis, kepatuhan struktur format anotasi YOLO, serta audit integritas dataset terbukti bebas kebocoran data. |
| **2. Detection Implementation** | 30 Poin | Implementasi pipeline inferensi YOLO pada CPU tereksekusi mulus tanpa galat, modul konversi koordinat (`xyxy` vs `xywh`) presisi, algoritma NMS terintegrasi tepat, dan visualisasi bounding box rapi serta mudah dibaca. |
| **3. Evaluation & Analysis** | 25 Poin | Evaluasi pencocokan prediksi ke ground truth berbasis IoU transparan, pelaporan metrik Precision, Recall, AP, dan mAP akurat tanpa fabrikasi angka, serta diseksi visual kesalahan deteksi (TP, FP, FN). |
| **4. Computational Considerations & Limitations** | 15 Poin | Pengukuran empiris waktu inferensi (ms) pada CPU, analisis trade-off ambang batas keyakinan (Confidence Threshold), dan pembahasan jujur mengenai batas keandalan model pada kondisi pencahayaan ekstrem atau oklusi. |
| **5. Documentation & References** | 15 Poin | Kode bersih terstruktur modular tanpa duplikasi, dokumentasi eksekusi reproducible, kepatuhan etika privasi citra, dan pencantuman 5 referensi akademik primer yang relevan. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 10 skrip Python di direktori `code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Formula matematis IoU terbukti menghasilkan nilai analitis yang cocok pada worked example ($1/7 \approx 0.142857$).
- [ ] Implementasi algoritma NMS mandiri menghasilkan keputusan supresi yang identik terhadap OpenCV `NMSBoxes`.
- [ ] Format anotasi YOLO ternormalisasi $[0.0, 1.0]$ tervalidasi dan dapat dikonversi ke VOC/COCO.
- [ ] Integritas dataset (pemeriksaan berkas hilang dan ketiadaan kebocoran data) telah diaudit.
- [ ] Pipeline inferensi YOLO pada CPU berjalan efisien dan menyimpan citra deteksi di `experiments/`.
- [ ] Analisis trade-off variasi ambang batas keyakinan (0.10 hingga 0.90) terdokumentasi rapi.
- [ ] Visualisasi diagnostik berkode warna (Hijau TP, Merah FP, Kuning FN) dihasilkan dan diverifikasi.
- [ ] Berkas log hasil eksperimen `experiments/results_ch25.json` terisi dengan metrik valid.
- [ ] Notebook pendamping `chapter_25_companion.ipynb` tervalidasi bersih dalam format v4.

