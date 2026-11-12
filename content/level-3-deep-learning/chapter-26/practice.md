# Panduan Praktik Laboratorium: Computer Vision Applications

Selamat datang di modul panduan praktik laboratorium **Computer Vision Applications** (Bab 26). Pada laboratorium ini, Anda akan mempraktikkan fondasi pemrosesan citra digital, segmentasi spasial, pelacakan temporal multi-objek, dan rekayasa pipeline pemantauan visual terintegrasi berbasis **OpenCV** dan **NumPy** pada jalur komputasi CPU: mulai dari representasi citra dan konversi ruang warna (BGR, RGB, HSV, LAB), penapisan spasial dan reduksi derau impulsif, deteksi tepi Sobel dan Canny, operasi morfologi matematis, segmentasi ambang batas Otsu dan ekstraksi komponen terhubung, evaluasi metrik segmentasi (Mask IoU dan Dice Score), hingga pelacakan berbasis deteksi (*Centroid Tracking*), penghitungan garis maya (*virtual tripwire*), serta profiling performa latensi pipeline modular secara *end-to-end*.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- Python: Versi 3.10+ (Direkomendasikan Python 3.12).
- Pustaka Inti: `opencv-python>=4.8.0` (teruji 5.0.0), `numpy>=2.0.0` (teruji 2.5.4), `matplotlib>=3.8.0`, `torch>=2.0.0`.
- Akselerator Perangkat Keras: Bersifat opsional. Seluruh 11 skrip pengujian, algoritma pelacak centroid, operasi morfologi, dan pipeline video sintetis dirancang untuk dieksekusi secara instan pada CPU komputer standar dalam hitungan detik tanpa mewajibkan GPU.

### Struktur Repositori Praktik Bab 26:
```
content/level-3-deep-learning/chapter-26/
├── code/
│   ├── README.md
│   ├── image_loading.py
│   ├── color_space_conversion.py
│   ├── image_filtering.py
│   ├── edge_detection.py
│   ├── morphological_operations.py
│   ├── threshold_segmentation.py
│   ├── segmentation_metrics.py
│   ├── centroid_tracking.py
│   ├── video_frame_processing.py
│   ├── object_counting_demo.py
│   └── computer_vision_pipeline.py
├── experiments/
│   ├── results_ch26.json
│   ├── synthetic_test_image.png
│   ├── sample_tracking_frame.png
│   └── demo_counting_frame.png
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_26_companion.ipynb
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

### Langkah 1: Pemuatan Citra, Dimensi, & Verifikasi Channel BGR
Jalankan skrip pemuatan citra untuk memvalidasi susunan matriks piksel dan penanganan galat berkas:
```bash
python content/level-3-deep-learning/chapter-26/code/image_loading.py
```
- **Fokus Pengamatan**:
  - Amati struktur tensor 3D citra $(H, W, C)$ dan tipe data standar `uint8` $[0, 255]$.
  - Buktikan bahwa urutan channel bawaan memori OpenCV adalah Blue-Green-Red (`BGR`), berbeda dengan format representasi pustaka visualisasi web/display standar (`RGB`).
  - Uji ketahanan mekanisme penanganan galat (*graceful error handling*) saat berkas citra tidak ditemukan di disk tanpa memicu *crash*.

### Langkah 2: Konversi Ruang Warna & Penapisan Warna (HSV/LAB)
Pelajari perbedaan representasi ruang warna dan isolasi objek berbasis warna:
```bash
python content/level-3-deep-learning/chapter-26/code/color_space_conversion.py
```
- **Fokus Pengamatan**:
  - Buktikan konversi ke Grayscale mematuhi pembobotan luminansi ITU-R BT.601 ($Y = 0.299R + 0.587G + 0.114B$), di mana piksel hijau murni $(0, 255, 0)$ terkonversi menjadi intensitas 150.
  - Pahami ruang warna HSV (*Hue, Saturation, Value*) yang memisahkan komponen kromatisitas murni (Hue) dari intensitas cahaya (Value).
  - Amati segmentasi biner warna hijau menggunakan fungsi penapisan ambang rentang ganda `cv2.inRange`.

### Langkah 3: Penyaringan Spasial & Reduksi Derau (*Denoising*)
Bandingkan filter konvolusi linear versus filter non-linear dalam mereduksi derau impulsif:
```bash
python content/level-3-deep-learning/chapter-26/code/image_filtering.py
```
- **Fokus Pengamatan**:
  - Amati dampak injeksi *Salt-and-Pepper Noise* terhadap penurunan rasio sinyal puncak terhadap derau (PSNR).
  - Bandingkan hasil restorasi: **Gaussian Filter** (linear) mengaburkan derau bersama tepi objek (PSNR $\approx 19.34\text{ dB}$), sedangkan **Median Filter** (non-linear) mengeliminasi derau impulsif secara radikal (PSNR $\approx 28.52\text{ dB}$, keunggulan $+9.18\text{ dB}$).
  - Pahami peran Bilateral Filter dalam menghaluskan tekstur datar sembari mempertahankan ketajaman kontur tepi (*edge-preserving*).

### Langkah 4: Deteksi Tepi: Gradien Sobel & Detektor Canny
Analisis diskontinuitas intensitas spasial menggunakan operator turunan dan algoritma multi-tahap Canny:
```bash
python content/level-3-deep-learning/chapter-26/code/edge_detection.py
```
- **Fokus Pengamatan**:
  - Hitung turunan pertama arah horisontal ($G_x$) dan vertikal ($G_y$) menggunakan kernel Sobel $3 \times 3$, lalu cari magnitudo gradien $\sqrt{G_x^2 + G_y^2}$.
  - Amati empat tahapan deterministik algoritma Canny: Gaussian smoothing $\rightarrow$ gradien Sobel $\rightarrow$ penekanan non-maksimum (*Non-Maximum Suppression*) untuk menipiskan tepi $\rightarrow$ penentuan ambang histeresis ganda (*hysteresis thresholding*).
  - Verifikasi bahwa hasil Canny menghasilkan kontur tepi tipis dengan lebar tepat satu piksel.

### Langkah 5: Operasi Morfologi Matematis
Eksplorasi pembentukan struktur biner menggunakan elemen penstruktur (*structuring element* / kernel):
```bash
python content/level-3-deep-learning/chapter-26/code/morphological_operations.py
```
- **Fokus Pengamatan**:
  - Erosi ($A \ominus B$): mengecilkan objek dan mengikis batas tepi luar.
  - Dilasi ($A \oplus B$): memperbesar objek dan mengisi celah tipis.
  - **Opening** (Erosi diikuti Dilasi): membersihkan derau bintik putih terisolasi (*false positive*) tanpa mengubah ukuran makro objek.
  - **Closing** (Dilasi diikuti Erosi): menutup lubang internal (*false negative*) dan merekatkan fraktur objek.
  - **Morphological Gradient** (Dilasi $-$ Erosi): mengekstrak perimeter batas terluar objek biner.

### Langkah 6: Segmentasi Ambang Batas & Analisis Komponen Terhubung
Segmentasi objek dari latar belakang dan ekstraksi geometri wilayah terhubung:
```bash
python content/level-3-deep-learning/chapter-26/code/threshold_segmentation.py
```
- **Fokus Pengamatan**:
  - Pelajari penentuan nilai ambang optimal otomatis metode Otsu yang memaksimalkan variansi antar-kelas (*inter-class variance*).
  - Pahami keunggulan Adaptive Gaussian Thresholding pada citra dengan gradien pencahayaan yang tidak homogen.
  - Ekstraksi properti komponen terhubung via `cv2.connectedComponentsWithStats`: label ID, bounding box $(x, y, w, h)$, luas area piksel, dan koordinat titik berat (*centroid*) $(\bar{x}, \bar{y})$.

### Langkah 7: Evaluasi Kualitas Segmentasi (Mask IoU & Dice Score)
Kalkulasi metrik evaluasi wilayah piksel biner dan pembuktian relasi matematisnya:
```bash
python content/level-3-deep-learning/chapter-26/code/segmentation_metrics.py
```
- **Fokus Pengamatan**:
  - Hitung luas irisan (True Positive), luas gabungan (Union), dan total piksel ground truth serta prediksi.
  - Verifikasi perhitungan analitis: kasus irisan 160 piksel dan gabungan 240 piksel menghasilkan $\text{Mask IoU} = 160/240 = 2/3 \approx 0.6667$ dan $\text{Dice Score} = (2 \times 160) / (200 + 200) = 320/400 = 0.8000$.
  - Buktikan secara komputasional relasi deterministik: $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$.

### Langkah 8: Pelacakan Objek Multi-Objek (Centroid Tracking)
Simulasikan pelacakan multi-frame menggunakan pendekatan *Tracking-by-Detection*:
```bash
python content/level-3-deep-learning/chapter-26/code/centroid_tracking.py
```
- **Fokus Pengamatan**:
  - Amati asosiasi spasial antar-frame berbasis matriks jarak Euclidean $D_{i,j} = \sqrt{(c_{x,i} - c_{x,j})^2 + (c_{y,i} - c_{y,j})^2}$.
  - Pelajari siklus hidup trek (*track lifecycle*): inisialisasi ID baru (*registration*), pembaruan koordinat (*update*), toleransi oklusi sementara via penghitung *disappeared*, hingga deregistrasi saat objek hilang permanen.
  - Buktikan bahwa ID objek tetap konsisten di sepanjang deret frame video sintetis.

### Langkah 9: Pemrosesan Deret Frame Video & Perhitungan FPS
Bangkitkan dan proses klip video sintetis secara modular:
```bash
python content/level-3-deep-learning/chapter-26/code/video_frame_processing.py
```
- **Fokus Pengamatan**:
  - Pahami antarmuka `cv2.VideoCapture` dan `cv2.VideoWriter`.
  - Bedakan antara laju putar video input (*Input Video FPS*) dan laju pemrosesan komputasi (*Processing FPS*).
  - Ukur latensi rata-rata pemrosesan per frame untuk memastikan kesiapan sistem terhadap persyaratan waktu nyata (*real-time processing*).

### Langkah 10: Penghitungan Objek Melintasi Garis Maya (*Virtual Tripwire*)
Implementasikan deteksi perlintasan arah dan mitigasi penghitungan ganda:
```bash
python content/level-3-deep-learning/chapter-26/code/object_counting_demo.py
```
- **Fokus Pengamatan**:
  - Definisikan garis batas virtual pada ordinat $y = y_{\text{tripwire}}$.
  - Deteksi arah pergerakan: objek melintas dari $y_{t-1} < y_{\text{tripwire}}$ ke $y_t \ge y_{\text{tripwire}}$ dihitung sebagai kendaraan masuk (*IN*), sedangkan kebalikan dihitung sebagai keluar (*OUT*).
  - Terapkan mekanisme pencatatan ID trek terhitung (*counted track set*) untuk mencegah insiden penghitungan ganda (*double-counting prevention*).

### Langkah 11: Arsitektur Pipeline Modular Terintegrasi & Profiling Latensi
Integrasikan seluruh komponen ke dalam pipeline end-to-end terstruktur:
```bash
python content/level-3-deep-learning/chapter-26/code/computer_vision_pipeline.py
```
- **Fokus Pengamatan**:
  - Amati modularisasi kelas arsitektur bersih: `FramePreprocessor`, `ObjectDetector`, `CentroidTracker`, dan `Visualizer`.
  - Telaah profil distribusi waktu komputasi CPU: tahap deteksi mengambil porsi terbesar (~87%), sementara pelacakan dan pra-pemrosesan berjalan sangat cepat (<1 ms).
  - Buktikan bahwa keseluruhan pipeline mampu mencapai throughput $> 30\text{ FPS}$ pada prosesor CPU standar.

---

## 3. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Konversi Ruang Warna ke Luminansi Grayscale**:
   Sebuah piksel citra digital memiliki nilai RGB $[R=100, G=200, B=50]$. Berapakah intensitas abu-abu (*grayscale*) piksel tersebut menurut standar bobot fisiologis manusia ITU-R BT.601 ($Y = 0.299R + 0.587G + 0.114B$)?
   - *Jawaban*:
     - $Y = 0.299 \times 100 + 0.587 \times 200 + 0.114 \times 50$
     - $Y = 29.90 + 117.40 + 5.70 = 153.00$
     - Nilai piksel grayscale terkuantisasi adalah $153$ (rentang $0 - 255$).

2. **Perhitungan Metrik Segmentasi (Dice dari IoU)**:
   Sebuah model segmentasi citra medis menghasilkan evaluasi wilayah biner dengan nilai Jaccard Index (Mask IoU) sebesar $0.60$ ($3/5$). Tanpa perlu mengetahui matriks piksel mentah, berapakah nilai Dice Similarity Coefficient (F1-Score) untuk segmentasi tersebut?
   - *Jawaban*:
     - Gunakan formula relasi deterministik: $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$
     - $\text{Dice} = \frac{2 \cdot 0.60}{1 + 0.60} = \frac{1.20}{1.60} = \frac{3}{4} = 0.7500$ (atau $75.0\%$).

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Kegagalan Penapisan Warna HSV pada Variasi Pencahayaan**:
   Seorang insinyur Computer Vision menggunakan penapisan HSV untuk mendeteksi bola tenis kuning pada lapangan luar ruangan. Algoritma bekerja sempurna pada pagi hari berawan, namun gagal total mendeteksi bola pada siang hari terik saat muncul bayangan pekat dan pantulan sinar matahari langsung. Jelaskan penyebab algoritmik masalah ini dan berikan solusinya!
   - *Jawaban*:
     - *Penyebab*: Pantulan sinar matahari langsung (*specular highlights*) meningkatkan nilai intensitas ke tingkat saturasi maksimum ($V \approx 255$) sembari mendegradasi saturasi ($S \rightarrow 0$), menyebabkan warna bola memutih di luar rentang ambang Hue kuning. Sebaliknya, bayangan pekat menurunkan intensitas ($V \rightarrow 0$), merusak ekstraksi komponen Hue.
     - *Solusi*: (1) Perlebar batas toleransi saturasi dan nilai ($S$ dan $V$) pada `cv2.inRange`. (2) Terapkan normalisasi pencahayaan sebelum penapisan, seperti *Contrast Limited Adaptive Histogram Equalization* (CLAHE) pada kanal $V$ atau kanal $L$ dalam ruang warna LAB. (3) Beralih ke model deteksi objek berbasis fitur spasial mendalam (*deep learning object detector*) yang invarian terhadap iluminasi.

2. **Dilema Ukuran Kernel Operasi Morfologi**:
   Saat membersihkan derau bintik putih pada citra biner teks hasil pemindaian dokumen, seorang peneliti menerapkan operasi Opening dengan kernel penstruktur persegi $9 \times 9$ piksel. Seluruh derau bintik putih berhasil hilang, namun huruf-huruf tipis seperti 'i', 'l', dan tanda baca titik ikut terhapus. Jelaskan prinsip morfologis di balik fenomena ini dan bagaimana memperbaikinya!
   - *Jawaban*:
     - *Prinsip*: Operasi Opening diawali dengan Erosi ($A \ominus B$). Erosi hanya mempertahankan piksel di mana seluruh elemen penstruktur $B$ dapat dimuat sepenuhnya di dalam objek target. Karena tebal guratan huruf 'i', 'l', dan titik lebih kecil dari diameter kernel $9 \times 9$, struktur huruf tersebut tergerus habis pada fase erosi dan tidak dapat direkonstruksi kembali oleh fase dilasi berikutnya.
     - *Solusi*: Perkecil ukuran kernel penstruktur menjadi $3 \times 3$ atau $2 \times 2$ piksel, atau gunakan kernel elips/silang yang lebih ramping. Sebagai alternatif, terapkan analisis komponen terhubung (*connected components*) dan hapus komponen bintik hanya jika luas area pikselnya berada di bawah ambang batas ukuran tertentu tanpa mengikis struktur huruf tipis.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Pencegahan ID Switching pada Pelacakan Centroid Saat Terjadi Oklusi Penuh**:
   Dua objek pejalan kaki bergerak berpapasan di lorong sempit. Pada frame $t=50$, keduanya berjarak 80 piksel. Pada frame $t=60$, keduanya saling menutupi (oklusi) sehingga detektor hanya menghasilkan satu bounding box gabungan. Pada frame $t=70$, keduanya terpisah kembali namun ID pelacakan mereka tertukar (*ID switching*).
   - Mengapa algoritma Centroid Tracking murni berbasis jarak Euclidean rentan terhadap *ID switching* dalam skenario ini?
   - Rancang modifikasi algoritmik untuk memitigasi masalah tersebut tanpa menggunakan model deep learning berat!
   - *Jawaban*:
     - *Penyebab*: Centroid Tracking murni bersifat amnesia tanpa memori spasial dinamis—ia hanya mengasosiasikan posisi instan frame berjalan dengan frame sebelumnya via jarak Euclidean terkecil. Saat oklusi terjadi, trek yang hilang tidak memiliki prediksi pergerakan, dan saat muncul kembali, asosiasi serakah (*greedy*) mengaitkan centroid terdekat tanpa memperhitungkan kontinuitas vektor kecepatan.
     - *Solusi Rekayasa*:
       1. Integrasikan **Kalman Filter** untuk memodelkan keadaan kinematika linier (posisi dan kecepatan $[x, y, v_x, v_y]$) setiap trek. Saat oklusi terjadi, Kalman Filter memprediksi posisi masa depan berdasarkan vektor momentum sebelumnya.
       2. Terapkan algoritma penugasan global optimal (**Hungarian Algorithm**) pada matriks jarak gabungan spasial dan kesamaan histogram warna (Bhattacharyya distance) dari ROI sebelum oklusi.
       3. Naikkan batas `max_disappeared` agar trek tidak tergesa-gesa dideregistrasi selama jendela oklusi berlangsung.

---

## 4. Mini Project Chapter 26

### Judul Tugas Proyek:
**"Integrated Computer Vision Pipeline for Object Monitoring"**

### Deskripsi Tugas:
Rancang, bangun, dan evaluasi pipeline pemrosesan citra dan video mandiri yang mengintegrasikan seluruh konsep Computer Vision tingkat lanjut yang dipelajari pada bab ini. Sistem harus mampu menerima klip video pemantauan (sintetis atau rekaman terbuka berlisensi CC/MIT) dan mengeksekusi alur kerja terstruktur:
1. **Pra-Pemrosesan Citra & Peningkatan Kualitas**: Melakukan normalisasi pencahayaan, reduksi derau spasial dengan filter yang tepat, dan konversi ruang warna adaptif.
2. **Segmentasi & Deteksi Wilayah Objek**: Mengisolasi wilayah objek minat (*foreground objects*) menggunakan kombinasi ambang batas adaptif/Otsu, operasi morfologi pembersihan (*opening/closing*), dan ekstraksi kontur/komponen terhubung berkarakteristik bounding box serta titik berat (*centroid*).
3. **Pelacakan Spasio-Temporal Multi-Objek**: Mengimplementasikan pelacak objek berbasis centroid dengan pelestarian ID trek stabil dan penanganan oklusi sementara.
4. **Logika Bisnis & Analitik Peristiwa**: Menerapkan modul penghitung perlintasan garis maya (*virtual tripwire counting*) dua arah (IN / OUT) dengan mekanisme pencegahan penghitungan ganda (*double-counting prevention*).
5. **Visualisasi Diagnostik & Telemetri**: Mengembangkan overlay visual informatif yang menampilkan bounding box, label trek, garis tripwire, jejak lintasan historis (*trajectory tail*), serta panel statistik waktu nyata.
6. **Evaluasi Kinerja & Tanggung Jawab Komputasi**: Mengukur latensi rata-rata per frame (ms), throughput pemrosesan (FPS) pada CPU, analisis batas kegagalan sistem pada kondisi degradasi visual, serta dokumentasi privasi citra (Responsible AI).

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Problem Formulation & Video Ingestion** | 15 Poin | Perumusan kasus pemantauan objek spesifik dan relevan, verifikasi sumber video dan lisensi etis, pemuatan deret frame video mulus dengan penanganan galat kokoh, serta pemahaman struktur data tensor visual. |
| **2. Image Preprocessing & Feature Extraction** | 20 Poin | Pemilihan filter reduksi derau yang tepat dengan bukti empiris PSNR, konversi ruang warna yang optimal, penerapan operasi morfologi yang efektif membersihkan derau dan menutup lubang biner. |
| **3. Segmentation & Detection Logic** | 20 Poin | Implementasi segmentasi biner adaptif/Otsu yang presisi, ekstraksi komponen terhubung yang akurat menghasilkan bounding box dan centroid, serta validasi metrik Mask IoU dan Dice Score. |
| **4. Multi-Object Tracking & Analytics Engine** | 20 Poin | Pelacak objek berbasis centroid yang stabil mempertahankan ID unik antar-frame, penanganan toleransi objek hilang sementara (*disappeared frames*), dan penghitungan garis maya (*tripwire*) akurat bebas dari galat duplikasi (*double counting*). |
| **5. Diagnostics, Profiling & Ethical Deployment** | 15 Poin | Profiling latensi CPU terperinci per modul pemrosesan, visualisasi overlay profesional dan terbaca jelas, serta analisis batas kegagalan sistem dan kepatuhan privasi (Responsible AI). |
| **6. Code Quality, Modular Architecture & References** | 10 Poin | Kode terstruktur rapi ke dalam kelas-kelas modular tanpa duplikasi (*clean code*), dokumentasi instruksi replikasi yang lengkap, dan penyertaan 5 referensi akademik primer terverifikasi. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 11 skrip Python di direktori `code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Representasi memori citra OpenCV (`BGR`) dan penanganan galat pemuatan berkas tervalidasi.
- [ ] Konversi Grayscale terbukti mematuhi formula bobot fisiologis luminansi ITU-R BT.601 ($Y=150$ untuk hijau murni).
- [ ] Efektivitas Median Filter atas derau impulsif terbukti melampaui Gaussian Filter dengan selisih PSNR $> 8\text{ dB}$.
- [ ] Detektor tepi Canny menghasilkan kontur tepi ramping satu piksel via penekanan non-maksimum.
- [ ] Operasi Opening dan Closing terbukti secara komputasional membersihkan derau luar dan menutup celah internal.
- [ ] Segmentasi Otsu dan ekstraksi komponen terhubung berhasil memisahkan objek dan menghitung centroid analitis.
- [ ] Hubungan matematis $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$ terverifikasi numerik identik.
- [ ] Pelacak Centroid mempertahankan kontinuitas ID trek melintasi deret frame video sintetis.
- [ ] Penghitung garis maya (*virtual tripwire*) berhasil menghitung perlintasan arah tanpa insiden duplikasi.
- [ ] Pipeline modular terintegrasi mencatat throughput $> 30\text{ FPS}$ pada perangkat CPU standar.
- [ ] Berkas log hasil eksperimen `experiments/results_ch26.json` terisi dengan metrik valid.
- [ ] Notebook pendamping `chapter_26_companion.ipynb` tervalidasi bersih dalam format v4.

