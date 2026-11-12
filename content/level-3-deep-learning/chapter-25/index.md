# Bab 25: Object Detection & YOLO

## Prinsip Lokalisasi Spasial, Intersection over Union, Non-Maximum Suppression, dan Arsitektur Deteksi Objek Real-Time YOLO

---

## 1. Fondasi Deteksi Objek: Dari Klasifikasi Global ke Pemahaman Spasial

Pada Bab 24, kita telah mempelajari dan mengimplementasikan jaringan saraf konvolusional (*Convolutional Neural Networks* / CNN) untuk tugas klasifikasi citra (*Image Classification*). Pada klasifikasi citra standar, model menerima sebuah citra masukan dan menghasilkan satu label kategori global tunggal untuk keseluruhan gambar (misalnya membedakan antara *T-shirt*, *Trouser*, atau *Ankle boot*).

Namun, skenario dunia nyata dalam visi komputer (*Computer Vision*) menuntut sistem kecerdasan buatan untuk memahami lingkungan visual dengan resolusi spasial yang jauh lebih detail:
1. Citra nyata jarang sekali hanya memuat satu objek terisolasi di bagian tengah kanvas.
2. Citra biasanya memuat banyak entitas fisik berbeda yang saling berinteraksi (misalnya beberapa mobil, pejalan kaki, pesepeda, dan rambu lalu lintas pada satu adegan jalan raya).
3. Agen cerdas (seperti mobil otonom atau robot industri) tidak hanya perlu mengetahui *apa* objek yang tampak, melainkan *di mana* koordinat pasti objek tersebut berada agar dapat merencanakan manuver kemudi atau gerakan lengan robotik secara aman.

```
+---------------------------------------------------------------------------------------------------------+
|                         SPEKTRUM TUGAS PENGENALAN VISUAL DALAM COMPUTER VISION                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. IMAGE CLASSIFICATION             2. OBJECT LOCALIZATION              3. OBJECT DETECTION           |
|      +---------------------+             +---------------------+             +---------------------+    |
|      |                     |             |     +---------+     |             | +-----+   +-------+ |    |
|      |        Kucing       |             |     | Kucing  |     |             | |Kucing|  | Anjing| |    |
|      |                     |             |     +---------+     |             | +-----+   +-------+ |    |
|      |                     |             |                     |             |        +----+       |    |
|      +---------------------+             +---------------------+             |        |Bebek|      |    |
|      Output: Label Tunggal               Output: Label Tunggal +             +---------------------+    |
|      (Kucing)                            1 Bounding Box                      Output: N Pasangan         |
|                                                                              (Label + Bounding Box)     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Taksonomi Tugas Pemahaman Visual:
1. **Image Classification**: Mengidentifikasi kategori objek dominan dalam citra tanpa menyediakan informasi lokasi spasial.
2. **Object Localization**: Mengidentifikasi kategori dari **satu objek tunggal** sekaligus menentukan lokasi spasialnya melalui satu kotak pembatas (*bounding box*).
3. **Object Detection**: Mengidentifikasi lokasi dan kategori dari **sejumlah objek arbitrer ($N$ entitas)** yang tersebar di sembarang posisi pada citra secara simultan.

---

## 2. Representasi Geometris Bounding Box & Format Koordinat

Bounding Box adalah persegi panjang terkecil yang berorientasi sejajar sumbu kartesian (*axis-aligned bounding box*) yang melingkupi seluruh batas piksel tampak dari suatu objek fisik. Di dunia rekayasa kecerdasan buatan, terdapat dua format koordinat utama yang digunakan:

```
+---------------------------------------------------------------------------------------------------------+
|                                FORMAT REPRESENTASI BOUNDING BOX 2D                                      |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   FORMAT PASCAL VOC (xyxy)                           FORMAT YOLO / COCO (xywh)                          |
|   (Titik Sudut Kiri-Atas & Kanan-Bawah)              (Titik Pusat Spasial, Lebar, dan Tinggi)           |
|                                                                                                         |
|   (x1, y1)                                                                                              |
|      +──────────────────────────+                       +──────────────────────────+                    |
|      │                          │                       │                          │                    |
|      │                          │                       │         (xc, yc)         │  Tinggi (h)        |
|      │                          │                       │            *             │                    |
|      │                          │                       │                          │                    |
|      +──────────────────────────+                       +──────────────────────────+                    |
|                               (x2, y2)                               Lebar (w)                          |
|                                                                                                         |
|   Koordinat: [x1, y1, x2, y2]                        Koordinat: [xc, yc, w, h]                          |
|   x1: absis sudut kiri-atas                          xc = x1 + w / 2 (absis titik pusat)                |
|   y1: ordinat sudut kiri-atas                        yc = y1 + h / 2 (ordinat titik pusat)              |
|   x2: absis sudut kanan-bawah                        w  = x2 - x1    (lebar kotak)                      |
|   y2: ordinat sudut kanan-bawah                      h  = y2 - y1    (tinggi kotak)                     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Persamaan Konversi Dua Arah (Bi-directional Conversion)
Transformasi antara format PASCAL VOC $[x_1, y_1, x_2, y_2]$ dan format pusat $[x_c, y_c, w, h]$ dirumuskan secara analitis:

$$
\begin{aligned}
w &= x_2 - x_1, & \quad h &= y_2 - y_1 \\
x_c &= x_1 + \frac{w}{2}, & \quad y_c &= y_1 + \frac{h}{2}
\end{aligned}
$$

Sebaliknya, untuk merekonstruksi format sudut dari format pusat:

$$
\begin{aligned}
x_1 &= x_c - \frac{w}{2}, & \quad y_1 &= y_c - \frac{h}{2} \\
x_2 &= x_c + \frac{w}{2}, & \quad y_2 &= y_c + \frac{h}{2}
\end{aligned}
$$

### Koordinat Absolut Piksel vs Koordinat Ternormalisasi
Dalam berkas anotasi citra mentah, koordinat sering kali dinyatakan dalam piksel absolut integer (misalnya $x_1 = 120, y_1 = 80$ pada citra berdimensi $640 \times 480$). Namun, pada arsitektur modern seperti YOLO, seluruh koordinat **wajib dinormalisasi** ke dalam rentang mengambang $[0.0, 1.0]$ terhadap lebar ($W_{\text{img}}$) dan tinggi ($H_{\text{img}}$) citra:

$$
x_{c,\text{norm}} = \frac{x_c}{W_{\text{img}}}, \quad y_{c,\text{norm}} = \frac{y_c}{H_{\text{img}}}, \quad w_{\text{norm}} = \frac{w}{W_{\text{img}}}, \quad h_{\text{norm}} = \frac{h}{H_{\text{img}}}
$$

Keuntungan normalisasi koordinat:
1. **Invariansi Skala**: Anotasi tetap valid meskipun citra diubah ukurannya (*resized*) selama proses pelatihan model.
2. **Stabilitas Optimasi Gradien**: Menjaga target regresi bounding box berada dalam skala numerik kecil yang cocok untuk fungsi aktivasi dan algoritma optimizer gradien (seperti Adam).

---

## 3. Matematika Intersection over Union (IoU)

Bagaimana kita mengukur seberapa akurat sebuah kotak prediksi model terhadap posisi objek sebenarnya (*ground-truth*)? Ukuran standar dalam visi komputer adalah **Intersection over Union (IoU)**, atau dikenal dalam literatur statistik sebagai **Koefisien Jaccard**.

Diberikan dua bounding box: kotak prediksi $B_p$ dan kotak acuan ground-truth $B_g$, nilai IoU didefinisikan sebagai rasio antara luas area irisan (*intersection*) dibagi luas area gabungan (*union*):

$$
\text{IoU}(B_p, B_g) = \frac{\text{Area}(B_p \cap B_g)}{\text{Area}(B_p \cup B_g)} = \frac{\text{Area}(B_p \cap B_g)}{\text{Area}(B_p) + \text{Area}(B_g) - \text{Area}(B_p \cap B_g)}
$$

```
+---------------------------------------------------------------------------------------------------------+
|                                GEOMETRI INTERSECTION OVER UNION (IoU)                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Kotak Prediksi (Bp)           Kotak Ground-Truth (Bg)                  Irisan (Intersection)      |
|       +-------------------+         . . . . . . . . . . . . .               +-------+                   |
|       │                   │         .                       .               │///////│                   |
|       │         +─────────┼─────────.─────+                 .               +-------+                   |
|       │         │/////////│/////////│     .                 .                                           |
|       +─────────┼─────────+         .     .                 .               Gabungan (Union)            |
|                 │                   .     .                 .               +-------------------+       |
|                 + . . . . . . . . . . . . +                 .               │                   │       |
|                                                             .               │                   │       |
|                                                                             │         +─────────┤       |
|                                                                             │         │         │       |
|                                                                             +─────────┴─────────+       |
|                                                                                                         |
|                                          Area(Bp ∩ Bg)                                                  |
|                        IoU  =  ─────────────────────────────────                                        |
|                                Area(Bp) + Area(Bg) - Area(Bp ∩ Bg)                                      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Prosedur Komputasi Luas Irisan (Intersection Area)
Misalkan kotak $B_a = [x_{1a}, y_{1a}, x_{2a}, y_{2a}]$ dan $B_b = [x_{1b}, y_{1b}, x_{2b}, y_{2b}]$.
Koordinat sudut persegi panjang irisan adalah:

$$
\begin{aligned}
x_{1,\text{inter}} &= \max(x_{1a}, x_{1b}) \\
y_{1,\text{inter}} &= \max(y_{1a}, y_{1b}) \\
x_{2,\text{inter}} &= \min(x_{2a}, x_{2b}) \\
y_{2,\text{inter}} &= \min(y_{2a}, y_{2b})
\end{aligned}
$$

Lebar ($w_{\text{inter}}$) dan tinggi ($h_{\text{inter}}$) irisan bernilai:

$$
w_{\text{inter}} = \max(0.0, x_{2,\text{inter}} - x_{1,\text{inter}}), \quad h_{\text{inter}} = \max(0.0, y_{2,\text{inter}} - y_{1,\text{inter}})
$$

Fungsi $\max(0.0, \cdot)$ memastikan bahwa jika kedua kotak terpisah total dan tidak beririsan ($x_{2,\text{inter}} < x_{1,\text{inter}}$), luas irisan otomatis bernilai $0.0$.

### Contoh Perhitungan Manual (Worked Example)
Diberikan dua kotak berdimensi $200 \times 200$ piksel:
- Kotak A: $[100, 100, 300, 300] \implies \text{Luas}_A = 200 \times 200 = 40.000$
- Kotak B: $[200, 200, 400, 400] \implies \text{Luas}_B = 200 \times 200 = 40.000$

Perhitungan irisan:
- $x_{1,\text{inter}} = \max(100, 200) = 200$, $y_{1,\text{inter}} = \max(100, 200) = 200$
- $x_{2,\text{inter}} = \min(300, 400) = 300$, $y_{2,\text{inter}} = \min(300, 400) = 300$
- $w_{\text{inter}} = 300 - 200 = 100$, $h_{\text{inter}} = 300 - 200 = 100$
- $\text{Area}_{\text{inter}} = 100 \times 100 = 10.000$

Perhitungan gabungan (Union):
- $\text{Area}_{\text{union}} = 40.000 + 40.000 - 10.000 = 70.000$

Nilai rasio IoU:
$$
\text{IoU} = \frac{10.000}{70.000} = \frac{1}{7} \approx \mathbf{0.142857}
$$

---

## 4. Algoritma Non-Maximum Suppression (NMS)

Ketika detektor objek mengevaluasi ribuan grid sel di seluruh citra secara simultan, satu objek fisik (misalnya sebuah mobil) hampir selalu memicu **banyak bounding box kandidat** yang saling tumpang tindih dengan skor keyakinan (*confidence score*) yang sedikit berbeda.

Untuk mengeliminasi kotak duplikat redundan ini, kita menggunakan algoritma **Non-Maximum Suppression (NMS)**:

```
+---------------------------------------------------------------------------------------------------------+
|                                ALUR KERJA NON-MAXIMUM SUPPRESSION (NMS)                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. FILTER AMBANG BATAS KEYAKINAN (CONFIDENCE THRESHOLDING)                                            |
|      Singkirkan seluruh prediksi yang memiliki skor keyakinan di bawah ambang batas (misal skor < 0.25).|
|                                                                                                         |
|   2. PERANGKINGAN SKOR (CONFIDENCE RANKING)                                                             |
|      Urutkan seluruh kotak kandidat tersisa secara menurun berdasarkan skor keyakinannya.               |
|                                                                                                         |
|   3. SUPRESI ITERATIF BERBASIS IoU (ITERATIVE SUPPRESSION)                                              |
|      Ambil kotak teratas (skor tertinggi) dan masukkan ke daftar "Kotak Terpilih".                      |
|      Hitung nilai IoU antara kotak terpilih ini dengan seluruh kotak kandidat lainnya:                  |
|      - Jika IoU > ambang_batas_NMS (misal > 0.50), kotak kandidat tersebut DIHAPUS (karena dianggap     |
|        mendeteksi objek fisik yang sama).                                                               |
|      - Jika IoU <= ambang_batas_NMS, kotak kandidat tersebut DIPERTAHANKAN untuk iterasi berikutnya.     |
|      Ulangi langkah 3 hingga seluruh kandidat selesai dievaluasi.                                       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Class-Agnostic vs Class-Aware NMS
- **Class-Agnostic NMS**: Algoritma NMS dijalankan secara global tanpa mempedulikan label kelas prediksi. Berbahaya jika dua objek dari kategori berbeda saling bertumpukan erat (misalnya seorang pengendara sepeda motor bertumpukan dengan sepeda motornya).
- **Class-Aware NMS**: Algoritma NMS dijalankan secara independen untuk setiap kategori kelas objek secara terpisah. Kotak mobil tidak akan meniadakan kotak pejalan kaki meskipun keduanya memiliki nilai IoU yang tinggi.

---

## 5. Protokol Evaluasi Deteksi Objek: Precision, Recall, AP, dan mAP

Dalam klasifikasi citra sederhana, kita mengevaluasi model menggunakan akurasi biner. Namun dalam deteksi objek, penentuan benar (*True Positive*) atau salah (*False Positive*) membutuhkan aturan pencocokan spasial eksplisit berbasis IoU.

### Prosedur Pencocokan Prediksi ke Ground-Truth (Greedy Matching)
1. Urutkan seluruh prediksi model berdasarkan skor keyakinan dari yang tertinggi ke terendah.
2. Untuk setiap prediksi, cari objek Ground-Truth (GT) dari kelas yang sama yang memiliki nilai IoU tertinggi.
3. Evaluasi kondisi:
   - Jika $\text{IoU} \ge \text{ambang batas}$ (standar: 0.50) **DAN** objek GT tersebut belum pernah dicocokkan oleh prediksi sebelumnya, maka prediksi divonis sebagai **True Positive (TP)**, dan objek GT tersebut ditandai telah terkunci (*matched*).
   - Jika $\text{IoU} < \text{ambang batas}$ ATAU objek GT tersebut sudah pernah dicocokkan oleh prediksi lain yang berskor lebih tinggi, maka prediksi divonis sebagai **False Positive (FP)** (prediksi duplikat atau salah tempat).
4. Setiap objek Ground-Truth yang tidak berhasil dicocokkan oleh prediksi mana pun hingga akhir proses divonis sebagai **False Negative (FN)**.

```
+---------------------------------------------------------------------------------------------------------+
|                                  TAKSONOMI HASIL PREDIKSI DETEKSI OBJEK                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   TRUE POSITIVE (TP)             FALSE POSITIVE (FP)                   FALSE NEGATIVE (FN)              |
|   Kotak prediksi mencakup        Kotak prediksi salah lokasi,           Objek fisik nyata pada citra    |
|   objek ground-truth secara      mendeteksi latar belakang kosong,      terlewatkan total dan tidak     |
|   akurat (IoU >= 0.50).          atau merupakan deteksi duplikat.       terdeteksi oleh model.          |
|                                                                                                         |
|   +-----------------------+      +-----------------------+             +-----------------------+        |
|   │ GT: [ Mobil ]         │      │ GT: [ Kosong / Pohon] │             │ GT: [ Pejalan Kaki ]  │        |
|   │ Pred: [ Mobil ]       │      │ Pred: [ Mobil ]       │             │ Pred: (Tidak ada!)    │        |
|   │ IoU: 0.88 (Cocok!)    │      │ IoU: 0.05 (Halusinasi)│             │                       │        |
|   +-----------------------+      +-----------------------+             +-----------------------+        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Perhitungan Presisi, Recall, dan Average Precision (AP)
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Seiring diturunkannya ambang batas skor keyakinan, kita memperoleh pasangan nilai $(\text{Recall}, \text{Precision})$ yang membentuk **Kurva Precision-Recall (PR Curve)**.

**Average Precision (AP)** dihitung sebagai luas area di bawah kurva PR setelah dilakukan interpolasi monotonik:

$$
\text{AP} = \sum_{k=1}^N (R_k - R_{k-1}) \cdot P_{\text{interp}}(R_k) \quad \text{dengan} \quad P_{\text{interp}}(r) = \max_{r' \ge r} P(r')
$$

### Perbedaan Krusial: mAP@0.5 vs mAP@0.5:0.95
- **mAP@0.5 (Standar PASCAL VOC)**: Rata-rata nilai AP di seluruh kelas objek pada satu ambang batas IoU tetap: $\text{IoU} = 0.50$. Metrik ini mengevaluasi apakah model berhasil menemukan lokasi kasar objek.
- **mAP@0.5:0.95 (Standar Resmi COCO)**: Rata-rata dari nilai mAP pada 10 ambang batas IoU berbeda: $\{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95\}$. Metrik ini memberikan penalti berat jika batas tepi kotak pembatas kurang pas atau longgar, sehingga menjadi tolok ukur ketelitian spasial internasional.

---

## 6. Arsitektur YOLO: You Only Look Once

Sebelum kemunculan YOLO, sistem deteksi objek didominasi oleh arsitektur **Two-Stage** (seperti R-CNN dan Faster R-CNN):
1. **Tahap 1**: Region Proposal Network (RPN) menghasilkan sekitar 2.000 usulan kotak wilayah (*region proposals*).
2. **Tahap 2**: Setiap kotak dipotong, diubah ukurannya, dan diproses oleh CNN klasifikasi.

Meskipun akurat, sistem two-stage sangat lambat (biasanya hanya 5–15 FPS) dan tidak dapat digunakan untuk aplikasi real-time kritis.

**YOLO (You Only Look Once)**, diperkenalkan oleh Joseph Redmon et al. (2016), merombak paradigma tersebut menjadi pendekatan **One-Stage Detection**:
- Citra masukan hanya diproses **satu kali** melalui jaringan konvolusional end-to-end.
- Lapisan terakhir jaringan langsung memprediksi probabilitas kelas dan koordinat regresi bounding box secara serentak untuk seluruh area citra.

```
+---------------------------------------------------------------------------------------------------------+
|                                ARSITEKTUR DETEKTOR ONE-STAGE YOLO MODERN                                |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [ Citra Input: 640x640x3 ]                                                                            |
|               │                                                                                         |
|               ▼                                                                                         |
|   ┌───────────────────────┐                                                                             |
|   │       BACKBONE        │  Ekstraksi fitur berhierarki melalui blok konvolusi CSP/C2f                 |
|   │ (CSPDarknet / C2f)    │  Menghasilkan peta fitur pada 3 skala downsampling:                         |
|   └───────────┬───────────┘  P3 (Skala 1/8: 80x80), P4 (Skala 1/16: 40x40), P5 (Skala 1/32: 20x20)      |
|               │                                                                                         |
|               ▼                                                                                         |
|   ┌───────────────────────┐                                                                             |
|   │         NECK          │  Agregasi piramida fitur (Feature Pyramid Network + Path Aggregation)       |
|   │   (PANet / BiFPN)     │  Menggabungkan konteks semantik tingkat tinggi dengan resolusi spasial      |
|   └───────────┬───────────┘  tingkat rendah melalui jalur top-down dan bottom-up ganda.                 |
|               │                                                                                         |
|               ▼                                                                                         |
|   ┌───────────────────────┐                                                                             |
|   │    DETECTION HEAD     │  Decoupled Head: Memisahkan cabang klasifikasi dan cabang regresi kotak     |
|   │  (Anchor-Free Head)   │  Prediksi multi-skala:                                                      |
|   └───────────┬───────────┘  - Skala Kecil (80x80) -> Mendeteksi objek kecil                             |
|               │              - Skala Sedang (40x40) -> Mendeteksi objek sedang                           |
|               ▼              - Skala Besar (20x20) -> Mendeteksi objek besar                             |
|                                                                                                         |
|   [ Pascapemrosesan: Confidence Thresholding + Non-Maximum Suppression ] ──> [ BBox + Label Final ]     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Konsep Anchor-Based vs Anchor-Free
- **Anchor-Based (YOLOv2 hingga YOLOv5)**: Menggunakan kotak acuan awal (*prior anchor boxes*) dengan rasio aspek tetap (misal $1:1, 1:2, 2:1$). Model memprediksi offset deviasi $(\Delta x, \Delta y, \Delta w, \Delta h)$ terhadap anchor terdekat.
- **Anchor-Free (YOLOv8 dan Arsitektur Modern)**: Model langsung memprediksi jarak dari titik pusat sel grid ke empat batas tepi kotak (*top, bottom, left, right*). Pendekatan ini mengeliminasi proses penentuan anchor box manual yang rumit dan meningkatkan akurasi deteksi pada objek berbentuk tak lazim.

---

## 7. Format Anotasi Dataset YOLO & Manajemen Partisi Bebas Leakage

Dalam ekosistem YOLO, setiap berkas citra `nama_file.jpg` dipasangkan dengan satu berkas teks anotasi berekstensi `.txt` dengan nama yang sama persis: `nama_file.txt`.

Setiap baris di dalam berkas teks tersebut merepresentasikan **satu objek fisik**:
```text
<class_id> <x_center> <y_center> <width> <height>
```

```
+---------------------------------------------------------------------------------------------------------+
|                                STRUKTUR BERKAS ANOTASI DATASET STANDAR YOLO                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   dataset_root/                                                                                         |
|   ├── data.yaml              <── Berkas konfigurasi: path dataset, jumlah kelas, daftar nama kelas     |
|   ├── images/                                                                                           |
|   │   ├── train/             <── Citra partisi latih (70%)                                              |
|   │   ├── val/               <── Citra partisi validasi (15%)                                           |
|   │   └── test/              <── Citra partisi uji terisolasi (15%)                                     |
|   └── labels/                                                                                           |
|       ├── train/             <── Berkas .txt label latih (format ternormalisasi [0.0, 1.0])             |
|       ├── val/               <── Berkas .txt label validasi                                             |
|       └── test/              <── Berkas .txt label uji                                                  |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Protokol Pencegahan Kebocoran Data (Zero Data Leakage)
1. **Pemisahan Tingkat Sesi / Rekaman**: Jika dataset dikumpulkan dari rekaman video CCTV, frame yang berasal dari rekaman waktu yang sama **wajib berada pada partisi yang sama**. Memasukkan frame detik ke-1 ke train dan frame detik ke-2 ke test set merupakan pelanggaran kebocoran data (*data leakage*) temporal parah.
2. **Isolasi Augmentasi**: Transformasi augmentasi citra agresif (seperti Mosaic, MixUp, Random Perspective) **hanya boleh diaktifkan pada partisi Train**. Partisi Val dan Test harus mempertahankan karakteristik distribusi dunia nyata asli tanpa manipulasi geometris buatan.

---

## 8. Inferensi Model pada CPU & Analisis Kompromi Latensi vs Akurasi

Untuk penggunaan praktis di perangkat komputasi tepi (*Edge AI*) atau komputer jinjing tanpa akselerator GPU eksternal, varian model ringan seperti **YOLOv8 Nano (`yolov8n.pt`)** merupakan pilihan standar industri:

```
+---------------------------------------------------------------------------------------------------------+
|                               KOMPARASI KELUARGA MODEL YOLOV8 PADA CPU STANDAR                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Varian Model    Jumlah Parameter    Beban FLOPs    Latensi CPU (ms)    Target Penggunaan              |
|   ───────────────────────────────────────────────────────────────────────────────────────────────────   |
|   YOLOv8n (Nano)  3,2 Juta Bobot       8,7 GFLOPs     ~15 - 30 ms        Edge IoT, CPU Laptop, Drone    |
|   YOLOv8s (Small) 11,2 Juta Bobot     28,6 GFLOPs     ~40 - 70 ms        SBC (Raspberry Pi 5), Edge Mini|
|   YOLOv8m (Medium)25,9 Juta Bobot     78,9 GFLOPs     ~120 - 200 ms      Server Lokal, GPU Ringan       |
|   YOLOv8x (Extra) 68,2 Juta Bobot    257,8 GFLOPs     ~400 - 800 ms      Server Cloud Batch Terpusat    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Hukum Kompromi Real-Time
Kecepatan video standar adalah 30 FPS, yang berarti sistem hanya memiliki anggaran waktu maksimum **33.3 milidetik** per frame.
Total waktu latensi sistem dihitung dari penjumlahan tiga tahap komputasi:

$$
T_{\text{total}} = T_{\text{preprocess}} + T_{\text{inference}} + T_{\text{postprocess (NMS)}} \le 33.3 \text{ ms}
$$

Jika $T_{\text{total}} > 33.3 \text{ ms}$, sistem akan mengalami fenomena keterlambatan bingkai (*frame lagging*) atau penumpukan antrean memori RAM.

---

## 9. Tinjauan Fine-Tuning & Transfer Learning pada Dataset Kustom

Melatih model deteksi objek dari awal (*from scratch*) dengan inisialisasi bobot acak membutuhkan puluhan ribu citra beranotasi dan ratusan jam komputasi akselerator GPU. Sebagai alternatif praktis, teknik **Fine-Tuning berbasis Transfer Learning** memungkinkan adaptasi domain secara cepat:
1. **Pretrained Weights**: Menggunakan bobot model yang telah dilatih sebelumnya pada dataset raksasa (seperti Microsoft COCO dengan 80 kelas dan 1,5 juta objek).
2. **Feature Reuse**: Lapisan awal Backbone telah belajar mengenali detektor visual dasar (tepi, gradien, tekstur lingkaran) yang berlaku universal untuk semua domain citra.
3. **Head Adaptation**: Kita hanya perlu menyesuaikan lapisan akhir Detection Head agar memetakan fitur ke kelas domain spesifik kita (misalnya mendeteksi komponen cacat pada jalur perakitan pabrik) dengan menggunakan beberapa ratus citra saja.

---

## 10. Etika, Privasi, dan Penerapan Bertanggung Jawab (Responsible Computer Vision)

Penerapan teknologi deteksi objek di ruang publik mengharuskan para insinyur mematuhi pedoman etika kecerdasan buatan (mengacu pada Bab 09):
- **Privasi Individu & Regulasi PDP**: Deteksi objek pada fasilitas umum wajib menerapkan penyamaran otomatis (*automatic blurring*) pada wajah pejalan kaki dan plat nomor kendaraan guna melindungi identitas pribadi.
- **Bias Representasi Geografis**: Dataset terbuka global umumnya didominasi oleh citra dari negara-negara maju (kondisi jalan bersih, rambu standar barat, cuaca cerah). Insinyur wajib melakukan validasi lapangan lokal (*field testing*) untuk memastikan keandalan model pada kondisi jalanan lokal Indonesia (gang sempit, cuaca hujan tropis lebat, kepadatan pedagang kaki lima).
- **Pengawasan Manusia (Human-in-the-Loop)**: Deteksi visual pada sistem berisiko tinggi (seperti sistem keamanan pabrik atau pengereman darurat) tidak boleh beroperasi secara tertutup sepenuhnya tanpa mekanisme verifikasi dan intervensi keselamatan manual manusia.

---

## 11. Latihan Mandiri 3-Tingkat Lengkap dengan Solusi

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Konversi Format Bounding Box**: Diberikan sebuah bounding box pada citra $1000 \times 1000$ piksel dengan format YOLO ternormalisasi: `class_id = 2, xc = 0.40, yc = 0.60, w = 0.20, h = 0.30`. Tuliskan koordinat absolut piksel dalam format Pascal VOC `[x1, y1, x2, y2]`!
   - *Jawaban*:
     - Dimensi absolut: $w_{\text{abs}} = 0.20 \times 1000 = 200$, $h_{\text{abs}} = 0.30 \times 1000 = 300$.
     - Titik pusat absolut: $x_{c,\text{abs}} = 0.40 \times 1000 = 400$, $y_{c,\text{abs}} = 0.60 \times 1000 = 600$.
     - Sudut sudut:
       $$x_1 = 400 - 200/2 = 300, \quad y_1 = 600 - 300/2 = 450$$
       $$x_2 = 400 + 200/2 = 500, \quad y_2 = 600 + 300/2 = 750$$
     - Hasil format VOC: `[300, 450, 500, 750]`.

2. **Perhitungan IoU Kasus Nyata**: Dua kotak prediksi saling berpotongan:
   - Kotak 1: $[50, 50, 150, 150] \implies \text{Luas} = 100 \times 100 = 10.000$
   - Kotak 2: $[100, 50, 200, 150] \implies \text{Luas} = 100 \times 100 = 10.000$
   Hitung luas irisan, luas gabungan, dan rasio IoU!
   - *Jawaban*:
     - Irisan: $[100, 50, 150, 150] \implies \text{Lebar} = 50, \text{Tinggi} = 100 \implies \text{Luas Irisan} = 5.000$.
     - Luas Gabungan: $10.000 + 10.000 - 5.000 = 15.000$.
     - $\text{IoU} = 5.000 / 15.000 = 1/3 \approx \mathbf{0.3333}$.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Analisis Duplikasi Deteksi pada NMS**: Mengapa algoritma Non-Maximum Suppression (NMS) berbasis skor keyakinan gagal mempertahankan objek kedua yang berada persis di depan objek pertama (misalnya seseorang berdiri tepat di depan pintu mobil) jika menggunakan NMS Class-Agnostic?
   - *Jawaban*:
     - Pada NMS Class-Agnostic, algoritma tidak membedakan label kelas antara orang dan mobil. Karena kotak orang berada di dalam atau tumpang tindih tinggi (misal $\text{IoU} > 0.50$) terhadap kotak mobil yang berskor lebih tinggi, kotak orang tersebut dianggap sebagai deteksi duplikat redundan dari mobil dan dihapus secara keliru.
     - Mitigasi yang tepat adalah menerapkan **Class-Aware NMS**, di mana penapisan IoU hanya dijalankan di antara sesama kotak berkategori kelas identik.

2. **Mendiagnosis Masalah Anotasi Bocor**: Seorang praktikan mendapati bahwa model deteksinya sering memprediksi kotak kosong pada area aspal jalan. Setelah diaudit, ditemukan bahwa pada 30 citra latihan, terdapat mobil-mobil kecil di kejauhan yang tidak diberi label bounding box oleh annotator karena dianggap tidak penting. Jelaskan dampak teoritis dari kelalaian anotasi ini!
   - *Jawaban*:
     - Dalam paradigma deteksi objek terbimbing, setiap area citra yang tidak tercakup oleh bounding box ground-truth dianggap oleh fungsi rugi (*loss function*) sebagai **latar belakang murni (background)**.
     - Kelalaian memberi label pada mobil di kejauhan secara aktif menghukum model jika model mendeteksinya (*false penalization*). Akibatnya, model mengalami disorientasi gradien dan belajar menganggap tekstur mobil kecil sebagai pola aspal latar belakang.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Perhitungan Metrik Average Precision (AP)**: Suatu model deteksi dievaluasi pada dataset yang memuat 4 objek Ground Truth. Model mengeluarkan 5 prediksi (diurutkan dari skor tertinggi):
   - Pred 0: TP
   - Pred 1: FP
   - Pred 2: TP
   - Pred 3: FP
   - Pred 4: TP
   Hitung nilai Precision dan Recall pada setiap langkah kumulatif, serta hitung nilai Average Precision (AP) menggunakan metode interpolasi all-points!
   - *Jawaban*:
     - Langkah 1 (Pred 0): $\text{TP}=1, \text{FP}=0 \implies \text{Precision} = 1/1 = 1.0000, \text{Recall} = 1/4 = 0.2500$
     - Langkah 2 (Pred 1): $\text{TP}=1, \text{FP}=1 \implies \text{Precision} = 1/2 = 0.5000, \text{Recall} = 1/4 = 0.2500$
     - Langkah 3 (Pred 2): $\text{TP}=2, \text{FP}=1 \implies \text{Precision} = 2/3 \approx 0.6667, \text{Recall} = 2/4 = 0.5000$
     - Langkah 4 (Pred 3): $\text{TP}=2, \text{FP}=2 \implies \text{Precision} = 2/4 = 0.5000, \text{Recall} = 2/4 = 0.5000$
     - Langkah 5 (Pred 4): $\text{TP}=3, \text{FP}=2 \implies \text{Precision} = 3/5 = 0.6000, \text{Recall} = 3/4 = 0.7500$
     - Interpolasi Monotonik Presisi:
       - Pada rentang $R \in [0.0, 0.25]$: $P_{\text{max}} = 1.0000$
       - Pada rentang $R \in (0.25, 0.50]$: $P_{\text{max}} = \max(0.6667, 0.6000) = 0.6667$
       - Pada rentang $R \in (0.50, 0.75]$: $P_{\text{max}} = 0.6000$
       - Pada rentang $R \in (0.75, 1.00]$: Objek ke-4 terlewat (FN) $\implies P_{\text{max}} = 0.0$
     - Luas Area di bawah kurva (AP):
       $$\text{AP} = (0.25 - 0.0) \times 1.0000 + (0.50 - 0.25) \times 0.6667 + (0.75 - 0.50) \times 0.6000 + (1.00 - 0.75) \times 0.0$$
       $$\text{AP} = 0.2500 + 0.1667 + 0.1500 + 0.0 = \mathbf{0.5667}$$

---

## 12. Panduan Mini Project, Rubrik Penilaian, dan Referensi Terverifikasi

### Mini Project Chapter 25: "Object Detection with YOLO on a Small Visual Dataset"
Sebagai tugas akhir Bab 25, Anda diminta merancang dan membangun pipeline deteksi objek mandiri menggunakan model **YOLO** pada dataset citra kecil yang terverifikasi (misalnya deteksi kendaraan dan pejalan kaki sintetis atau dataset terbuka berlisensi CC/MIT) pada jalur eksekusi CPU.

Panduan teknis langkah-demi-langkah, instruksi eksekusi kode, dan checklist penyerahan tersedia secara terperinci pada dokumen [practice.md](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-3-deep-learning/chapter-25/practice.md).

### Rubrik Penilaian Proyek (Total: 100 Poin)

| Komponen Evaluasi | Bobot | Kriteria Kinerja Maksimal |
|---|:---:|---|
| **1. Problem & Dataset Understanding** | 15 Poin | Perumusan masalah deteksi objek lugas, kejelasan sumber dataset dan lisensi penggunaan etis, kepatuhan struktur format anotasi YOLO, serta audit integritas dataset terbukti bebas kebocoran data. |
| **2. Detection Implementation** | 30 Poin | Implementasi pipeline inferensi YOLO pada CPU tereksekusi mulus tanpa galat, modul konversi koordinat (`xyxy` vs `xywh`) presisi, algoritma NMS terintegrasi tepat, dan visualisasi bounding box rapi serta mudah dibaca. |
| **3. Evaluation & Analysis** | 25 Poin | Evaluasi pencocokan prediksi ke ground truth berbasis IoU transparan, pelaporan metrik Precision, Recall, AP, dan mAP akurat tanpa fabrikasi angka, serta diseksi visual kesalahan deteksi (TP, FP, FN). |
| **4. Computational Considerations & Limitations** | 15 Poin | Pengukuran empiris waktu inferensi (ms) pada CPU, analisis trade-off ambang batas keyakinan (Confidence Threshold), dan pembahasan jujur mengenai batas keandalan model pada kondisi pencahayaan ekstrem atau oklusi. |
| **5. Documentation & References** | 15 Poin | Kode bersih terstruktur modular tanpa duplikasi, dokumentasi eksekusi reproducible, kepatuhan etika privasi citra, dan pencantuman 5 referensi akademik primer yang relevan. |

---

## 13. Daftar Referensi Terverifikasi

1. **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A.** (2016). *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), 779-788. [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
2. **Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L.** (2014). *Microsoft COCO: Common Objects in Context*. European Conference on Computer Vision (ECCV 2014), 740-755. [https://arxiv.org/abs/1405.0312](https://arxiv.org/abs/1405.0312)
3. **Everingham, M., Van Gool, L., Williams, C. K., Winn, J., & Zisserman, A.** (2010). *The Pascal Visual Object Classes (VOC) Challenge*. International Journal of Computer Vision (IJCV), 88(2), 303-338. [https://link.springer.com/article/10.1007/s11263-009-0275-4](https://link.springer.com/article/10.1007/s11263-009-0275-4)
4. **Jocher, G., Chaurasia, A., & Qiu, J.** (2023). *Ultralytics YOLOv8 Documentation and Software Repository*. GitHub. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
5. **IEEE Computer Society & ACM.** (2023). *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. Joint Task Force on Computing Curricula. [https://csed.acm.org](https://csed.acm.org)

