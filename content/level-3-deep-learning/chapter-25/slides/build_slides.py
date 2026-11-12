"""
Chapter 25 - Object Detection & YOLO
Script Pembangkit Slide Presentasi Reveal.js (deck.json)
Menghasilkan 36 slide terstruktur sesuai silabus Section E11.
"""

import json
from pathlib import Path

slides = [
    # 1. Cover
    {
        "kind": "cover",
        "title": "Object Detection & YOLO",
        "body": "### Bab 25: Deteksi Objek Spasial, Intersection over Union, Non-Maximum Suppression, dan Arsitektur Real-Time YOLO\n\n**Artificial Intelligence Learning Hub**  \n*From AI Fundamentals to Intelligent Systems*",
        "notes": "Selamat datang di perkuliahan Bab 25 mengenai Object Detection dan arsitektur YOLO. Hari ini kita melangkah dari klasifikasi citra global menuju lokalisasi spasial objek ganda secara simultan dan real-time."
    },
    # 2. Why Object Detection?
    {
        "kind": "content",
        "title": "Mengapa Deteksi Objek Sangat Krusial?",
        "body": "Klasifikasi citra standar hanya menjawab pertanyaan: *Objek apa yang ada pada gambar ini?* secara agregat.\n\nNamun sistem cerdas di dunia nyata menuntut pemahaman kontekstual yang jauh lebih kaya:\n\n- **Kendaraan Otonom**: Wajib mengetahui *di mana* pejalan kaki dan kendaraan lain berada untuk pengereman darurat.\n- **Sistem Pengawasan & Keamanan**: Melacak pergerakan intrusi pada koordinat area terlarang.\n- **Otomasi Industri**: Memandu lengan robotik untuk mencengkeram komponen produk cacat secara presisi spasial.",
        "notes": "Tekankan perbedaan fungsional antara mengetahui kehadiran objek dengan mengetahui koordinat persis tempat objek berada dalam ruang 2D."
    },
    # 3. Learning Objectives
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran Bab 25",
        "body": "Setelah menyelesaikan sesi ini, Anda diharapkan mampu:\n\n- Membedakan Classification, Localization, dan Object Detection.\n- Menguasai konversi koordinat Bounding Box (`xyxy` vs `xywh`, absolut vs normalisasi).\n- Menghitung matematis Intersection over Union (IoU) dan menjalankan algoritma NMS.\n- Mengevaluasi performa model melalui metrik Precision, Recall, AP, dan mAP standar COCO.\n- Memahami evolusi arsitektur one-stage YOLO (Backbone, Neck, Head).\n- Menyiapkan format anotasi YOLO dan menjalankan pipeline inferensi pada CPU.",
        "notes": "Paparkan sasaran pembelajaran dengan jelas agar mahasiswa memahami capaian teori matematis dan keterampilan praktis yang akan dinilai."
    },
    # 4. Classification vs Detection
    {
        "kind": "split",
        "title": "Klasifikasi vs Lokalisasi vs Deteksi Objek",
        "body": "**Image Classification**\n- Input: Citra tunggal\n- Output: Label kelas global tunggal (misal: 'Kucing')\n- Lokasi spasial objek diabaikan total\n\n**Object Detection**\n- Input: Citra dengan $N$ objek arbitrer\n- Output: Kumpulan $(c, x_1, y_1, x_2, y_2, p)$\n- Mengidentifikasi kategori sekaligus kotak pembatas setiap entitas fisik",
        "notes": "Bahas bagaimana kompleksitas komputasi melonjak dari memprediksi 1 probabilitas kelas menjadi memprediksi banyak pasangan (koordinat + kelas) untuk jumlah objek yang bervariasi."
    },
    # 5. Object Localization
    {
        "kind": "content",
        "title": "Object Localization: Deteksi Objek Tunggal",
        "body": "Lokalisasi objek merupakan jembatan transisi antara klasifikasi murni dan deteksi multi-objek:\n\n- Arsitektur jaringan memiliki **dua kepala keluaran (multi-head output)**:\n  1. *Classification Head*: Vektor probabilitas kelas $\\hat{y} \\in \\mathbb{R}^C$.\n  2. *Bounding Box Regression Head*: Vektor koordinat $\\mathbf{b} = (x, y, w, h) \\in \\mathbb{R}^4$.\n- Fungsi rugi gabungan (*Joint Loss*):\n  $$\\mathcal{L} = \\mathcal{L}_{\\text{cls}}(y, \\hat{y}) + \\lambda \\cdot \\mathcal{L}_{\\text{reg}}(\\mathbf{b}, \\mathbf{\\hat{b}})$$",
        "notes": "Jelaskan bahwa lokalisasi objek mengasumsikan hanya ada tepat satu objek dominan di dalam citra, sehingga output regresi berukuran tetap."
    },
    # 6. Bounding Boxes
    {
        "kind": "content",
        "title": "Representasi Geometris Bounding Box",
        "body": "Bounding box adalah persegi panjang terkecil yang membatasi seluruh siluet tampak dari suatu objek fisik pada bidang citra 2D:\n\n- Direpresentasikan oleh 4 nilai skalar numerik riil.\n- Bersifat axis-aligned (sejajar dengan sumbu horizontal dan vertikal citra).\n- Menghilangkan variasi orientasi rotasi demi efisiensi representasi dan simplifikasi komputasi regresi gradien.",
        "notes": "Soroti bahwa bounding box axis-aligned adalah kompromi praktis antara kesederhanaan komputasi dan ketepatan cakupan objek."
    },
    # 7. Bounding Box Formats
    {
        "kind": "split",
        "title": "Format Koordinat: xyxy vs xywh",
        "body": "**Format PASCAL VOC (`xyxy`)**\n- Koordinat dua sudut berseberangan:\n  $$[x_{\\text{min}}, y_{\\text{min}}, x_{\\text{max}}, y_{\\text{max}}]$$\n- Sangat intuitif untuk komputasi luas irisan (Intersection Area).\n\n**Format YOLO / COCO (`xywh`)**\n- Titik pusat dan rentang dimensi:\n  $$[x_{\\text{center}}, y_{\\text{center}}, \\text{width}, \\text{height}]$$\n- Format YOLO selalu dinormalisasi ke interval $[0.0, 1.0]$.",
        "notes": "Ingatkan mahasiswa untuk selalu mengecek format koordinat yang diminta sebelum memberi makan ke fungsi loss atau penampil grafis."
    },
    # 8. Intersection over Union
    {
        "kind": "content",
        "title": "Konsep Matematis Intersection over Union (IoU)",
        "body": "Intersection over Union (IoU) atau indeks Jaccard mengukur derajat tumpang tindih spasial antara Bounding Box Prediksi ($B_p$) dan Bounding Box Ground-Truth ($B_g$):\n\n$$\\text{IoU}(B_p, B_g) = \\frac{\\text{Area}(B_p \\cap B_g)}{\\text{Area}(B_p \\cup B_g)} = \\frac{\\text{Area}(B_p \\cap B_g)}{\\text{Area}(B_p) + \\text{Area}(B_g) - \\text{Area}(B_p \\cap B_g)}$$\n\n- $\\text{IoU} = 1.0$: Tumpang tindih sempurna (100% cocok).\n- $\\text{IoU} = 0.0$: Kedua kotak terpisah total tanpa ada irisan.",
        "notes": "Tekankan bahwa IoU bernilai invarian terhadap skala citra, sehingga kotak kecil dan kotak besar dinilai dengan keadilan matematis yang setara."
    },
    # 9. IoU Worked Example
    {
        "kind": "case",
        "title": "Contoh Terhitung Manual Nilai IoU",
        "body": "**Diberikan dua kotak berdimensi 200x200:**\n- Kotak A: $[100, 100, 300, 300] \\to \\text{Luas} = 200 \\times 200 = 40.000$\n- Kotak B: $[200, 200, 400, 400] \\to \\text{Luas} = 200 \\times 200 = 40.000$\n\n**Perhitungan:**\n- Irisan: $[\\max(100, 200), \\max(100, 200), \\min(300, 400), \\min(300, 400)]$\n- Koordinat Irisan: $[200, 200, 300, 300] \\to \\text{Luas} = 100 \\times 100 = 10.000$\n- Luas Gabungan (Union): $40.000 + 40.000 - 10.000 = 70.000$\n- $\\text{IoU} = \\frac{10.000}{70.000} = \\frac{1}{7} \\approx 0.142857$",
        "notes": "Ajak mahasiswa menelusuri bagaimana rumus min dan max menentukan koordinat irisan secara deterministik."
    },
    # 10. Confidence Score
    {
        "kind": "content",
        "title": "Skor Keyakinan (Confidence Score)",
        "body": "Pada detektor objek modern (seperti YOLO), setiap prediksi menghasilkan skor keyakinan terpadu:\n\n$$\\text{Confidence} = P(\\text{Object}) \\times \\text{IoU}_{\\text{pred}}^{\\text{truth}}$$\n\n- $P(\\text{Object}) \\in [0, 1]$: Probabilitas bahwa kotak tersebut memuat objek fisik (bukan latar belakang kosong / *background*).\n- $\\text{IoU}_{\\text{pred}}^{\\text{truth}}$: Ekspektasi seberapa akurat batas kotak tersebut membatasi objek secara pas.",
        "notes": "Jelaskan bahwa skor keyakinan menggabungkan kepastian klasifikasi objek dan kepastian kualitas lokalisasi kotak."
    },
    # 11. Duplicate Detections
    {
        "kind": "content",
        "title": "Masalah Duplikasi Deteksi pada Citra",
        "body": "Karena model deteksi mengevaluasi ribuan grid sel atau anchor box di seluruh citra secara paralel:\n\n- Satu objek fisik nyata (misal sebuah mobil) sering kali memicu **puluhan bounding box kandidat** yang saling bertumpuk di sekitarnya.\n- Masing-masing kotak memiliki skor keyakinan yang sedikit berbeda.\n- Membiarkan seluruh kotak tersebut muncul akan memicu alarm palsu masif dan membingungkan sistem pengambil keputusan.",
        "notes": "Gambarkan visual bagaimana satu mobil tampak dikelilingi oleh 15 kotak hijau yang sedikit bergeser beberapa piksel satu sama lain."
    },
    # 12. Non-Maximum Suppression
    {
        "kind": "content",
        "title": "Algoritma Non-Maximum Suppression (NMS)",
        "body": "NMS bertugas menapis prediksi duplikat agar hanya satu kotak terbaik yang dipertahankan:\n\n```python\n# 1. Tapis kotak dengan skor < threshold_keyakinan\n# 2. Urutkan kotak tersisa berdasarkan skor keyakinan menurun\nwhile kotak_kandidat:\n    kotak_terbaik = ambil_skor_tertinggi(kotak_kandidat)\n    simpan(kotak_terbaik)\n    for k in sisa_kotak:\n        if IoU(kotak_terbaik, k) > threshold_iou:\n            hapus(k) # Singkirkan karena dianggap duplikat\n```",
        "notes": "Jelaskan bahwa threshold IoU yang umum adalah 0.45 hingga 0.50. Jika disetel terlalu rendah, dua objek bersebelahan bisa ikut terhapus."
    },
    # 13. Detection Evaluation
    {
        "kind": "content",
        "title": "Evaluasi Deteksi: TP, FP, dan FN",
        "body": "Dalam deteksi objek, label kebenaran ditentukan melalui pencocokan berbasis IoU terhadap Ground Truth (GT):\n\n- **True Positive (TP)**: Prediksi dengan $\\text{IoU} \\ge \\text{threshold}$ terhadap GT kelas yang sama (dan GT tersebut belum dicocokkan).\n- **False Positive (FP)**: Prediksi dengan $\\text{IoU} < \\text{threshold}$, prediksi pada latar kosong, atau prediksi duplikat pada GT yang sudah terisi.\n- **False Negative (FN)**: Objek GT yang tidak berhasil dideteksi oleh prediksi mana pun.",
        "notes": "Soroti bahwa prediksi duplikat kedua pada objek yang sama dihitung sebagai False Positive, bukan True Positive."
    },
    # 14. Precision and Recall
    {
        "kind": "split",
        "title": "Presisi, Recall, dan Kurva PR",
        "body": "**Precision & Recall**\n$$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$$\n$$\\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$\n\n**Kurva Precision-Recall (PR Curve)**\n- Dihasilkan dengan menghitung presisi kumulatif seiring diururkannya prediksi berdasarkan skor keyakinan.\n- Menampilkan kompromi (*trade-off*) deteksi agresif vs konservatif.",
        "notes": "Jelaskan bahwa kurva PR deteksi objek dihitung secara kumulatif dari deteksi paling percaya diri ke yang paling ragu-ragu."
    },
    # 15. AP and mAP
    {
        "kind": "content",
        "title": "Average Precision (AP) dan Mean AP (mAP)",
        "body": "Average Precision (AP) mengukur luas area di bawah kurva PR untuk satu kelas objek tertentu:\n\n$$\\text{AP} = \\int_0^1 P(r) \\, dr \\quad \\approx \\sum_{k=1}^N (R_k - R_{k-1}) \\cdot P_{\\text{interp}}(R_k)$$\n\n- **mAP@0.5**: Rata-rata nilai AP di seluruh kelas pada ambang batas $\\text{IoU} = 0.50$ (Protokol PASCAL VOC).\n- **mAP@0.5:0.95**: Rata-rata AP di seluruh kelas pada 10 ambang batas IoU ($0.50, 0.55, \\dots, 0.95$) (Protokol resmi COCO).",
        "notes": "Klarifikasi kepada mahasiswa bahwa mAP@0.5:0.95 selalu memiliki angka lebih rendah daripada mAP@0.5 karena menuntut akurasi batas kotak yang sangat presisi."
    },
    # 16. YOLO Introduction
    {
        "kind": "content",
        "title": "Pengantar YOLO: You Only Look Once",
        "body": "Diperkenalkan oleh Joseph Redmon et al. (2016), YOLO merevolusi paradigma deteksi objek visual:\n\n- Mengubah deteksi objek dari masalah klasifikasi berbasis usulan wilayah (*Region Proposal*) menjadi **masalah regresi spasial tunggal langsung dari piksel ke koordinat**.\n- Mampu memproses citra dalam satu kali forward pass secara simultan (*You Only Look Once*).\n- Memungkinkan pemrosesan video berkecepatan tinggi secara real-time pada 45 hingga 150+ FPS.",
        "notes": "Ceritakan latar belakang sejarah kemunculan YOLO yang mengakhiri dominasi sistem multi-stage yang lambat."
    },
    # 17. One-Stage vs Two-Stage
    {
        "kind": "split",
        "title": "Two-Stage vs One-Stage Detectors",
        "body": "**Two-Stage Detectors (R-CNN, Faster R-CNN)**\n- Tahap 1: Region Proposal Network (RPN) membangkitkan ribuan usulan area objek.\n- Tahap 2: Klasifikasi dan penyesuaian batas setiap proposal.\n- *Karakteristik*: Sangat akurat namun lambat (5-15 FPS).\n\n**One-Stage Detectors (YOLO, SSD, RetinaNet)**\n- Memprediksi bounding box dan probabilitas kelas langsung dalam satu langkah forward pass.\n- *Karakteristik*: Sangat cepat (30-100+ FPS) dengan akurasi modern setara.",
        "notes": "Bahas kapan industri memilih one-stage (sistem real-time, robotika) vs two-stage (inspeksi medis beresolusi sangat lambat)."
    },
    # 18. YOLO Evolution
    {
        "kind": "content",
        "title": "Evolusi Arsitektur Keluarga YOLO",
        "body": "Keluarga YOLO terus berevolusi secara signifikan dan tidak memiliki arsitektur identik di setiap generasinya:\n\n- **YOLOv1 (2016)**: Grid $7 \\times 7$, detektor berbasis fully-connected, rentan terhadap objek kecil berdekatan.\n- **YOLOv2/v3 (2017-2018)**: Darknet, anchor boxes, dan multi-scale detection pada 3 skala piramida fitur.\n- **YOLOv4/v5 (2020)**: CSPDarknet, PANet neck, augmentasi Mosaic canggih, dan ekosistem PyTorch portabel.\n- **YOLOv8/v11 (2023-2024)**: Arsitektur *Anchor-Free*, C2f blocks, decoupling head, dan dukungan multi-task terpadu.",
        "notes": "Tegaskan bahwa YOLO modern (seperti YOLOv8) telah beralih ke Anchor-Free yang mengeliminasi kebutuhan penyesuaian manual anchor box."
    },
    # 19. YOLO Architecture
    {
        "kind": "content",
        "title": "Tiga Komponen Utama Arsitektur YOLO",
        "body": "Arsitektur detektor modern dibangun dari tiga pilar blok modular:\n\n```\n[ Citra Input ] ──> [ BACKBONE ] ──> [ NECK ] ──> [ DETECTION HEAD ] ──> [ BBox + Kelas ]\n                      Ekstraksi       Agregasi     Prediksi Multi-Skala\n                      Fitur Dasar     Multi-Skala  (Koordinat & Logits)\n```\n\n1. **Backbone**: Lapisan konvolusional dalam untuk mengekstraksi representasi visual berhierarki.\n2. **Neck**: Jaringan agregasi fitur piramidal (FPN/PANet) untuk menggabungkan konteks semantik dan detail spasial.\n3. **Head**: Lapisan akhir yang memproyeksikan fitur menjadi koordinat bounding box dan skor kelas.",
        "notes": "Jelaskan peran masing-masing komponen agar mahasiswa memahami bagaimana aliran tensor diproses dari hulu ke hilir."
    },
    # 20. Backbone
    {
        "kind": "content",
        "title": "Backbone: Ekstraksi Fitur Berhierarki",
        "body": "Backbone bertugas memproses citra resolusi penuh menjadi peta fitur beresolusi rendah dengan kedalaman kanal tinggi:\n\n- Menggunakan blok konvolusional modern seperti *Cross Stage Partial (CSP)* atau *C2f*.\n- Menerapkan *residual connections* untuk mempertahankan gradien pada lapisan dalam.\n- Menghasilkan representasi fitur pada berbagai tingkat resolusi downsampling ($P_3: 1/8$, $P_4: 1/16$, $P_5: 1/32$).",
        "notes": "Ingatkan konsep hierarki fitur dari Bab 24: lapisan awal menangkap tepi, lapisan dalam menangkap bentuk semantik."
    },
    # 21. Neck and Head
    {
        "kind": "split",
        "title": "Neck (PANet) dan Head (Decoupled)",
        "body": "**Neck: Feature Pyramid Network (PANet)**\n- Menggabungkan fitur semantik kuat dari lapisan atas dengan detail spasial presisi dari lapisan bawah.\n- Menggunakan jalur *top-down* dan *bottom-up* ganda.\n\n**Head: Decoupled Detection Head**\n- Memisahkan cabang komputasi untuk regresi kotak dan klasifikasi objek.\n- Mengurangi interferensi gradien antara tugas lokalisasi dan kategorisasi.",
        "notes": "Jelaskan mengapa pemisahan kepala regresi dan klasifikasi (Decoupled Head) meningkatkan konvergensi pelatihan secara signifikan."
    },
    # 22. Multi-Scale Detection
    {
        "kind": "content",
        "title": "Deteksi Multi-Skala (Multi-Scale Detection)",
        "body": "Salah satu kekuatan utama YOLO modern adalah kemampuan mendeteksi objek dengan variasi ukuran ekstrem:\n\n- **Skala Kecil (Resolusi Fitur $P_3$, Grid $80 \\times 80$)**: Mengidentifikasi objek kecil (seperti rambu lalu lintas atau burung di kejauhan).\n- **Skala Sedang (Resolusi Fitur $P_4$, Grid $40 \\times 40$)**: Mengidentifikasi objek berukuran proporsional (seperti pejalan kaki).\n- **Skala Besar (Resolusi Fitur $P_5$, Grid $20 \\times 20$)**: Mengidentifikasi objek besar yang memenuhi bidang (seperti bus atau truk).",
        "notes": "Bahas bagaimana pembagian 3 skala grid ini menyelesaikan masalah utama kegagalan deteksi objek kecil pada YOLOv1."
    },
    # 23. Dataset Annotation
    {
        "kind": "content",
        "title": "Anotasi Citra untuk Deteksi Objek",
        "body": "Pelatihan model deteksi terbimbing (*supervised*) menuntut anotasi spasial berkualitas tinggi:\n\n- Setiap objek diberi kotak pembatas ketat (*tight bounding box*) tanpa menyisakan terlalu banyak latar belakang.\n- Seluruh objek relevan dalam citra **wajib** diberi label; melewatkan satu objek akan mengajari model bahwa objek tersebut adalah latar belakang (*false supervision*)!\n- Alat anotasi standar industri: CVAT, LabelImg, Roboflow, dan VoTT.",
        "notes": "Tekankan bahaya anotasi yang terlewat (omission error) yang dapat membingungkan fungsi loss penalti background."
    },
    # 24. YOLO Label Format
    {
        "kind": "code",
        "title": "Format Berkas Label Standar YOLO (.txt)",
        "body": "Setiap citra `gambar.jpg` dipasangkan dengan berkas teks anotasi `gambar.txt` dengan baris format:\n\n```text\n<class_id> <x_center> <y_center> <width> <height>\n0 0.450000 0.520000 0.300000 0.600000\n1 0.820000 0.350000 0.150000 0.250000\n```\n\n- Seluruh koordinat berada pada interval ternormalisasi $[0.0, 1.0]$.\n- Invarian terhadap perubahan resolusi citra asli.",
        "notes": "Jelaskan perhitungan normalisasi: x_center = x_center_pixels / image_width."
    },
    # 25. Dataset Preparation
    {
        "kind": "content",
        "title": "Tata Kelola Partisi Dataset Bebas Leakage",
        "body": "Protokol penyusunan dataset deteksi objek terstandar:\n\n- **Train (70%)**: Sumber pembaruan bobot model dengan augmentasi citra aktif (Mosaic, Random Flip, Color Jitter).\n- **Validation (15%)**: Evaluasi mAP per epoch untuk penyesuaian hyperparameter dan early stopping.\n- **Test (15%)**: Partisi evaluasi akhir yang diisolasi ketat bebas kebocoran data (*zero leakage*).\n- *Aturan Krusial*: Jika frame video digunakan, seluruh frame dari satu sekuens video yang sama WAJIB berada pada partisi yang sama guna mencegah korelasi temporal.",
        "notes": "Peringatkan mahasiswa agar tidak mencampur potongan frame video dari rekaman yang sama ke dalam train dan test set."
    },
    # 26. Pretrained Inference
    {
        "kind": "code",
        "title": "Inferensi Pretrained YOLO pada CPU",
        "body": "Menjalankan inferensi model terlatih menggunakan API Ultralytics:\n\n```python\nfrom ultralytics import YOLO\nimport cv2\n\n# Memuat model ringkas teroptimasi CPU (YOLOv8 Nano ~6MB)\nmodel = YOLO('yolov8n.pt')\n\n# Jalankan inferensi pada citra\nresults = model('scene.jpg', conf=0.25, device='cpu')\n\n# Ekstraksi koordinat dan kelas\nfor box in results[0].boxes:\n    xyxy = box.xyxy[0].tolist()\n    conf = float(box.conf[0])\n    cls_id = int(box.cls[0])\n    print(f'Objek {cls_id}: conf={conf:.2f}, bbox={xyxy}')\n```",
        "notes": "Tunjukkan bahwa API modern membuat inferensi deteksi objek sangat deklaratif dan terstruktur."
    },
    # 27. Detection Visualization
    {
        "kind": "content",
        "title": "Rendering Hasil Deteksi & Visualisasi",
        "body": "Penyajian visual hasil deteksi objek yang baik harus memenuhi standar keterbacaan:\n\n- Ketebalan garis proporsional terhadap resolusi citra masukan.\n- Kartu latar belakang teks dengan warna kontras untuk mencegah teks tak terbaca di atas citra berpola ramai.\n- Palet warna unik untuk setiap kategori kelas objek.\n- Menampilkan label kelas bersama dengan nilai persentase skor keyakinan.",
        "notes": "Diskusikan aspek estetika dan fungsional visualisasi komputer agar hasil analisis mudah dipahami oleh pengguna non-teknis."
    },
    # 28. Model Performance
    {
        "kind": "content",
        "title": "Evaluasi Komparasi Varian Ukuran Model",
        "body": "Keluarga model YOLOv8 menyediakan variasi ukuran model untuk berbagai target perangkat:\n\n- **YOLOv8n (Nano)**: 3.2M params | 8.7 GFLOPs | Ideal untuk CPU laptop dan perangkat Edge AI hemat daya.\n- **YOLOv8s (Small)**: 11.2M params | 28.6 GFLOPs | Keseimbangan baik antara kecepatan dan ketepatan deteksi.\n- **YOLOv8m (Medium)**: 25.9M params | 78.9 GFLOPs | Menuntut akselerator GPU untuk performa real-time.\n- **YOLOv8x (eXtra Large)**: 68.2M params | 257.8 GFLOPs | Akurasi tertinggi untuk server cloud batch.",
        "notes": "Ajak mahasiswa menganalisis bahwa model terbesar tidak selalu menjadi pilihan terbaik jika target deployment adalah CPU lokal atau embedded system."
    },
    # 29. Latency and Computational Cost
    {
        "kind": "split",
        "title": "Trade-Off: Akurasi vs Latensi vs Biaya Komputasi",
        "body": "**Tuntutan Real-Time Latency**\n- Video 30 FPS menuntut latensi inferensi total di bawah 33.3 ms per frame.\n- Pada CPU standar, YOLOv8n mencatat latensi ~15–30 ms (memenuhi ambang batas real-time).\n\n**Trade-Off Pertimbangan Insinyur**\n- Menaikkan resolusi citra ($640 \\to 1280$) melipatgandakan FLOPs hingga 4 kali lipat!\n- Pemilihan model wajib didasarkan pada batasan perangkat keras fisik di lapangan.",
        "notes": "Tekankan hukum kuadrat komputasi konvolusi terhadap ukuran resolusi spasial citra."
    },
    # 30. Fine-Tuning Overview
    {
        "kind": "content",
        "title": "Pengantar Fine-Tuning pada Dataset Kustom",
        "body": "Menerapkan Transfer Learning dengan bobot pretrained COCO pada dataset domain khusus:\n\n- Memanfaatkan fitur umum (tepi, bentuk, tekstur) yang telah dipelajari dari jutaan citra COCO.\n- Membekukan sebagian lapisan Backbone awal dan melatih ulang Detection Head dengan laju pembelajaran rendah.\n- Memungkinkan konvergensi model kustom hanya dengan ratusan sampel citra dalam hitungan puluhan epoch tanpa perlu melatih dari nol (*from scratch*).",
        "notes": "Jelaskan bahwa fine-tuning menghemat ribuan jam komputasi dan menjamin model tidak overfitting pada dataset kecil."
    },
    # 31. Responsible Computer Vision
    {
        "kind": "content",
        "title": "Etika & Tanggung Jawab dalam Computer Vision",
        "body": "Penerapan sistem deteksi objek di ruang publik menuntut pertimbangan etis ketat:\n\n- **Privasi & Penyamaran Wajah/Plat Nomor**: Deteksi objek tidak boleh disalahgunakan untuk pengawasan massal tanpa izin (*unwarranted mass surveillance*).\n- **Bias Representasi Geografis & Demografis**: Model yang dilatih pada dataset negara maju sering gagal mendeteksi objek khas lokal di negara berkembang.\n- **Hak Cipta Citra**: Pengumpulan dataset wajib mematuhi lisensi hak cipta dan ketentuan penggunaan data terbuka.",
        "notes": "Kaitkan kembali materi ini dengan fondasi AI Ethics dari Bab 09 mengenai prinsip perlindungan data pribadi dan keadilan algoritmik."
    },
    # 32. Python Demonstration
    {
        "kind": "content",
        "title": "Demonstrasi Praktik Laboratorium Bab 25",
        "body": "Seluruh eksperimen mandiri dapat diuji melalui 10 modul skrip terstruktur di `code/`:\n\n1. Pembuktian formula IoU dan presisi konversi koordinat.\n2. Verifikasi algoritma NMS mandiri melawan OpenCV `NMSBoxes`.\n3. Perhitungan analitis kurva PR dan mAP@0.5 vs mAP@0.5:0.95.\n4. Parser dan audit integritas dataset YOLO.\n5. Inferensi CPU model terlatih dengan visualisasi diagnostik TP/FP/FN.",
        "notes": "Bimbing mahasiswa untuk mulai mengeksekusi skrip Python satu per satu dari terminal."
    },
    # 33. Case Study
    {
        "kind": "case",
        "title": "Studi Kasus: Monitoring Kepadatan Lalu Lintas Jalan Raya",
        "body": "**Skenario Permasalahan:**\nDepartemen Perhubungan ingin menghitung kepadatan kendaraan di persimpangan jalan kota secara otomatis menggunakan kamera CCTV resolusi 720p yang terhubung ke komputer mini CPU lokal.\n\n**Solusi Rekayasa:**\n- Memilih model ringan YOLOv8 Nano yang berjalan pada ~25 ms per frame pada CPU.\n- Menerapkan NMS Class-Aware agar sepeda motor yang berdekatan dengan mobil tidak terhapus keliru.\n- Mengkalibrasi ambang batas keyakinan pada 0.40 guna menekan False Alarm akibat bayangan aspal tanpa melewatkan kendaraan.",
        "notes": "Ajak mahasiswa berdiskusi tentang trade-off ambang batas keyakinan pada kondisi cuaca hujan atau malam hari."
    },
    # 34. Discussion
    {
        "kind": "split",
        "title": "Pertanyaan Refleksi & Diskusi Kelas",
        "body": "**Pertanyaan 1: Bounding Box Bertumpuk**\nJika dua orang berjalan beriringan dan saling merangkul, apa risiko yang dapat terjadi pada tahap Non-Maximum Suppression? Bagaimana mitigasinya?\n\n**Pertanyaan 2: Resolusi Spasial**\nMengapa menaikkan resolusi citra input sangat membantu mendeteksi objek kecil, namun dapat menggagalkan sistem real-time pada CPU?",
        "notes": "Gunakan pertanyaan ini untuk memicu interaksi aktif mahasiswa di kelas sebelum sesi penutup."
    },
    # 35. Summary
    {
        "kind": "summary",
        "title": "Rangkuman Pembelajaran Bab 25",
        "body": "- Deteksi objek menggabungkan klasifikasi kategori dan regresi koordinat spasial bounding box.\n- IoU mengukur kualitas tumpang tindih spasial antara prediksi dan ground-truth.\n- NMS mengeliminasi prediksi duplikat redundan dengan mempertahankan kotak berskor keyakinan tertinggi.\n- Evaluasi model didasarkan pada kurva PR terinterpolasi dan metrik mAP (mAP@0.5 dan mAP@0.5:0.95).\n- YOLO menghadirkan deteksi one-stage berkecepatan tinggi dengan mengintegrasikan Backbone, Neck, dan Head.",
        "notes": "Tutup sesi teori dengan merangkum lima poin kunci sebelum berpindah ke praktik laboratorium."
    },
    # 36. References
    {
        "kind": "content",
        "title": "Referensi Akademik Terverifikasi",
        "body": "1. **Redmon, J., et al.** (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016, 779-788.\n2. **Lin, T. Y., et al.** (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014, 740-755.\n3. **Everingham, M., et al.** (2010). *The Pascal Visual Object Classes (VOC) Challenge*. IJCV 88(2), 303-338.\n4. **Jocher, G., et al.** (2023). *Ultralytics YOLOv8 Documentation and Architecture*. https://docs.ultralytics.com\n5. **ACM / IEEE-CS.** (2024). *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. https://csed.acm.org",
        "notes": "Sampaikan referensi resmi kepada mahasiswa untuk penelusuran literatur akademik lanjutan."
    }
]

output_path = Path(__file__).resolve().parent / "deck.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(slides, f, indent=2, ensure_ascii=False)

print(f"Berhasil menghasilkan {len(slides)} slide Reveal.js di: {output_path}")

