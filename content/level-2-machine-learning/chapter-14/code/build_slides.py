"""
Script builder untuk slides deck Chapter 14: deck.json (32 slide)
Chapter 14: Supervised Learning — Classification Algorithms
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "Classification Algorithms",
        "body": "### Regresi Logistik, Naive Bayes, Kalibrasi Ambang Batas & Metrik Evaluasi\n\n**Level 2 — Machine Learning | Chapter 14**\n\nArtificial Intelligence Learning Hub — Institut Teknologi Sumatera (ITERA)\n*Created & Developed by Aji Pamungkas Tri Nurcahyo*",
        "notes": "Selamat datang di perkuliahan Chapter 14: Supervised Learning — Classification Algorithms. Hari ini kita membahas algoritma pemodelan kategori diskrit, mulai dari Regresi Logistik, Naive Bayes, optimasi decision threshold, kurva ROC/PR, hingga penanganan ketidakseimbangan kelas."
    },
    {
        "kind": "content",
        "title": "Classification Applications",
        "body": "Penerapan nyata algoritma klasifikasi di dunia modern:\n\n- **Pemeliharaan Prediktif Pabrik**: Membedakan status mesin Normal vs Kerusakan Bantalan vs Overheating secara otomatis.\n- **Keamanan Siber & Fraud Detection**: Mendeteksi paket jaringan berbahaya atau transaksi perbankan anomali.\n- **Diagnosis Medis Cerdas**: Mengklasifikasikan citra medis menjadi tumor jinak (*benign*) vs ganas (*malignant*).\n- **Pemrosesan Bahasa Alami**: Penyaringan email spam dan analisis sentimen ulasan pelanggan.",
        "notes": "Tunjukkan bahwa klasifikasi adalah tulang punggung sistem pendukung keputusan cerdas di berbagai bidang."
    },
    {
        "kind": "content",
        "title": "Learning Objectives",
        "body": "Setelah menyelesaikan bab ini, mahasiswa diharapkan mampu:\n\n- **Menjelaskan** konsep Classification dalam Supervised Learning dan perbedaannya dengan Regression.\n- **Membedakan** Binary Classification dan Multiclass Classification.\n- **Menjelaskan** formulasi Logistic Regression, fungsi Sigmoid, dan Log-Odds.\n- **Menjelaskan** pengklasifikasi Gaussian dan Multinomial Naive Bayes serta Laplace Smoothing.\n- **Mengoptimasi** decision threshold berdasarkan trade-off kerugian bisnis.\n- **Menginterpretasikan** Confusion Matrix, Accuracy, Precision, Recall, Specificity, dan F1-Score.\n- **Menganalisis** kurva ROC, ROC-AUC, Precision-Recall Curve, dan Average Precision.\n- **Menerapkan** evaluasi klasifikasi yang bertanggung jawab pada data tidak seimbang.",
        "notes": "Rincikan 8 capaian kompetensi utama perkuliahan hari ini."
    },
    {
        "kind": "split",
        "title": "Classification Problem",
        "body": "Memahami karakteristik masalah klasifikasi:\n\n### Ruang Label Diskrit\n- Memetakan fitur input $X$ ke dalam himpunan kategori terhingga: $y \\in \\{c_1, \\dots, c_K\\}$.\n- Menjawab pertanyaan kualitatif: *\"Masuk kategori mana sampel ini?\"*\n\n### Evaluasi Partisi Wilayah\n- Membagi ruang fitur menjadi beberapa wilayah keputusan (*decision regions*).\n- Evaluasi berfokus pada kebenaran asosiasi kategori dan tingkat kepastian probabilitas.",
        "notes": "Bandingkan dengan regresi kontinu dari Bab 13."
    },
    {
        "kind": "split",
        "title": "Binary vs Multiclass",
        "body": "Dua bentuk masalah klasifikasi berdasarkan jumlah kelas:\n\n### Binary Classification ($K = 2$)\n- Tepat dua kelas yang saling bertolak belakang: $y \\in \\{0, 1\\}$.\n- Contoh: Mesin Normal (0) vs Mesin Rusak (1).\n\n### Multiclass Classification ($K > 2$)\n- Lebih dari dua kategori diskrit: $y \\in \\{0, 1, 2, \\dots, K-1\\}$.\n- Contoh: Normal, Kerusakan Bantalan, Overheating, Lonjakan Listrik.\n- Tiap sampel hanya memiliki tepat satu label benar.",
        "notes": "Jelaskan konvensi kelas 0 sebagai status normal dan kelas 1 sebagai target anomali yang dicari."
    },
    {
        "kind": "content",
        "title": "Features and Labels",
        "body": "Struktur representasi data klasifikasi:\n\n- **Matriks Fitur ($X \\in \\mathbb{R}^{N \\times D}$)**: Vektor kuantitatif sensor (suhu, getaran RMS, kebisingan akustik, tekanan, tegangan).\n- **Vektor Target ($y \\in \\mathcal{C}^N$)**: Label kelas kategorikal yang ingin dipelajari polanya.\n- **Probabilitas Kelas ($\\hat{p}_k$)**: Derajat keyakinan model terhadap kelas ke-$k$ ($0 \\le \\hat{p}_k \\le 1$).\n- **Label Prediksi ($\\hat{y}$)**: Kategori akhir hasil penetapan ambang batas atau nilai probabilitas tertinggi.",
        "notes": "Ingatkan kembali representasi data tabular dari Bab 07."
    },
    {
        "kind": "split",
        "title": "Decision Boundary",
        "body": "Pemisah geometris ruang hipotesis:\n\n### Konsep Batas Keputusan\n- Garis atau bidang di mana model berada pada tingkat keraguan seimbang antar-kelas ($P(y=1|x) = 0.5$).\n\n### Batas Linier vs Non-Linier\n- **Model Linier (Logistic Regression)**: Membentuk bidang datar linier $\\mathbf{w}^T \\mathbf{x} + b = 0$.\n- **Model Non-Linier (Trees, RBF Kernel)**: Membentuk kontur batas yang melengkung dan fleksibel.",
        "notes": "Visualisasikan garis pemisah dua warna titik data pada bidang kartesius."
    },
    {
        "kind": "content",
        "title": "Logistic Regression",
        "body": "Prinsip dasar Regresi Logistik (Logistic Regression):\n\n- **Definisi**: Model klasifikasi probabilistik yang menghitung kombinasi linier fitur $z = \\mathbf{w}^T \\mathbf{x} + b$, lalu memetakannya ke interval probabilitas $[0, 1]$.\n- **Bukan Regresi Numerik**: Disebut regresi karena memodelkan hubungan linier terhadap log-odds, namun output akhirnya adalah klasifikasi kategori.\n- **Karakteristik**: Sangat cepat dilatih, tahan terhadap overfitting moderat, dan menghasilkan interpretasi Odds Ratio yang sangat bernilai bagi domain pakar.",
        "notes": "Tegaskan paradoks nama: Logistic Regression adalah algoritma klasifikasi, bukan regresi kontinu."
    },
    {
        "kind": "split",
        "title": "Sigmoid Function",
        "body": "Fungsi aktivasi pengubah skor linier menjadi probabilitas:\n\n### Formula Matematis\n$$\\sigma(z) = \\frac{1}{1 + e^{-z}} = \\frac{e^z}{1 + e^z}$$\n\n### Properti Kunci\n- Asimtotik tertutup pada interval $(0, 1)$.\n- Nilai tengah simetris: $\\sigma(0) = 0.5$.\n- Jika $z \\to +\\infty$, $\\sigma(z) \\to 1$; jika $z \\to -\\infty$, $\\sigma(z) \\to 0$.",
        "notes": "Tunjukkan grafik kurva huruf S dari fungsi Sigmoid."
    },
    {
        "kind": "content",
        "title": "Log-Odds and Odds Ratio",
        "body": "Interpretasi statistika parameter model logistik:\n\n- **Odds**: Rasio probabilitas sukses terhadap gagal: $\\text{Odds} = p / (1 - p)$.\n- **Fungsi Logit (Log-Odds)**: Hubungan linier terhadap parameter bobot:\n  $$\\ln\\left(\\frac{p}{1 - p}\\right) = \\mathbf{w}^T \\mathbf{x} + b$$\n- **Odds Ratio $\\exp(w_j)$**: Pengali risiko marginal untuk setiap penambahan 1 satuan pada fitur $x_j$.\n- **Contoh Riil Sensor**: $w_{\\text{vibration}} = 1.7575 \\implies \\exp(w) = 5.80\\times$. Setiap kenaikan 1 mm/s getaran melipatgandakan risiko kerusakan mesin sebesar $5.80$ kali lipat!",
        "notes": "Tekankan bahwa interpretasi odds ratio ini adalah kekuatan utama regresi logistik di bidang medis dan industri."
    },
    {
        "kind": "split",
        "title": "Decision Threshold",
        "body": "Menetapkan batas klasifikasi biner:\n\n### Ambang Batas Default ($T = 0.5$)\n- Jika $P(y=1|x) \\ge 0.5 \\implies \\hat{y} = 1$ (Fault).\n- Jika $P(y=1|x) < 0.5 \\implies \\hat{y} = 0$ (Normal).\n\n### Fleksibilitas Rekayasa\n- Ambang $0.5$ hanyalah konvensi simetris.\n- Pada aplikasi industri, ambang batas harus digeser sesuai biaya risiko kegagalan.",
        "notes": "Jelaskan bahwa thresholding adalah jembatan antara matematika model dan keputusan bisnis."
    },
    {
        "kind": "content",
        "title": "Logistic Loss",
        "body": "Fungsi biaya Binary Cross-Entropy Loss:\n\n- **Mengapa Bukan MSE?**: MSE pada Sigmoid menghasilkan fungsi non-konveks yang rentan terjebak di minimum lokal.\n- **Formulasi Log Loss**:\n  $$J(\\mathbf{w}) = -\\frac{1}{N} \\sum_{i=1}^N \\left[ y_i \\ln(\\hat{p}_i) + (1 - y_i) \\ln(1 - \\hat{p}_i) \\right]$$\n- **Penalti Asimtotik Ekstrem**: Jika label aktual adalah $y=1$ tetapi model memprediksi $\\hat{p} \\to 0$, penalti $-\\ln(\\hat{p})$ melonjak menuju tak terhingga ($+\\infty$).",
        "notes": "Tunjukkan bahwa Cross-Entropy adalah fungsi konveks yang menjamin konvergensi ke minimum global."
    },
    {
        "kind": "split",
        "title": "Multiclass Logistic Regression",
        "body": "Ekspansi regresi logistik ke banyak kelas ($K > 2$):\n\n### Fungsi Softmax (Multinomial)\n$$P(y = k \\mid \\mathbf{x}) = \\frac{e^{\\mathbf{w}_k^T \\mathbf{x} + b_k}}{\\sum_{j=1}^K e^{\\mathbf{w}_j^T \\mathbf{x} + b_j}}$$\n- Menjamin penjumlahan seluruh probabilitas tepat bernilai $1.0$.\n\n### Strategi One-vs-Rest (OvR)\n- Melatih $K$ model biner independen; memilih kelas dengan skor tertinggi.",
        "notes": "Softmax adalah generalisasi langsung dari Sigmoid untuk K kelas."
    },
    {
        "kind": "content",
        "title": "Naive Bayes Classifier",
        "body": "Klasifikasi probabilistik berbasis Teorema Bayes:\n\n- **Formula Dasar**:\n  $$P(y \\mid X) = \\frac{P(X \\mid y) \\cdot P(y)}{P(X)} \\propto P(y) \\cdot P(X \\mid y)$$\n- **Komponen Kunci**:\n  - $P(y)$ (**Prior**): Prevalensi kelas di masa lalu.\n  - $P(X \\mid y)$ (**Likelihood**): Peluang sensor menghasilkan pola $X$ jika status mesin diketahui $y$.\n  - $P(y \\mid X)$ (**Posterior**): Derajat keyakinan status mesin setelah melihat pembacaan sensor.",
        "notes": "Ingatkan kaitan dengan Teorema Bayes dari Bab 06."
    },
    {
        "kind": "split",
        "title": "Conditional Independence",
        "body": "Asumsi penyederhanaan Naive Bayes:\n\n### Asumsi \"Naive\"\n- Seluruh fitur dianggap saling bebas bersyarat terhadap label kelas:\n  $$P(x_1, \\dots, x_D \\mid y) = \\prod_{j=1}^D P(x_j \\mid y)$$\n\n### Ketangguhan Praktis\n- Meskipun asumsi ini jarang terpenuhi murni di dunia nyata (suhu dan getaran saling berkorelasi),\n- Naive Bayes terbukti sangat cepat dan berkinerja luar biasa di lapangan.",
        "notes": "Jelaskan mengapa model sederhana yang salah asumsinya tetap bisa menghasilkan klasifikasi yang tepat."
    },
    {
        "kind": "split",
        "title": "Gaussian vs Multinomial Naive Bayes",
        "body": "Menyesuaikan distribusi likelihood dengan tipe data:\n\n### Gaussian Naive Bayes (Kontinu)\n- Untuk sensor numerik kontinu (suhu, bar, RPM).\n- Likelihood berdistribusi normal: $P(x_j \\mid y) \\sim \\mathcal{N}(\\mu_{yj}, \\sigma_{yj}^2)$.\n\n### Multinomial Naive Bayes (Diskrit)\n- Untuk frekuensi alarm atau kemunculan kata.\n- Menggunakan Laplace Smoothing ($\\alpha = 1.0$) untuk mencegah probabilitas nol.",
        "notes": "Tekankan perbedaan pemilihan model berdasarkan tipe data sensor."
    },
    {
        "kind": "content",
        "title": "Confusion Matrix",
        "body": "Tabel kontingensi evaluasi performa klasifikasi:\n\n- **True Positive (TP)**: Kerusakan mesin aktual yang berhasil dideteksi sistem sebagai kerusakan.\n- **True Negative (TN)**: Mesin normal yang benar diidentifikasi oleh sistem sebagai mesin normal.\n- **False Positive (FP / Type I Error)**: Alarm palsu; mesin normal yang salah dituduh mengalami kerusakan.\n- **False Negative (FN / Type II Error)**: Bencana operasional; mesin rusak yang gagal terdeteksi dan lolos sebagai normal.\n- **Aturan Urutan Label**: Perhatikan konvensi indeks label (0 vs 1) pada Scikit-learn agar tidak salah menafsirkan susunan matriks.",
        "notes": "Jelaskan struktur kuadran matriks konfusi secara perlahan kepada mahasiswa."
    },
    {
        "kind": "split",
        "title": "Accuracy and The Accuracy Paradox",
        "body": "Metrik akurasi dan jebakan fatalnya:\n\n### Formulasi Akurasi\n$$\\text{Accuracy} = \\frac{\\text{TP} + \\text{TN}}{\\text{TP} + \\text{TN} + \\text{FP} + \\text{FN}}$$\n\n### Paradoks Akurasi (The Paradox)\n- Jika dari 1000 mesin terdapat 990 Normal dan 10 Rusak:\n- Model bodoh yang selalu menebak \"Normal\" mendapat akurasi **99.0%**!\n- Namun **Recall = 0.0%** (semua mesin rusak lolos tanpa terdeteksi!).",
        "notes": "Tegaskan larangan menggunakan akurasi pada kasus penipuan atau kerusakan pabrik."
    },
    {
        "kind": "split",
        "title": "Precision",
        "body": "Presisi dan pengendalian alarm palsu:\n\n### Formulasi Presisi\n$$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$$\n\n### Makna Bisnis\n- Dari seluruh alarm bahaya yang dibunyikan model, berapa persen yang benar-benar rusak?\n- Sangat krusial saat biaya menangani alarm palsu (FP) sangat mahal.",
        "notes": "Presisi menjawab: 'Seberapa akurat alarm yang berbunyi?'"
    },
    {
        "kind": "split",
        "title": "Recall",
        "body": "Recall (Sensitivitas / True Positive Rate):\n\n### Formulasi Recall\n$$\\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$\n\n### Makna Bisnis\n- Dari seluruh mesin yang benar-benar rusak di pabrik, berapa persen yang berhasil ditangkap?\n- Mutlak diutamakan pada sistem keselamatan jiwa dan kegagalan katastropik.",
        "notes": "Recall menjawab: 'Seberapa banyak kerusakan yang berhasil kita tangkap?'"
    },
    {
        "kind": "content",
        "title": "F1-Score",
        "body": "Rata-rata harmonik antara Presisi dan Recall:\n\n$$\\text{F1-Score} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2\\text{TP}}{2\\text{TP} + \\text{FP} + \\text{FN}}$$\n\n- **Mengapa Rata-rata Harmonik?**: Rata-rata aritmatika bersikap terlalu ramah jika salah satu metrik hancur ($P=1.0, R=0.0 \\implies \\text{Mean}=0.5$).\n- **Penalti Ekstrem F1**: Rata-rata harmonik langsung jatuh ke nilai $0.0$ jika salah satu metrik bernilai nol, menjadikannya hakim yang sangat adil.",
        "notes": "Tunjukkan sifat matematika rata-rata harmonik yang sensitif terhadap nilai terendah."
    },
    {
        "kind": "split",
        "title": "Class Imbalance in Classification",
        "body": "Tantangan ketidakseimbangan kelas populasi:\n\n### Karakteristik Masalah\n- Mesin normal melimpah ($70-99\\%$), insiden kerusakan langka ($1-30\\%$).\n- Gradien optimasi standar cenderung mengabaikan kelas minoritas.\n\n### Solusi Metodologis\n- Gunakan `class_weight='balanced'` pada Logistic Regression.\n- Geser ambang keputusan pada Validation Set.\n- Gunakan kurva Precision-Recall sebagai acuan utama evaluasi.",
        "notes": "Hubungkan dengan materi mitigasi class imbalance dari Bab 12."
    },
    {
        "kind": "split",
        "title": "Multiclass Averaging Strategies",
        "body": "Merangkum metrik multikelas ($K > 2$):\n\n### Macro Average\n- Rata-rata sederhana skor per-kelas tanpa memandang ukuran sampel.\n- Memberi bobot **setara** pada kelas langka Electrical Surge.\n\n### Weighted Average\n- Membobot skor kelas berdasarkan proporsi sampel (*support*).\n- Didominasi oleh kinerja kelas mayoritas Normal.",
        "notes": "Tegaskan bahwa mahasiswa wajib menyatakan metode averaging yang digunakan saat melaporkan F1 multikelas."
    },
    {
        "kind": "content",
        "title": "ROC Curve",
        "body": "Kurva Receiver Operating Characteristic (ROC):\n\n- **Sumbu-Y**: True Positive Rate (Recall / Sensitivitas).\n- **Sumbu-X**: False Positive Rate ($1 - \\text{Spesifisitas}$).\n- **Evaluasi Multi-Threshold**: Menggambarkan kapabilitas pemisahan model di seluruh rentang ambang keputusan ($T \\in [0, 1]$).\n- **Garis Diagonal Acak**: Garis $(0,0)$ ke $(1,1)$ adalah tebakan lemparan koin acak murni (AUC = $0.5$).\n- **Model Ideal**: Melengkung tajam mendekati pojok kiri atas $(0, 1)$.",
        "notes": "Jelaskan asal-usul kurva ROC dari teknologi radar militer Perang Dunia II."
    },
    {
        "kind": "split",
        "title": "ROC-AUC Score",
        "body": "Area Under the ROC Curve (ROC-AUC):\n\n### Makna Probabilistik\n- Nilai AUC persis sama dengan probabilitas bahwa sampel positif acak akan diberi skor risiko lebih tinggi dibanding sampel negatif acak:\n  $$\\text{AUC} = P(\\text{skor}_{\\text{positif}} > \\text{skor}_{\\text{negatif}})$$\n\n### Peringatan Kritis\n- **Wajib menggunakan probabilitas kontinu** (`predict_proba`).\n- Prediksi biner diskrit mereduksi kurva ROC menjadi hanya satu titik patah!",
        "notes": "Tunjukkan hasil eksperimen script roc_auc_analysis.py."
    },
    {
        "kind": "split",
        "title": "Precision-Recall Curve",
        "body": "Kurva Precision-Recall (PR) untuk data tidak seimbang:\n\n### Keunggulan Atas ROC\n- Kurva ROC dapat tampak optimis palsu jika jumlah True Negative sangat besar.\n- Kurva PR berfokus murni pada kelas minoritas anomali.\n\n### Baseline Prevalensi\n- Baseline acak kurva PR adalah **prevalensi kelas positif** ($P / N$), bukan $0.5$.\n- Area di bawah kurva PR disebut **Average Precision (AP)**.",
        "notes": "Gunakan PR curve saat mengevaluasi kasus anomali yang sangat langka (<10%)."
    },
    {
        "kind": "content",
        "title": "Threshold Selection",
        "body": "Memilih ambang batas keputusan berdasarkan analisis biaya bisnis:\n\n- **Fungsi Kerugian Finansial**:\n  $$\\text{Biaya Total} = (\\text{FN} \\times C_{\\text{FN}}) + (\\text{FP} \\times C_{\\text{FP}})$$\n- **Studi Kasus Keselamatan Pabrik**:\n  - Biaya mesin rusak mendadak ($C_{\\text{FN}}$) = $\$50.000$.\n  - Biaya inspeksi pencegahan teknisi ($C_{\\text{FP}}$) = $\$200$.\n- **Keputusan Optimal**: Turunkan ambang ke $T = 0.20$. Menghasilkan sedikit kenaikan alarm palsu, tetapi mengeliminasi kerusakan katastropik secara total.",
        "notes": "Tunjukkan bahwa data science sejati harus berakar pada mitigasi dampak finansial dan keselamatan kerja."
    },
    {
        "kind": "split",
        "title": "Python Demonstration",
        "body": "Membangun Pipeline klasifikasi bebas kebocoran statistik:\n\n```python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\npipe = Pipeline([\n    ('scaler', StandardScaler()),\n    ('clf', LogisticRegression(class_weight='balanced'))\n])\npipe.fit(X_train, y_train)\nprobs = pipe.predict_proba(X_test)[:, 1]\n```\n\n- Otomatis menstandarisasi fitur secara konsisten.\n- Siap diekspor sebagai artefak produksi.",
        "notes": "Ingatkan kembali aturan emas pra-pemrosesan dari Bab 12."
    },
    {
        "kind": "content",
        "title": "Case Study: Industrial Fault Diagnosis",
        "body": "Hasil pengujian model klasifikasi pada 350 rekaman sensor pabrik:\n\n- **Dummy Baseline (Mayoritas)**: Akurasi semu $68.57\\%$, namun **Recall $= 0.0000$** (gagal mendeteksi kerusakan).\n- **Logistic Regression (Softmax)**: Akurasi $98.57\\%$, Macro F1 $0.9530$, ROC-AUC $1.0000$.\n- **Gaussian Naive Bayes**: Akurasi $100.00\\%$, ROC-AUC $1.0000$, inferensi ultra-cepat.\n- **Deteksi Multikelas**: 69 dari 70 sampel uji terdiagnosis secara presisi hingga ke tipe kerusakan spesifik.",
        "notes": "Rangkum pencapaian empiris dari skrip praktikum hari ini."
    },
    {
        "kind": "content",
        "title": "Responsible Classification",
        "body": "Prinsip etika dan tanggung jawab dalam sistem klasifikasi cerdas:\n\n- **Akuntabilitas Operator**: Sistem AI adalah alat bantu diagnostik (*decision-support*); tanggung jawab akhir keselamatan tetap pada insinyur manusia (*Human-in-the-Loop*).\n- **Hak atas Penjelasan (Explainability)**: Operator berhak mengetahui fitur mana (getaran atau lonjakan arus) yang memicu alarm bahaya.\n- **Audit Keadilan & Keselamatan**: Memastikan sistem tidak bias dan memantau degradasi model (*concept drift*) secara berkala.",
        "notes": "Ingatkan kaitan dengan prinsip AI Ethics dari Bab 09."
    },
    {
        "kind": "split",
        "title": "Discussion & Questions",
        "body": "Pertanyaan reflektif untuk penguasaan materi:\n\n### Diskusi Kelas\n1. Kapan akurasi $95\\%$ justru menandakan bahwa model Anda tidak berguna sama sekali?\n2. Mengapa kurva Precision-Recall lebih disukai daripada ROC pada kasus deteksi kerusakan langka?\n3. Mengapa menaikkan ambang keputusan dari $0.5$ ke $0.8$ akan menurunkan Recall tetapi menaikkan Presisi?",
        "notes": "Buka sesi diskusi interaktif dengan mahasiswa."
    },
    {
        "kind": "content",
        "title": "Summary & References",
        "body": "Kesimpulan pembelajaran Chapter 14:\n\n- **Klasifikasi** memprediksi label kategori diskrit melalui batas keputusan linier atau non-linier.\n- **Regresi Logistik** memodelkan probabilitas via fungsi Sigmoid dan log-odds.\n- **Naive Bayes** memanfaatkan Teorema Bayes dengan asumsi independensi fitur dan Laplace Smoothing.\n- **Metrik Evaluasi** bertumpu pada Matriks Konfusi (Precision, Recall, F1, ROC-AUC, PR-AUC).\n- **Referensi Otoritatif**:\n  - Géron, A. (2022). *Hands-On Machine Learning*, 3rd Edition.\n  - Hastie et al. (2009). *The Elements of Statistical Learning*, 2nd Edition.\n  - Russell & Norvig (2020). *Artificial Intelligence: A Modern Approach*, 4th Edition.\n  - CS2023 ACM/IEEE-CS/AAAI (2024). *Artificial Intelligence Area*.",
        "notes": "Tutup perkuliahan dan hubungkan dengan bab berikutnya tentang Decision Trees dan Ensemble Learning."
    }
]


def main():
    deck_path = Path("content/level-2-machine-learning/chapter-14/slides/deck.json")
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deck_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated {len(slides)} slides at: {deck_path}")


if __name__ == "__main__":
    main()
