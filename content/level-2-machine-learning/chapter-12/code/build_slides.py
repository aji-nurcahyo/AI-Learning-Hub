"""
Script builder untuk slides deck Chapter 12: deck.json (34 slide)
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "ML Workflow, Dataset & Feature Engineering",
        "body": "### Arsitektur Alur Kerja Data, Pencegahan Kebocoran & Rekayasa Fitur Siap Produksi\n\n**Level 2 — Machine Learning | Chapter 12**\n\nArtificial Intelligence Learning Hub — Institut Teknologi Sumatera (ITERA)\n*Created & Developed by Aji Pamungkas Tri Nurcahyo*",
        "notes": "Selamat datang di perkuliahan Chapter 12: ML Workflow, Dataset & Feature Engineering. Bab ini adalah pilar metodologis penentu dalam Level 2, di mana kita membahas cara merancang alur rekayasa data yang disiplin, aman dari kebocoran informasi, dan siap dideploy ke sistem produksi."
    },
    {
        "kind": "content",
        "title": "Why Workflow Matters",
        "body": "Mengapa alur kerja data terstruktur (*workflow*) sangat krusial?\n\n- **Beban Teknis Tersembunyi (*Hidden Technical Debt*)**: Riset Google membuktikan kode matematika pemodelan hanya menyumbang <10% dari basis kode sistem ML industri; sisanya adalah rekayasa data.\n- **Krisis Reproducibility**: Eksperimen yang tidak memiliki alur pipeline baku gagal direplikasi saat berpindah lingkungan dari komputer lokal ke server.\n- **Degradasi Performa di Produksi**: Tanpa standarisasi pra-pemrosesan, performa model anjlok drastis akibat data drift dan inkonsistensi kalkulasi fitur.\n- **Pencegahan Data Leakage**: Alur kerja yang disiplin adalah satu-satunya benteng pertahanan terhadap optimisme evaluasi palsu.",
        "notes": "Tekankan bahwa data scientist profesional dinilai dari keandalan arsitektur alur kerja datanya, bukan sekadar jumlah algoritma yang ia hafal."
    },
    {
        "kind": "content",
        "title": "Capaian Pembelajaran (Objectives)",
        "body": "Setelah menyelesaikan bab ini, mahasiswa diharapkan mampu:\n\n- **Memahami** tahapan end-to-end Machine Learning workflow dari perumusan masalah hingga pemantauan.\n- **Membagi** dataset menjadi train, validation, dan test set dengan proporsi dan strategi yang tepat.\n- **Mengidentifikasi dan mencegah** berbagai bentuk data leakage (pre-split scaling, temporal bias).\n- **Melakukan** pra-pemrosesan data numerik dan kategorikal secara metodologis menggunakan data train.\n- **Mengembangkan** fitur domain baru, interaksi non-linear, dan fitur temporal.\n- **Menangani** ketidakseimbangan kelas (class imbalance) secara bertanggung jawab.\n- **Menerapkan** validasi silang (cross-validation) untuk estimasi generalisasi yang reliabel.\n- **Membangun** pipeline modular menggunakan Scikit-learn Pipeline dan ColumnTransformer.",
        "notes": "Rincikan 10 capaian pembelajaran ini sebagai peta jalan kuliah hari ini."
    },
    {
        "kind": "content",
        "title": "End-to-End ML Lifecycle",
        "body": "Siklus hidup lengkap rekayasa Machine Learning di industri:\n\n- **Hulu (Upstream)**:\n  - Perumusan sasaran bisnis terukur (*Problem Formulation*).\n  - Pengumpulan data mentah terstandarisasi (*Data Ingestion*).\n  - Eksplorasi data awal & audit integritas (*Exploratory Data Analysis*).\n- **Tengah (Midstream)**:\n  - Isolasi partisi data tanpa kebocoran (*Dataset Splitting*).\n  - Pra-pemrosesan numerik & kategorikal (*Preprocessing*).\n  - Rekayasa fitur domain (*Feature Engineering*).\n- **Hilir (Downstream)**:\n  - Pelatihan model & validasi silang (*Training & Cross-Validation*).\n  - Analisis kesalahan & evaluasi akhir (*Model Evaluation*).\n  - Pengemasan pipeline & pemantauan performa (*Deployment & Monitoring*).",
        "notes": "Gambarkan alur bolak-balik: temuan pada tahap pemodelan sering kali menuntut kembali ke tahap rekayasa fitur."
    },
    {
        "kind": "split",
        "title": "CRISP-DM Framework",
        "body": "Kerangka kerja standar industri untuk proyek AI/ML:\n\n### Tahapan Utama CRISP-DM\n- **Business Understanding**: Menetapkan target bisnis dan kriteria sukses terukur.\n- **Data Understanding**: Audit ketersediaan, kualitas, dan distribusi data mentah.\n- **Data Preparation**: Pembersihan data, imputasi, dan rekayasa fitur (70-80% waktu proyek).\n\n### Siklus Evaluasi & Rilis\n- **Modeling**: Melatih berbagai kandidat model dan optimasi hyperparameter.\n- **Evaluation**: Menguji generalisasi model terhadap tujuan bisnis awal.\n- **Deployment**: Mengemas pipeline ke sistem produksi.",
        "notes": "Jelaskan bahwa CRISP-DM bukan alur air terjun (waterfall) linier kaku, melainkan siklus berulang."
    },
    {
        "kind": "split",
        "title": "Problem Formulation: Business to ML",
        "body": "Menerjemahkan bahasa operasional menjadi perumusan matematis:\n\n### Pertanyaan Bisnis\n- *\"Bagaimana mencegah mesin pabrik mati mendadak saat jam produksi sibuk?\"*\n- *\"Kapan komponen hidrolik harus diservis secara preventif?\"*\n\n### Rumusan Machine Learning\n- **Tipe Tugas**: Klasifikasi biner terbobot (*Cost-Sensitive Binary Classification*).\n- **Variabel Target ($y$)**: Status kegagalan dalam 24 jam ke depan ($y \\in \\{0, 1\\}$).\n- **Matriks Fitur ($X$)**: Nilai getaran RMS, suhu, tekanan fluida, dan total jam operasi.",
        "notes": "Tunjukkan kesalahan umum pemula yang langsung menulis kode model sebelum merumuskan target prediktif secara jelas."
    },
    {
        "kind": "content",
        "title": "Data Ingestion & Collection",
        "body": "Strategi pengumpulan data mentah untuk sistem cerdas:\n\n- **Batch Ingestion**: Pemuatan data terjadwal (harian/mingguan) dari gudang data (*Data Warehouse*) berbasis Apache Spark atau SQL tables.\n- **Streaming Ingestion**: Pemuatan aliran data berkecepatan tinggi secara real-time dari sensor IoT menggunakan message broker (Kafka, MQTT).\n- **Validasi Skema (Schema Enforcement)**: Menolak data masuk yang tidak sesuai tipe data atau memiliki nilai di luar batas kewajaran fisik.\n- **Pencatatan Metadata**: Setiap sampel wajib disertai stempel waktu (*timestamp*) dan tanda pengenal perangkat (*device ID*) untuk audit rekayasa.",
        "notes": "Ingatkan bahwa data yang masuk ke model di masa depan akan memiliki kualitas yang sama buruknya dengan saat data tersebut dikumpulkan jika tidak ada validasi skema."
    },
    {
        "kind": "content",
        "title": "Exploratory Data Analysis (EDA) for ML",
        "body": "EDA sebelum pemodelan berfokus pada empat pemeriksaan wajib:\n\n- **Peta Missing Values**: Identifikasi persentase nilai kosong tiap kolom; tentukan apakah mekanisme kehilangan bersifat MCAR, MAR, atau MNAR.\n- **Kardinalitas Kategorikal**: Hitung jumlah kelas unik; variabel berkardinalitas tinggi (>100 unik) membutuhkan perlakuan encoding khusus.\n- **Bentuk Distribusi & Outlier**: Analisis kemencengan (*skewness*); fitur berdistribusi long-tail memerlukan transformasi logaritmik.\n- **Korelasi & Multikolinearitas**: Hitung korelasi linear Pearson dan nilai VIF (*Variance Inflation Factor*) untuk menghindari redundansi fitur.",
        "notes": "Tegaskan perbedaan EDA untuk pelaporan bisnis vs EDA untuk kesiapan pemodelan prediktif."
    },
    {
        "kind": "content",
        "title": "Why Split the Dataset?",
        "body": "Mengapa kita wajib membagi dataset menjadi beberapa partisi?\n\n- **Filosofi Generalisasi**: Tujuan utama Machine Learning adalah berkinerja unggul pada data baru yang **belum pernah dilihat sebelumnya** (*unseen data*).\n- **Bahaya Evaluasi pada Data Latih**: Menilai model pada data yang digunakan untuk melatihnya sama seperti menguji siswa dengan soal ujian yang persis sama dengan lembar jawaban latihan—hanya mengukur hafalan, bukan pemahaman.\n- **Deteksi Overfitting**: Perbedaan performa antara data latih dan data uji adalah indikator utama gejala overfitting.",
        "notes": "Gunakan analogi ujian sekolah untuk memperjelas konsep generalisasi."
    },
    {
        "kind": "split",
        "title": "Train, Validation, Test Split",
        "body": "Tiga partisi independen dalam Machine Learning profesional:\n\n### 1. Training Set (60%)\n- Digunakan algoritma untuk mengoptimasi parameter internal (bobot $w$, bias $b$).\n\n### 2. Validation Set (20%)\n- Digunakan insinyur data untuk tuning hyperparameter dan seleksi model terbaik.\n\n### 3. Test Set (20%)\n- **Hanya disentuh satu kali** di akhir untuk mengukur performa final secara objektif.",
        "notes": "Tekankan larangan mutlak menyentuh Test Set selama proses pemilihan model."
    },
    {
        "kind": "content",
        "title": "Splitting Ratios: Rules of Thumb",
        "body": "Pedoman proporsi pembagian dataset berdasarkan skala data ($N$):\n\n- **Dataset Kecil ($N < 1.000$)**:\n  - Proporsi: $70\\% / 15\\% / 15\\%$ atau gunakan $k$-Fold Cross-Validation.\n  - Alasan: Setiap sampel sangat berharga; CV meminimalkan variasi evaluasi.\n- **Dataset Menengah ($1.000 \\le N \\le 100.000$)**:\n  - Proporsi: $60\\% / 20\\% / 20\\%$ (Standar Emas Industri).\n  - Alasan: Memberikan volume sampel representatif untuk validasi dan uji statistik.\n- **Dataset Sangat Besar ($N > 1.000.000$)**:\n  - Proporsi: $98\\% / 1\\% / 1\\%$.\n  - Alasan: $1\\%$ dari sejuta data sudah menyediakan $10.000$ sampel uji yang kokoh.",
        "notes": "Jelaskan bahwa rasio 80/20 tidak selalu baku; skala data menentukan kebutuhan representasi statistik."
    },
    {
        "kind": "split",
        "title": "Random Splitting",
        "body": "Pembagian acak sederhana (*Simple Random Splitting*):\n\n### Konsep & Karakteristik\n- Membagi baris data menggunakan generator bilangan acak semu (`random_state`).\n- Cepat, sederhana, dan didukung penuh oleh `sklearn.model_selection.train_test_split`.\n\n### Batasan Kritis\n- Berasumsi data bersifat *Independent and Identically Distributed* ($i.i.d.$).\n- **Gagal** pada dataset dengan ketidakseimbangan kelas ekstrem.\n- **Gagal** pada data deret waktu (memicu time leakage).\n- **Gagal** pada data kelompok entitas (memicu group leakage).",
        "notes": "Tunjukkan kapan random splitting aman digunakan (data i.i.d seimbang) dan kapan dilarang."
    },
    {
        "kind": "content",
        "title": "Stratified Splitting",
        "body": "Menjaga proporsi kelas target pada data tidak seimbang (*imbalanced*):\n\n- **Distribusi Target Asli**: Misal populasi memiliki 89.2% normal dan 10.8% kegagalan.\n- **Hasil Random Split Bias**: Train dapat memiliki 12.0% kegagalan, sementara Test hanya 6.0% kegagalan (terjadi distorsi evaluasi parah).\n- **Hasil Stratified Split**: Mempertahankan tepat 10.8% kegagalan pada Train, Validation, dan Test set secara konsisten.\n- **Implementasi**: `StratifiedKFold` atau parameter `stratify=y` pada `train_test_split`.\n- **Manfaat Utama**: Menjamin setiap lipatan evaluasi memiliki jumlah representatif sampel kelas minoritas.",
        "notes": "Hubungkan dengan hasil aktual skrip stratified_split.py di mana variansi proporsi turun drastis."
    },
    {
        "kind": "split",
        "title": "Group-Based Splitting",
        "body": "Mencegah kebocoran lintas entitas fisik (*Group Leakage*):\n\n### Bahaya Naive Split\n- Rekaman sensor Mesin Alpha muncul di data latih dan data uji.\n- Model menghafal sidik jari kebisingan Mesin Alpha, bukan pola kegagalan umum (*pseudoreplication*).\n\n### Solusi Group-Aware Split\n- `GroupShuffleSplit` & `GroupKFold` membagi berdasarkan entitas `machine_id`.\n- Mesin di data uji **belum pernah dilihat sama sekali** pada data latih.\n- Menjamin model mampu beroperasi saat pabrik menambah unit mesin baru.",
        "notes": "Gunakan contoh rekam medis pasien: jangan pernah memecah data pasien yang sama ke train dan test."
    },
    {
        "kind": "content",
        "title": "Time-Series Splitting",
        "body": "Pemisahan data berurutan waktu untuk mengeliminasi *Look-Ahead Bias*:\n\n- **Look-Ahead Bias**: Terjadi jika data masa depan bocor untuk memprediksi masa lalu akibat pengacakan acak (`shuffle=True`).\n- **Prinsip Temporal**: Model hanya boleh dilatih pada data hingga waktu $t$ dan dievaluasi pada data periode masa depan ($t+1$ hingga $t+k$).\n- **Struktur Expanding Window**:\n  - Fold 1: Train $[0, t_1] \\to$ Test $(t_1, t_2]$\n  - Fold 2: Train $[0, t_2] \\to$ Test $(t_2, t_3]$\n  - Fold 3: Train $[0, t_3] \\to$ Test $(t_3, t_4]$\n- **Implementasi**: `sklearn.model_selection.TimeSeriesSplit`.",
        "notes": "Tekankan bahwa di dunia nyata kita tidak pernah bisa melatih model menggunakan data besok."
    },
    {
        "kind": "content",
        "title": "What Is Data Leakage?",
        "body": "Pengertian dan bahaya fatal kebocoran data (*Data Leakage*):\n\n- **Definisi**: Kondisi patologis di mana informasi dari luar data latih (*out-of-sample data*) secara tidak sah merembes ke dalam proses pelatihan model.\n- **Ilusi Performa**: Menghasilkan metrik pelatihan dan validasi yang tampak luar biasa tinggi (misal akurasi 98%).\n- **Bencana Produksi**: Saat sistem dideploy di lapangan, model mengalami kegagalan prediksi total karena informasi bocor tersebut tidak tersedia di dunia nyata.\n- **Kerugian Nyata**: Menyebabkan penarikan sistem, pemborosan biaya operasional, dan hilangnya kepercayaan pemangku kepentingan.",
        "notes": "Jelaskan bahwa data leakage adalah musuh nomor satu insinyur machine learning di industri."
    },
    {
        "kind": "content",
        "title": "Common Forms of Data Leakage",
        "body": "Empat bentuk kebocoran data yang paling sering terjadi:\n\n- **1. Pre-Split Preprocessing**: Menghitung mean, median, min, max pada seluruh dataset sebelum partisi train/test dipisahkan.\n- **2. Target Leakage**: Menggunakan fitur yang baru tercipta *setelah* kejadian target terjadi (misal: fitur 'surat tugas teknisi perbaikan' untuk memprediksi mesin rusak).\n- **3. Feature Engineering Leakage**: Melakukan agregasi global atau normalisasi frekuensi lintas seluruh partisi data.\n- **4. Temporal / Lookahead Leakage**: Mengabaikan kronologi waktu pada data deret waktu sehingga masa depan bocor ke masa lalu.",
        "notes": "Bahas contoh target leakage yang sering lolos dalam tinjauan kode pemula."
    },
    {
        "kind": "split",
        "title": "Golden Rule: Fit on Train Only",
        "body": "Aturan emas pra-pemrosesan data dalam Machine Learning:\n\n### Hukum Mutlak Pemanggilan Fungsi\n- **`.fit()` / `.fit_transform()`**: **HANYA** boleh dipanggil pada **TRAINING SET**!\n- **`.transform()`**: Dipanggil pada Training Set, Validation Set, dan Test Set.\n\n### Konsekuensi Matematis\n- Statistik pra-pemrosesan (rata-rata $\\mu$, simpangan baku $\\sigma$, nilai median, daftar kategori unik) harus terkunci murni dari data latih.\n- Data validasi dan data uji diperlakukan sebagai simulasi masa depan yang sepenuhnya misterius.",
        "notes": "Ingatkan mahasiswa: satu kali saja Anda memanggil fit() pada data uji, seluruh hasil evaluasi Anda batal demi hukum ilmiah."
    },
    {
        "kind": "content",
        "title": "Numeric Preprocessing: Imputation",
        "body": "Penanganan nilai hilang (*missing values*) pada atribut numerik:\n\n- **Mean Imputation**: Mengganti NaN dengan nilai rata-rata data latih. Tepat jika data berdistribusi normal simetris tanpa pencilan.\n- **Median Imputation**: Mengganti NaN dengan nilai tengah data latih. Sangat disarankan untuk data industri yang rawan lonjakan ekstrem (*robust*).\n- **Constant Imputation**: Mengisi dengan konstanta penanda khusus (misal 0 atau -999) yang memiliki signifikansi domain tertentu.\n- **KNN / Iterative Imputer**: Memodelkan nilai hilang sebagai fungsi multivariat dari fitur-fitur lain yang tersedia.",
        "notes": "Jelaskan mengapa SimpleImputer(strategy='median') adalah opsi default paling aman untuk telemetri sensor."
    },
    {
        "kind": "split",
        "title": "Feature Scaling: Standard vs MinMax vs Robust",
        "body": "Menyamakan rentang skala variabel numerik:\n\n### 1. StandardScaler (Z-Score)\n- Rumus: $z = (x - \\mu) / \\sigma$.\n- Rata-rata menjadi 0, simpangan baku menjadi 1.\n\n### 2. MinMaxScaler\n- Rumus: $x' = (x - x_{\\min}) / (x_{\\max} - x_{\\min})$.\n- Memetakan ke interval $[0, 1]$; rentan rusak oleh outlier.\n\n### 3. RobustScaler\n- Rumus: $x' = (x - \\text{median}) / IQR$.\n- Menggunakan kuartil; kebal terhadap pencilan ekstrem.",
        "notes": "Tegaskan mengapa model berbasis gradien (Regresi Logistik, Neural Net) membutuhkan penskalaan, sedangkan pohon keputusan kebal terhadap skala."
    },
    {
        "kind": "content",
        "title": "Distribution Transformations",
        "body": "Mentransformasikan distribusi menceng (*skewed*) menjadi normal:\n\n- **Log Transformation**: $x' = \\log(x + c)$. Efektif untuk distribusi miring ke kanan dengan ekor panjang (*heavy right-tailed*).\n- **Box-Cox Transformation**: Transformasi berbasis parameter daya $\\lambda$; menyaratkan seluruh nilai data bernilai positif murni ($x > 0$).\n- **Yeo-Johnson Transformation**: Perluasan dari Box-Cox yang mampu menangani nilai nol dan nilai negatif.\n- **Manfaat**: Memenuhi asumsi linearitas dan homoskedastisitas pada model parametrik, mempercepat konvergensi gradien.",
        "notes": "Perlihatkan kurva histogram sebelum dan sesudah transformasi log."
    },
    {
        "kind": "split",
        "title": "Outlier Handling Strategies",
        "body": "Tiga strategi menangani pencilan tanpa membuang data berharga:\n\n### 1. Winsorizing (Clipping)\n- Membatasi nilai di luar batas persentil (misal persentil ke-1 dan ke-99).\n- Nilai di atas batas diganti nilainya tepat dengan nilai batas tersebut.\n\n### 2. Outlier Flagging (Masking)\n- Membuat fitur biner baru `is_outlier = 1` saat nilai melampaui $3\\sigma$.\n- Memberi sinyal eksplisit pada model tentang kondisi ekstrem.\n\n### 3. Pengecekan Fisika Domain\n- Membedakan antara kesalahan alat ukur (*error*) vs anomali nyata operasional.",
        "notes": "Ingatkan bahwa menghapus baris outlier dapat mengubah proporsi kelas target pada data imbalanced."
    },
    {
        "kind": "split",
        "title": "Categorical Preprocessing: Nominal vs Ordinal",
        "body": "Membedakan jenis data kategorikal sebelum transformasi:\n\n### 1. Kategorikal Nominal\n- Tidak memiliki hierarki atau urutan intrinsik.\n- Contoh: Mode Pendingin (`ACTIVE`, `STANDBY`, `OFF`).\n- Metode yang tepat: **One-Hot Encoding**.\n\n### 2. Kategorikal Ordinal\n- Memiliki hierarki peringkat tingkatan yang bermakna.\n- Contoh: Tingkat Peringatan (`LOW` < `MEDIUM` < `HIGH`).\n- Metode yang tepat: **Ordinal Encoding** dengan urutan eksplisit.",
        "notes": "Jangan pernah menerapkan ordinal encoder secara acak pada data nominal karena model akan salah menyimpulkan besaran urutan."
    },
    {
        "kind": "content",
        "title": "One-Hot Encoding & The Dummy Trap",
        "body": "Representasi biner dan penanganan bahaya pada One-Hot Encoding:\n\n- **Representasi Vektor Biner**: Kategori dengan $K$ kelas diubah menjadi $K$ kolom bernilai 0 atau 1.\n- **Dummy Variable Trap**: Multikolinearitas sempurna jika seluruh $K$ kolom dipertahankan pada model linier tanpa regularisasi. Solusi: gunakan `drop='first'`.\n- **Ketahanan Terhadap Unseen Categories**:\n  - Di produksi, sensor mungkin memunculkan status baru yang belum pernah tercatat di data latih (misal `EMERGENCY_VENT`).\n  - Parameter `handle_unknown='ignore'` mengubah status asing tersebut menjadi baris semua-nol tanpa menghentikan sistem (*zero-crash*).",
        "notes": "Tunjukkan implementasi handle_unknown='ignore' pada sklearn OneHotEncoder."
    },
    {
        "kind": "content",
        "title": "Ordinal Encoding",
        "body": "Pemetaan terstruktur fitur bertingkat ke representasi bilangan bulat:\n\n- **Spesifikasi Hierarki**: Menyediakan daftar urutan kategori secara eksplisit, misal `[['LOW', 'MEDIUM', 'HIGH']]`.\n- **Integritas Semantik**: Memastikan `LOW` dipetakan ke 0, `MEDIUM` ke 1, dan `HIGH` ke 2.\n- **Mencegah Urutan Alfabetis Acak**: Tanpa spesifikasi eksplisit, pengurutan default alfabetis akan memetakan `HIGH`=0, `LOW`=1, `MEDIUM`=2 yang bertentangan dengan makna fisik.\n- **Penanganan Nilai Asing**: Konfigurasikan `handle_unknown='use_encoded_value'` dengan `unknown_value=-1` untuk fleksibilitas di produksi.",
        "notes": "Tekankan pentingnya menyuplai list urutan secara eksplisit ke dalam parameter categories."
    },
    {
        "kind": "content",
        "title": "Target Encoding (Mean Encoding)",
        "body": "Solusi untuk variabel kategorikal berkardinalitas tinggi (*high-cardinality*):\n\n- **Tantangan Kardinalitas Tinggi**: Atribut seperti ID Kota (500 kategori) akan menghasilkan 500 kolom sparse jika di-OHE, memicu lonjakan dimensi komputasi.\n- **Mekanisme Target Encoding**: Mengganti kategori dengan rata-rata nilai target ($y$) dari kategori tersebut pada data latih.\n- **Formula Smoothing M-Estimate**:\n  $$\\hat{x}_c = \\frac{n_c \\cdot \\bar{y}_c + m \\cdot \\bar{y}_{\\text{global}}}{n_c + m}$$\n- **Peringatan Keras**: Target encoding rawan memicu kebocoran data (*target leakage*); wajib dihitung secara terisolasi di dalam tiap lipatan CV.",
        "notes": "Jelaskan peran parameter m dalam mencegah kategori beranggotakan 1 sampel mengalami overfitting ekstrem."
    },
    {
        "kind": "split",
        "title": "Feature Engineering Principles",
        "body": "Prinsip dasar rekayasa fitur (*Feature Engineering*):\n\n### Filosofi \"Garbage In, Garbage Out\"\n- Model tercanggih sekalipun tidak dapat memprediksi jika fitur masukan tidak memiliki korelasi informasi terhadap target.\n\n### Peran Domain Knowledge\n- Algoritma hanya menghitung optimasi numerik; pemahaman proses fisik dan bisnis berasal dari insinyur yang merancang fitur.\n- Fitur yang baik mempermudah model menemukan batas keputusan (*decision boundaries*) yang elegan dan sederhana.",
        "notes": "Kutipan Andrew Ng: 'Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering.'"
    },
    {
        "kind": "content",
        "title": "Domain-Specific Feature Creation",
        "body": "Mentransformasikan pemahaman fisika mesin menjadi fitur prediktif:\n\n- **Rasio Stres Mekanik-Tekanan**: $\\text{Vibration} / (\\text{Pressure} + \\epsilon)$. Lonjakan rasio ini menandakan ketidakstabilan fluida dan kavitasi pompa.\n- **Laju Kenaikan Suhu Relatif**: $\\text{Temperature} / (\\text{Operating Hours} / 1000)$. Suhu tinggi pada mesin baru adalah anomali manufaktur; pada mesin tua adalah keausan wajar.\n- **Ambang Batas Operasional Kritis**: Fitur biner penanda darurat: `(temperature > 75.0) & (warning == 'HIGH')`.\n- **Dampak Nyata**: Mengubah relasi non-linear yang rumit menjadi sinyal linier yang sangat mudah dipelajari oleh model sederhana.",
        "notes": "Hubungkan dengan hasil feature_engineering.py di mana korelasi target melonjak tinggi."
    },
    {
        "kind": "content",
        "title": "Temporal & Cyclical Features",
        "body": "Mengekstrak informasi prediktif dari dimensi waktu:\n\n- **Dekomposisi Kalender**: Mengekstrak jam (`hour`), hari dalam minggu (`dayofweek`), dan penanda akhir pekan (`is_weekend`).\n- **Transformasi Trigonometri Siklis**: Jam 23 dan jam 00 berdekatan secara fisik. Transformasi sinus-kosinus mempertahankan sifat sirkular waktu:\n  $$x_{\\sin} = \\sin(2\\pi \\cdot \\text{jam} / 24), \\quad x_{\\cos} = \\cos(2\\pi \\cdot \\text{jam} / 24)$$\n- **Fitur Lag & Rolling Window**: Mengukur tren jangka pendek:\n  - `lag_1`: Nilai sensor 1 jam sebelumnya.\n  - `rolling_mean_6h`: Rata-rata bergerak selama 6 jam terakhir.",
        "notes": "Tunjukkan diagram lingkaran 24 jam untuk menjelaskan transformasi sinus-kosinus."
    },
    {
        "kind": "split",
        "title": "Interaction Terms & Non-Linear Features",
        "body": "Menangkap sinyal sinergis antar-variabel:\n\n### Konsep Suku Interaksi\n- Di dunia nyata, kerusakan terjadi ketika dua kondisi buruk terjadi bersamaan.\n- Suku interaksi perkalian: $x_3 = x_1 \\times x_2$.\n\n### Indeks Kerusakan Termomekanikal\n- $\\text{Vibration} \\times \\text{Temperature}$.\n- Getaran sedang tidak merusak jika dingin; suhu tinggi tidak merusak jika diam.\n- Keduanya tinggi bersamaan menghasilkan kegagalan katastropik.",
        "notes": "Tunjukkan bagaimana perkalian dua fitur dapat memisahkan data pada scatter plot secara linear."
    },
    {
        "kind": "content",
        "title": "Class Imbalance & The Accuracy Paradox",
        "body": "Paradoks akurasi pada dataset tidak seimbang (*Class Imbalance*):\n\n- **Skenario Nyata**: Dari 1.000 mesin industri, 990 mesin normal (99%) dan 10 mesin mengalami kegagalan (1%).\n- **Model Naif (Dummy Predictor)**: Selalu memprediksi \"NORMAL\" untuk seluruh mesin.\n  - **Akurasi Mentah**: **99.0%** (Tampak sangat hebat!).\n  - **Recall Kegagalan**: **0.0%** (Semua mesin rusak lolos tanpa terdeteksi!).\n- **Metrik Wajib untuk Data Imbalanced**:\n  - **Precision**: Keakuratan sinyal alarm kegagalan.\n  - **Recall**: Daya tangkap sistem terhadap seluruh kegagalan nyata.\n  - **F1-Score & Balanced Accuracy**: Evaluasi performa yang seimbang.",
        "notes": "Tegaskan larangan menggunakan akurasi mentah pada kasus deteksi kerusakan atau penipuan."
    },
    {
        "kind": "split",
        "title": "Cost-Sensitive Learning & Threshold Tuning",
        "body": "Solusi metodologis menangani ketidakseimbangan kelas:\n\n### 1. Cost-Sensitive Learning\n- Memberikan penalti kesalahan lebih berat pada kelas minoritas.\n- Scikit-learn: `class_weight='balanced'`.\n- Bobot inversi: $w_k = N / (K \\cdot n_k)$.\n\n### 2. Decision Threshold Tuning\n- Default ambang probabilitas klasifikasi biner adalah 0.5.\n- Jika biaya melewatkan kerusakan sangat mahal, turunkan ambang ke 0.25 atau 0.30.\n- **Wajib dilakukan pada Validation Set**, bukan Test Set!",
        "notes": "Jelaskan mengapa pendekatan ini jauh lebih aman dan bersih dibanding melakukan SMOTE pra-split."
    },
    {
        "kind": "content",
        "title": "Cross-Validation Strategies",
        "body": "Strategi validasi silang untuk estimasi performa yang reliabel:\n\n- **$k$-Fold Standar**: Membagi data menjadi $k$ subset acak seragam. Baik untuk data seimbang $i.i.d$.\n- **Stratified $k$-Fold**: Menjamin proporsi kelas target identik pada setiap lipatan (*fold*). Standar baku untuk klasifikasi imbalanced.\n- **Group $k$-Fold**: Menjamin tidak ada entitas kelompok (`machine_id`, `patient_id`) yang terbagi ke dalam fold latih dan uji secara bersamaan.\n- **TimeSeriesSplit**: Validasi temporal dengan jendela mengembang (*expanding window*), menjamin masa lalu menguji masa depan.\n- **Analisis Varians**: Laporkan rata-rata dan deviasi standar (Mean $\\pm$ Std) lintas fold.",
        "notes": "Rangkum empat strategi CV dan kondisi kapan masing-masing wajib digunakan."
    },
    {
        "kind": "split",
        "title": "Scikit-learn Pipeline & ColumnTransformer",
        "body": "Arsitektur rekayasa data siap produksi (*Production-Ready*):\n\n### ColumnTransformer\n- Mengarahkan kolom numerik ke imputasi median + standardisasi.\n- Mengarahkan kolom nominal ke OneHotEncoder.\n- Mengarahkan kolom ordinal ke OrdinalEncoder.\n\n### Scikit-learn Pipeline\n- Mengikat ColumnTransformer dan Model Classifier ke dalam satu objek tunggal.\n- Menjamin **nol kebocoran data** (*zero-leakage*) secara otomatis.\n- Siap diekspor sebagai artefak inferensi produksi.",
        "notes": "Tunjukkan bahwa pipeline adalah standar emas arsitektur Machine Learning modern."
    }
]


def main():
    deck_path = Path("content/level-2-machine-learning/chapter-12/slides/deck.json")
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deck_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated {len(slides)} slides at: {deck_path}")


if __name__ == "__main__":
    main()
