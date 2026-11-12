"""
Chapter 30 - Deep Learning Capstone
Script Pembangkit Slide Presentasi Kuliah Reveal.js
Menghasilkan slides/deck.json dengan struktur 27 slide semantik terstandar.
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "Bab 30: Deep Learning Capstone Project",
        "body": "### Integrasi Teori, Rekayasa Komputasi, Evaluasi Diagnostik, dan Tata Kelola AI Bertanggung Jawab\n\n**Mata Kuliah**: Kecerdasan Buatan / Deep Learning  \n**Program Studi**: Teknik Informatika / Sains Komputer  \n**Pengembang**: Aji Pamungkas Tri Nurcahyo  \n**Institusi**: Institut Teknologi Sumatera",
        "notes": "Selamat datang di Bab 30: Proyek Capstone penutup resmi Level 3 Deep Learning & Generative AI. Pada sesi ini, kita mengintegrasikan seluruh teori dan keterampilan yang telah dipelajari dari Bab 21 hingga Bab 29 ke dalam karya rekayasa nyata yang terstandar."
    },
    {
        "kind": "content",
        "title": "Apa Itu Deep Learning Capstone?",
        "body": "- **Bukan Sekadar Kode**: Capstone adalah proyek komprehensif yang mensimulasikan siklus hidup rekayasa kecerdasan buatan tingkat industri.\n- **Integrasi Holistik**: Menghubungkan perumusan masalah, etika data, perancangan arsitektur, pelatihan konvergen, diagnosis galat, kepatuhan Responsible AI, hingga komunikasi ilmiah.\n- **Bebas Kompromi Ilmiah**: Menolak klaim performa tanpa pembuktian independen (*evidence-based machine learning*).",
        "notes": "Tekankan bahwa capstone ini adalah puncak pembuktian kompetensi mahasiswa pada Level 3 sebelum melangkah ke Level 4 Advanced AI."
    },
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran Bab 30",
        "body": "Setelah menyelesaikan bab ini, mahasiswa mampu:\n\n1. **Merumuskan** masalah rekayasa Deep Learning dengan batasan dan metrik keberhasilan terukur.\n2. **Memilih** arsitektur model yang tepat sesuai sifat domain data (Visi, Sekuens, Teks, Generatif).\n3. **Menyiapkan** dataset dengan partisi 3-way bebas kebocoran data (*zero data leakage*).\n4. **Membangun** baseline acuan dan mengimplementasikan pipeline pelatihan PyTorch modular.\n5. **Mendiagnosis** galat prediksi melalui *error analysis* dan audit keadilan subgrup.\n6. **Menyusun** Model Card dan laporan ilmiah bereproduksi penuh (*reproducible*).",
        "notes": "Paparkan 6 sasaran utama pembelajaran proyek capstone yang berfokus pada kemandirian rekayasa dan integritas ilmiah."
    },
    {
        "kind": "summary",
        "title": "Kilas Balik Fondasi Level 3 (Bab 21–29)",
        "body": "- **Bab 21–23**: Neuron buatan, Backpropagation, autograd DAG, dan pipeline modular `torch.nn.Module`.\n- **Bab 24–26**: Convolutional Neural Networks (CNN), lokalisasi deteksi objek YOLO, dan analitik video waktu-nyata.\n- **Bab 27–28**: Natural Language Processing, Simple RNN, LSTM, GRU, dan arsitektur Scaled Dot-Product Multi-Head Attention.\n- **Bab 29**: Large Language Models, Generative AI, adaptasi efisien LoRA, dan pertahanan Guardrails.\n- **Bab 30 (Kini)**: Mengintegrasikan seluruh blok pembangun ini ke dalam satu proyek akhir mandiri.",
        "notes": "Ajak mahasiswa mengingat kembali perjalanan panjang dari perceptron sederhana hingga Transformer raksasa."
    },
    {
        "kind": "content",
        "title": "Tahap 1: Perumusan Masalah & Hipotesis Rekayasa",
        "body": "- **Masalah Nyata**: Masalah harus memiliki nilai guna praktis (*practical utility*) dan batasan ruang lingkup yang jelas.\n- **Stakeholders**: Siapa pengguna sistem ini? Apa dampak kegagalan prediksi (*False Negative vs False Positive*)?\n- **Pertanyaan Riset**: Contoh: *'Apakah penambahan koneksi residu mampu meningkatkan stabilitas konvergensi model pada telemetri getaran sensor industri?'*\n- **Kriteria Sukses Terukur**: Target performa kuantitatif (*e.g.*, Macro F1 > 0.85 dengan latensi CPU < 10 ms).",
        "notes": "Ingatkan mahasiswa untuk tidak langsung melompat ke koding sebelum merumuskan masalah dan kriteria sukses secara tertulis."
    },
    {
        "kind": "split",
        "title": "Enam Pilihan Jalur Topik Proyek Capstone",
        "body": "Mahasiswa dapat memilih salah satu dari 6 jalur spesialisasi:\n\n| No | Jalur Proyek | Domain Data | Model Acuan Rekomendasi |\n|---|---|---|---|\n| **1** | **Image Classification** | Citra Medis / Tekstil / Agrikultur | CNN Modular + Residual Connections |\n| **2** | **Object Detection** | Bounding Box Citra Industri | YOLOv8n / Anchor-Free Detector |\n| **3** | **Sensor Sequence** | Deret Waktu IoT / Telemetri Getaran | LSTM / GRU / 1D-CNN Sequence Net |\n| **4** | **Text Classification** | Ulasan Dokumen / Analisis Sentimen | PyTorch Embedding + BiLSTM / Tiny Transformer |\n| **5** | **Text Generation** | Pembangkitan Teks Domain Spesifik | TinyGPT Decoder-Only Causal Attention |\n| **6** | **Applied CV Monitoring** | Video Tracking & Tripwire | OpenCV Pipeline + Centroid Tracker |",
        "notes": "Beri kebebasan mahasiswa memilih topik yang paling mereka minati, asalkan memenuhi standar metodologi yang setara."
    },
    {
        "kind": "content",
        "title": "Tahap 2: Dataset Provenance & Etika Penggunaan Data",
        "body": "- **Data Provenance**: Dari mana data berasal? Siapa pembuatnya? Apa lisensi penggunaannya (MIT, CC-BY, Apache-2.0)?\n- **Integritas Privasi**: Pastikan tidak ada data sensitif atau informasi identitas pribadi (PII) tanpa persetujuan subjek.\n- **Eksplorasi Data (EDA)**: Periksa keseimbangan kelas (*class imbalance*), nilai hilang (*missing values*), dan pencilan ekstrem (*outliers*).\n- **Data Dictionary**: Dokumentasikan arti fisik setiap kolom fitur dan rentang satuannya.",
        "notes": "Tekankan aspek hukum dan etika data: AI yang hebat tidak boleh dibangun di atas data ilegal atau melanggar hak cipta."
    },
    {
        "kind": "content",
        "title": "Tahap 3: Partisi 3-Way & Pencegahan Data Leakage",
        "body": "Protokol partisi data yang tidak dapat ditawar:\n\n$$\\text{Korpus Data} \\xrightarrow{\\text{Stratified Split}} \\begin{cases} \\textbf{Training Set (70\\%)} & \\text{Pembelajaran parameter model} \\\\ \\textbf{Validation Set (15\\%)} & \\text{Tuning hiperparameter \\& early stopping} \\\\ \\textbf{Final Test Set (15\\%)} & \\text{Audit kinerja akhir (Sekali pakai!)} \\end{cases}$$\n\n- **Hukum Kebocoran Data (Zero Data Leakage)**: Penskalaan fitur (`StandardScaler`) wajib di-*fit* **HANYA pada Training Set**, kemudian men-transformasi Validation dan Test Set.",
        "notes": "Tegaskan kembali bahaya data leakage yang sering menjadi penyebab model tampak hebat di lab namun hancur saat implementasi."
    },
    {
        "kind": "content",
        "title": "Tahap 4: Baseline Benchmark Model",
        "body": "- **Mengapa Butuh Baseline?**: Untuk membuktikan bahwa kompleksitas komputasi model deep learning benar-benar memberikan nilai tambah (*value-added*).\n- **Bentuk Baseline**: Model sederhana yang cepat dan mudah diinterpretasikan (misalnya *Logistic Regression*, *Random Baseline*, atau *Simple Heuristic*).\n- **Kaidah Rekayasa**: Jika deep neural network berukuran besar hanya menghasilkan performa yang sama atau lebih buruk dari Logistic Regression, model deep learning tersebut harus ditolak.",
        "notes": "Jadikan baseline sebagai pilar utama pembuktian ilmiah agar mahasiswa tidak terjebak dalam tren memakai model rumit tanpa alasan."
    },
    {
        "kind": "content",
        "title": "Tahap 5: Arsitektur Model Deep Learning PyTorch",
        "body": "Struktur modular berbasis `torch.nn.Module`:\n\n```python\nclass DeepPredictiveMaintenanceNet(nn.Module):\n    def __init__(self, in_features=4, hidden_dim=64, num_classes=3):\n        super().__init__()\n        self.input_layer = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())\n        self.res_block1 = ResidualBlock(hidden_dim)\n        self.res_block2 = ResidualBlock(hidden_dim)\n        self.head = nn.Linear(hidden_dim, num_classes)\n    \n    def forward(self, x):\n        return self.head(self.res_block2(self.res_block1(self.input_layer(x))))\n```\n\n- Gunakan koneksi residu untuk memelihara kelancaran aliran gradien.",
        "notes": "Perlihatkan kebersihan rancangan arsitektur berorientasi objek yang modular dan mudah diuji unit."
    },
    {
        "kind": "content",
        "title": "Tahap 6: Siklus Pelatihan Profesional & Early Stopping",
        "body": "- **Fungsi Rugi**: `nn.CrossEntropyLoss` untuk klasifikasi multikelas (menerima logits mentah).\n- **Optimizer**: `torch.optim.AdamW` dengan bobot peluruhan (*weight decay*) untuk regularisasi L2.\n- **Gradient Clipping**: Membatasi norma gradien maksimum $\\|g\\| \\le 1.0$ guna mencegah ledakan gradien (*exploding gradients*).\n- **Early Stopping**: Menyimpan status bobot terbaik (*best checkpoint*) saat Validation Loss terendah tercapai; menghentikan pelatihan jika tidak ada perbaikan selama $N$ epoch.",
        "notes": "Jelaskan peran vital early stopping dalam mencegah overfitting dan menghemat energi komputasi."
    },
    {
        "kind": "content",
        "title": "Tahap 7: Evaluasi Akhir & Error Analysis Diagnostics",
        "body": "- **Evaluasi Independen**: Jalankan inferensi pada Final Test Set yang disembunyikan menggunakan `model.eval()` dan `torch.no_grad()`.\n- **Metrik Multi-Dimensi**: Akurasi, Precision, Recall, Macro F1, dan Matriks Konfusi.\n- **Analisis Galat (Error Analysis)**:\n  - Sampel mana saja yang salah diprediksi?\n  - Mengapa model salah? Apakah karena nilai sensor berada di wilayah perbatasan (*boundary cases*) atau karena derau ekstrem?\n  - Klasifikasikan galat: *False Positives* vs *False Negatives*.",
        "notes": "Ajak mahasiswa membedah kegagalan model; sains yang baik justru belajar dari diseksi kesalahan prediksi."
    },
    {
        "kind": "content",
        "title": "Tahap 8: Audit Responsible AI & Keadilan Subgrup",
        "body": "- **Subgroup Fairness Audit**: Apakah performa model adil di seluruh subgrup data (misalnya performa pada mesin UNIT-01 berbanding UNIT-10)?\n  - Ukur disparitas metrik: $\\Delta \\text{F1} = \\max(\\text{F1}) - \\min(\\text{F1}) \\le 0.30$.\n- **Ketahanan terhadap Derau (Noise Robustness)**: Bagaimana respon model jika sensor mengalami gangguan sinyal derau Gauss $\\mathcal{N}(0, \\sigma^2)$?\n  - Model tangguh menunjukkan penurunan bertahap (*graceful degradation*), bukan keruntuhan katastrofik.",
        "notes": "Bahas aspek keadilan dan ketahanan operasional sebagai syarat kelayakan sistem AI di dunia industri."
    },
    {
        "kind": "content",
        "title": "Tahap 9: Manajemen Checkpoint & Reproducibility",
        "body": "- **Penyimpanan Aman**: Gunakan `torch.save(payload, path)` yang memuat `state_dict`, metadata arsitektur, dan parameter penskalaan (`scaler.mean_`, `scaler.scale_`).\n- **Integritas Muat Ulang**: Buktikan bahwa model yang dimuat ulang menghasilkan luaran yang **100% identik secara bitwise** (deviasi $< 1.0 \\times 10^{-7}$).\n- **Seed Deterministik**: Kunci seluruh generator bilangan acak (`torch.manual_seed`, `np.random.seed`).",
        "notes": "Tekankan pentingnya reproduksibilitas: eksperimen ilmiah yang tidak dapat diulang oleh peneliti lain dianggap tidak valid."
    },
    {
        "kind": "case",
        "title": "Implementasi Acuan: Hasil Nyata Proyek Capstone",
        "body": "Hasil eksekusi riil pada skrip `capstone_pipeline_runner.py`:\n\n- **Dataset**: 1.000 sampel telemetri 4-sensor multi-unit industri.\n- **Baseline (Logistic Regression)**: Akurasi Test = **88.67%**, Macro F1 = **0.8841**.\n- **Deep Learning (DeepPredictiveMaintenanceNet)**: Akurasi Test = **93.33%**, Macro F1 = **0.9328**.\n- **Keunggulan Nyata**: Peningkatan akurasi **+4.66%** dan F1 **+0.0487** di atas baseline.\n- **Disparitas Keadilan Subgrup**: Disparitas F1 antar-unit $\\Delta = 0.1624 \\le 0.30$ (**Lolos Uji Keadilan**).\n- **Waktu Komputasi**: Seluruh pipeline konvergen dalam **~2.8 detik** pada CPU standar.",
        "notes": "Paparkan data audit nyata ini sebagai bukti bahwa proyek capstone berjalan terbukti secara empiris dan ilmiah."
    },
    {
        "kind": "content",
        "title": "Enam Tonggak Proyek Capstone (Milestones)",
        "body": "Rencana kerja terjadwal penyelesaian proyek capstone:\n\n1. **Milestone 1**: Proposal Proyek & Perumusan Masalah (Minggu 1).\n2. **Milestone 2**: Eksplorasi Data, Lisensi & Pembuatan Pipeline Input (Minggu 2).\n3. **Milestone 3**: Implementasi Model Baseline & Arsitektur Deep Learning (Minggu 3).\n4. **Milestone 4**: Pelatihan Konvergen, Tuning & Early Stopping (Minggu 4).\n5. **Milestone 5**: Evaluasi Data Uji, Error Analysis & Audit Responsible AI (Minggu 5).\n6. **Milestone 6**: Penyusunan Laporan Ilmiah & Sidang Presentasi (Minggu 6).",
        "notes": "Jelaskan lini masa proyek agar mahasiswa dapat mengatur waktu kerja dengan disiplin."
    },
    {
        "kind": "split",
        "title": "Rubrik Penilaian Capstone (Total: 100 Poin)",
        "body": "Komponen penilaian holistik proyek capstone:\n\n| Kriteria Penilaian | Bobot | Indikator Skor Penuh (Full Credit) |\n|---|:---:|---|\n| **1. Problem & Design** | 15 Poin | Perumusan masalah lugas, batasan jelas, metrik terukur, relevansi tinggi. |\n| **2. Dataset & Ethics** | 15 Poin | Provenance transparan, lisensi legal, partisi 3-way bebas kebocoran data. |\n| **3. Architecture & Code** | 25 Poin | Kode PyTorch bersih modular, baseline acuan tersedia, model konvergen. |\n| **4. Training & Evaluation** | 25 Poin | Evaluasi multi-metrik pada Test Set terisolasi, perbandingan vs baseline valid. |\n| **5. Error & Responsible AI** | 10 Poin | Diseksi kasus salah prediksi mendalam, audit disparitas keadilan subgrup. |\n| **6. Documentation & Report** | 10 Poin | Laporan ilmiah lengkap, Model Card rapi, 100% reproduktif pada CPU. |",
        "notes": "Jelaskan transparansi rubrik penilaian kepada mahasiswa: poin tersebar merata di seluruh siklus hidup rekayasa."
    },
    {
        "kind": "case",
        "title": "Tujuh Kesalahan Umum yang Wajib Dihindari",
        "body": "- **1. Memilih Model Sebelum Memahami Data**: Memaksa menggunakan Transformer raksasa untuk data tabular kecil.\n- **2. Data Leakage pada Preprocessing**: Menghitung rata-rata dan deviasi standar pada seluruh dataset sebelum partisi.\n- **3. Mengabaikan Baseline**: Mengklaim akurasi 90% hebat tanpa menyadari tebakan acak kelas mayoritas bernilai 92%.\n- **4. Mengotak-atik Test Set**: Melakukan tuning hiperparameter berulang kali pada Test Set.\n- **5. Evaluasi Metrik Tunggal**: Hanya melihat akurasi tanpa memeriksa Recall pada kelas kritis yang jarang.\n- **6. Ketiadaan Error Analysis**: Tidak mengetahui mengapa model salah prediksi.\n- **7. Ketiadaan Dokumentasi Reproduktif**: Kode tidak dapat dijalankan oleh pihak luar.",
        "notes": "Ulas 7 jebakan umum ini agar mahasiswa tidak mengulangi kesalahan metodologis klasik."
    },
    {
        "kind": "content",
        "title": "Pedoman Presentasi Sidang Capstone",
        "body": "Struktur paparan presentasi berdurasi 15 menit:\n\n1. **Latar Belakang & Masalah** (2 menit): Mengapa masalah ini penting?\n2. **Karakteristik Data & Metrik** (2 menit): Sumber data, kelas, dan kriteria sukses.\n3. **Baseline vs Arsitektur Deep Learning** (3 menit): Mengapa model ini dipilih?\n4. **Hasil Pelatihan & Validasi** (3 menit): Kurva loss konvergen dan evaluasi data uji.\n5. **Diseksi Galat & Temuan Menarik** (3 menit): Kasus sulit dan audit keadilan.\n6. **Kesimpulan & Batasan Sistem** (2 menit): Keterbatasan dan rencana perbaikan.",
        "notes": "Berikan panduan presentasi yang ringkas, berbobot, dan berfokus pada bukti empiris."
    },
    {
        "kind": "summary",
        "title": "Rangkuman Intisari Bab 30",
        "body": "- **Penutupan Resmi Level 3**: Memadukan seluruh konsep neural networks, optimasi, computer vision, NLP, dan generative AI.\n- **Metodologi Disiplin**: Partisi 3-way bebas kebocoran, pembanding baseline, dan early stopping.\n- **Integritas Ilmiah**: Evaluasi satu kali pada held-out test set, error analysis kasus salah prediksi, dan audit disparitas subgrup.\n- **Reproduksibilitas Penuh**: Kode modular yang dapat dieksekusi secara instan dan deterministik pada CPU komputer standar.",
        "notes": "Tutup sesi materi dengan merayakan tuntasnya penguasaan mahasiswa atas Level 3 Deep Learning."
    },
    {
        "kind": "content",
        "title": "Gerbang Menuju Level 4: Advanced AI & Autonomy",
        "body": "Selamat! Anda telah menuntaskan seluruh **Level 3: Deep Learning & Generative AI**.\n\nPada **Level 4: Advanced AI & Autonomy (Bab 31–40)**, kita akan melangkah ke perbatasan teknologi otonom tingkat lanjut:\n- **Bab 31–32**: Advanced Fine-Tuning, Embeddings & Vector Search.\n- **Bab 33–34**: Production RAG Systems, AI Agents & Tool Integration.\n- **Bab 35–37**: Reinforcement Learning, MLOps Deployment & Edge AI.\n- **Bab 38–40**: Multimodal Systems, AI Governance, dan Advanced AI Capstone.",
        "notes": "Beri motivasi dan pengantar inspiratif menyambut materi mutakhir Level 4 pada semester berikutnya."
    },
    {
        "kind": "content",
        "title": "Referensi Akademik Bab 30",
        "body": "1. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.\n2. **Mitchell, M., et al. (2019).** *Model Cards for Model Reporting.* Proceedings of FAT* 2019, 220–229.\n3. **Sculley, D., et al. (2015).** *Hidden Technical Debt in Machine Learning Systems.* Advances in NeurIPS 2015.\n4. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.\n5. **ACM / IEEE-CS Joint Task Force on Computing Curricula. (2023).** *Computer Science Curricula 2023 (CS2023): Artificial Intelligence and Machine Learning Curricular Guidelines.*",
        "notes": "Paparkan daftar pustaka acuan primer standar kurikulum internasional yang mendasari metodologi proyek capstone."
    }
]

out_path = Path("content/level-3-deep-learning/chapter-30/slides/deck.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(slides, f, indent=2, ensure_ascii=False)

print(f"Berhasil menghasilkan {len(slides)} slide Reveal.js di: {out_path.resolve()}")
