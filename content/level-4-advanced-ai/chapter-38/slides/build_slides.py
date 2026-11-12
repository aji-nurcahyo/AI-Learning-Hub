"""Slide Deck Generator for Chapter 38: Multimodal AI Systems.

Generates 37 slides conforming strictly to deck schema:
- Semantic kinds: cover, content, split, code, case, summary
- Body <= 1000 chars, code blocks <= 18 lines, substantive speaker notes.
"""

import json
import os

slides = [
    {
        "kind": "cover",
        "title": "Multimodal AI Systems",
        "body": "Representasi Lintas Modalitas, Strategi Fusi (Early, Late, Cross-Attention), Contrastive Alignment (CLIP), dan Cross-Modal Retrieval\n\n**Artificial Intelligence Learning Hub — Level 4 (Chapter 38)**\n*Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*",
        "notes": "Selamat datang di Chapter 38. Bab ini mengeksplorasi bagaimana kecerdasan buatan memadukan beragam modalitas persepsi seperti citra, teks, dan sensor ke dalam ruang representasi terpadu."
    },
    {
        "kind": "content",
        "title": "Mengapa Multimodal AI?",
        "body": "- **Dunia Bersifat Multimodal:** Manusia memahami dunia bukan hanya lewat teks, melainkan menyatukan penglihatan (citra/video), pendengaran (audio), dan bahasa secara bersamaan.\n- **Sinergi Resolusi Ambiguitas:** Cacat visual pada gambar komponen industri hanya bisa didiagnosis sebagai kerusakan kritis jika digabungkan dengan laporan teks telemetri suhu.\n- **Bukan Sekadar Dua Model Terpisah:** Multimodal AI mempelajari interaksi dan keselarasan antarsinyal.",
        "notes": "Ingatkan contoh di multimodal_foundations.py di mana neither image alone nor text alone resolves industrial critical alert."
    },
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran Bab 38",
        "body": "1. **Menjelaskan** konsep Multimodal AI dan ragam modalitas data.\n2. **Membedakan** arsitektur *Early*, *Late*, *Intermediate*, dan *Cross-Attention Fusion*.\n3. **Menurunkan** perhitungan *Scaled Dot-Product Cross-Attention* ($Q, K, V$ lintas modalitas).\n4. **Memahami** prinsip *Contrastive Learning* (InfoNCE) dan dual encoder bergaya CLIP.\n5. **Mengimplementasikan** prototipe miniatur *Image–Text Retrieval* dua arah pada CPU.\n6. **Mengidentifikasi** masalah *missing modality*, *modality gap*, serta bias dataset multimodal.",
        "notes": "Capaian menekankan pemahaman konseptual dan verifikasi matematis dari fusi hingga temu kembali silang."
    },
    {
        "kind": "split",
        "title": "Single-Modality vs Multimodal AI",
        "body": "- **Unimodal Models:**\n  - CNN (Bab 24) hanya memproses piksel citra.\n  - Transformer Teks (Bab 27–28) hanya memproses urutan token.\n- **Multimodal Systems:**\n  - Memetakan fitur heterogen (grid spasial 2D, token diskret 1D, deret waktu sensor) ke ruang semantik yang saling memahami.",
        "notes": "Hubungkan kembali dengan Chapter 24 (CNN), Chapter 27 (NLP), Chapter 28 (Attention), dan Chapter 32 (Embeddings)."
    },
    {
        "kind": "content",
        "title": "Ragam Modalitas & Domain Aplikasi",
        "body": "- **Image + Text:** Pencarian foto dari deskripsi bahasa, *Visual Question Answering* (VQA).\n- **Image + Sensor Telemetry:** *Smart Agriculture* (citra daun + kelembaban tanah) dan inspeksi mesin industri.\n- **Network Logs + Text Reports:** Monitoring gangguan jaringan telekomunikasi.\n- **Audio + Video + Text:** Transkripsi rapat dan pengenalan emosi pembicara.",
        "notes": "Setiap pasangan domain menuntut strategi penyelarasan waktu dan resolusi yang berbeda."
    },
    {
        "kind": "content",
        "title": "Representation Learning Lintas Modalitas",
        "body": "- Tantangan mendasar multimodal adalah **Heterogenitas Struktural**:\n  - Citra adalah tensor piksel kontinu 2D/3D $(C \\times H \\times W)$.\n  - Teks adalah barisan token diskret 1D berdimensi variabel.\n- Solusi: Setiap modalitas diproses oleh *encoder* spesifiknya terlebih dahulu sebelum diproyeksikan ke **Shared Embedding Space** berdimensi sama $d$.",
        "notes": "Tanpa proyeksi ke dimensi dan skala yang seragam, vektor citra dan teks tidak bisa diukur jarak kemiripannya."
    },
    {
        "kind": "content",
        "title": "Image Embeddings (`TinyImageEncoder`)",
        "body": "- Pada praktik CPU kita (`image_encoder.py`), citra RGB `32x32` melewati 3 blok konvolusi (`3 -> 16 -> 32 -> 64`) + `MaxPool2d`.\n- Menghasilkan dua keluaran berguna:\n  1. **Spatial Patch Grid `(B, 16, 64)`:** Peta fitur $4\\times 4 = 16$ patch spasial untuk *Cross-Attention*.\n  2. **Global Pooled Vector `(B, 64)`:** Ringkasan semantik citra utuh.",
        "notes": "Mempertahankan representasi spasial 4x4 memungkinkan token kata tertentu memperhatikan lokasi spesifik objek dalam gambar."
    },
    {
        "kind": "content",
        "title": "Text Embeddings (`TinyTextEncoder`)",
        "body": "- Teks deskripsi singkat (misal `\"blue square\"`) diubah menjadi indeks kosakata $\\to$ `nn.Embedding(vocab, 64)` + lapisan proyeksi linier.\n- Menghasilkan:\n  1. **Token Sequence `(B, L, 64)`:** Digunakan sebagai vektor *Query* dalam *Cross-Attention*.\n  2. **Masked Mean-Pooled Vector `(B, 64)`:** Mengabaikan token `<pad>` untuk mewakili kalimat utuh.",
        "notes": "Masking token <pad> memastikan panjang kalimat yang berbeda tidak mengencerkan vektor rata-rata teks."
    },
    {
        "kind": "content",
        "title": "Shared Embedding Space & Modality Gap",
        "body": "- Kedua vektor diproyeksikan via matriks $W_{\\text{img}}, W_{\\text{txt}}$ ke dimensi $d=32$, lalu dinormalisasi $L_2$:\n$$\\hat{\\mathbf{z}}_I = \\frac{W_I h_I}{\\|W_I h_I\\|_2}, \\quad \\hat{\\mathbf{z}}_T = \\frac{W_T h_T}{\\|W_T h_T\\|_2}$$\n- **Modality Gap:** Sebelum dilatih, embedding citra dan teks berkumpul di wilayah hipersfer yang terpisah (mean cosine similarity awal $\\approx 0$). Pelatihan kontrastif menjembatani celah ini.",
        "notes": "Normalisasi L2 memastikan dot product sama persis dengan Cosine Similarity."
    },
    {
        "kind": "content",
        "title": "Konsep Multimodal Fusion",
        "body": "- Bagaimana cara menggabungkan fitur dari dua atau lebih modalitas?\n- Tiga tingkat fusi utama:\n  1. **Early Fusion** (Tingkat Fitur Awal).\n  2. **Intermediate Fusion** (Tingkat Lapisan Tersembunyi / *Cross-Attention*).\n  3. **Late Fusion** (Tingkat Keputusan Akhir / *Ensemble*).",
        "notes": "Pemilihan strategi fusi bergantung pada korelasi antarsinyal serta seberapa sering salah satu sensor mungkin mati (missing modality)."
    },
    {
        "kind": "content",
        "title": "Early Fusion (Feature Concatenation)",
        "body": "- Menggabungkan langsung vektor fitur sebelum masuk ke MLP gabungan:\n$$\\mathbf{z}_{\\text{joint}} = [\\mathbf{z}_{\\text{img}} \\; ; \\; \\mathbf{z}_{\\text{txt}}] \\in \\mathbb{R}^{128}$$\n- **Kelebihan:** Mempelajari interaksi fitur lintas modalitas sejak lapisan awal.\n- **Kelemahan:** Bobot MLP terikat kuat pada kedua masukan; bila satu modalitas hilang (*zeroed*), akurasi bisa merosot tajam (pada uji kita: `100%` $\\to$ `25.00%` saat teks hilang!).",
        "notes": "Data dari early_fusion.py dan multimodal_comparison.py secara empiris mengilustrasikan kerapuhan Early Fusion saat modalitas teks hilang."
    },
    {
        "kind": "content",
        "title": "Intermediate Fusion",
        "body": "- Masing-masing modalitas diproses beberapa lapisan terlebih dahulu untuk mengekstrak representasi tingkat menengah yang lebih bersih dari derau mentah.\n- Baru di tengah jaringan dilakukan penyatuan fitur atau mekanisme **Cross-Attention**.\n- Menyeimbangkan pemodelan interaksi halus dengan modularitas arsitektur.",
        "notes": "Hampir seluruh arsitektur Vision-Language modern menggunakan variasi Intermediate Fusion."
    },
    {
        "kind": "content",
        "title": "Late Fusion (Decision-Level Ensemble)",
        "body": "- Melatih cabang unimodal secara independen ($P_{\\text{img}}(y \\mid x_I)$ dan $P_{\\text{txt}}(y \\mid x_T)$), lalu menggabungkan probabilitas keputusannya:\n$$P_{\\text{fused}}(y) = w \\cdot P_{\\text{img}}(y) + (1 - w) \\cdot P_{\\text{txt}}(y)$$\n- **Keunggulan Nyata:** Sangat tangguh terhadap *missing modality*. Jika kamera rusak, sistem otomatis memakai `w = 0.0` (100% mengandalkan cabang teks!).",
        "notes": "Pada uji missing modality, Late Fusion mempertahankan 43.75% (Image Only) dan 100.00% (Text Only) dengan transisi mulus tanpa retraining."
    },
    {
        "kind": "content",
        "title": "Komparasi Empiris Tiga Strategi Fusi (Hasil Aktual)",
        "body": "Evaluasi pada 96 pasangan gambar–teks uji tak terlihat (`multimodal_comparison.py`):\n\n| Arsitektur Fusi | Kedua Modalitas | Hanya Gambar (Teks Hilang) | Hanya Teks (Gambar Hilang) |\n| :--- | :---: | :---: | :---: |\n| **Early Fusion** | **100.00%** | 25.00% | 93.75% |\n| **Late Fusion** | **100.00%** | **43.75%** | **100.00%** |\n| **Cross-Attention** | **100.00%** | 25.00% | 62.50% |",
        "notes": "Perhatikan bagaimana Late Fusion lebih tahan banting terhadap kehilangan modalitas tanpa pelatihan dropout khusus."
    },
    {
        "kind": "content",
        "title": "Cross-Attention Fusion",
        "body": "- **Cross-Attention bukan sekadar penggabungan (`concat`)!**\n- Dalam **Self-Attention**, $Q, K, V$ berasal dari modalitas yang sama. Dalam **Cross-Attention**, **Query ($Q$) berasal dari satu modalitas** (misal Teks), sedangkan **Key ($K$) dan Value ($V$) berasal dari modalitas lain** (misal Patch Spasial Citra):\n$$\\text{CrossAttn}(Q_{\\text{txt}}, K_{\\text{img}}, V_{\\text{img}}) = \\text{softmax}\\left(\\frac{Q_{\\text{txt}} K_{\\text{img}}^T}{\\sqrt{d_k}}\\right) V_{\\text{img}}$$",
        "notes": "Hal ini memungkinkan setiap kata dalam kalimat secara selektif menyorot patch piksel tertentu pada gambar."
    },
    {
        "kind": "code",
        "title": "Worked Example Tensor Cross-Attention (NumPy = PyTorch)",
        "body": "```python\n# Q (2 token teks x d_k=4), K & V (3 patch citra x d_k=4)\nd_k = 4\nscores = (Q @ K.T) / np.sqrt(d_k)        # Shape: (2, 3)\nattn_weights = softmax(scores, axis=-1)  # Baris dijamin berjumlah 1.0\nout = attn_weights @ V                   # Shape: (2, 4)\n\n# Hasil aktual (verifikasi paritas PyTorch atol < 1e-5):\n# Row sums == [1.0, 1.0] (PASS)\n```",
        "notes": "Kode verifikasi silang NumPy dan PyTorch tersedia di cross_attention_fusion.py."
    },
    {
        "kind": "content",
        "title": "Contrastive Learning Foundations",
        "body": "- Alih-alih melatih pengklasifikasi label diskret $K$-kelas yang kaku, **Contrastive Learning** belajar dari pasangan data langsung:\n  - **Positive Pairs $(I_i, T_i)$:** Gambar dan deskripsi yang benar berpasangan $\\to$ **tarik berdekatan** di hipersfer.\n  - **Negative Pairs $(I_i, T_{j \\neq i})$:** Gambar dan deskripsi yang tidak cocok $\\to$ **dorong menjauh**.",
        "notes": "Pendekatan kontrastif tidak membutuhkan label kelas diskret tertutup, membuka kemampuan zero-shot ke konsep baru."
    },
    {
        "kind": "content",
        "title": "Positive, Batch Negatives & False Negatives",
        "body": "- Dalam satu mini-batch berukuran $N$, terdapat $N$ pasangan positif di diagonal utama dan $N^2 - N$ pasangan negatif di luar diagonal.\n- **Peringatan *False Negatives*:** Jika sebuah batch acak memuat dua sampel berkonsep sama (misal dua gambar `\"red circle\"`), InfoNCE standar keliru menganggap pasangan silang keduanya sebagai negatif!\n- **Solusi:** Gunakan *unique-concept batch sampling* (1 sampel per konsep unik per batch).",
        "notes": "Penjelasan ini krusial: pada clip_miniature.py, batch ukuran 16 mengambil 1 sampel unik dari masing-masing 16 konsep geometris."
    },
    {
        "kind": "content",
        "title": "Symmetric InfoNCE / CLIP Contrastive Loss",
        "body": "- Diberikan matriks kesamaan kosinus diskalakan suhu $\\tau$: $L = \\frac{1}{\\tau} \\hat{Z}_I \\hat{Z}_T^T \\in \\mathbb{R}^{N \\times N}$.\n- Fungsi rugi simetris adalah rata-rata *Cross-Entropy* arah baris (Image $\\to$ Text) dan arah kolom (Text $\\to$ Image):\n$$\\mathcal{L}_{\\text{CLIP}} = \\frac{1}{2} \\left( \\text{CE}(L, \\mathbf{y}) + \\text{CE}(L^T, \\mathbf{y}) \\right), \\quad \\mathbf{y} = [0, 1, \\dots, N-1]$$",
        "notes": "Contoh manual 3x3 di contrastive_loss.py cocok persis dengan output F.cross_entropy PyTorch."
    },
    {
        "kind": "content",
        "title": "Pengantar Arsitektur CLIP (Radford et al., 2021)",
        "body": "- **CLIP (*Contrastive Language–Image Pre-training*):** Dilatih pada 400 juta pasangan gambar–teks internet.\n- **Keunggulan Zero-Shot:** Setelah dilatih, CLIP mampu mengklasifikasi gambar ke kategori baru dengan membandingkan embedding gambar melawan teks prompt seperti `\"a photo of a {label}\"`.\n- **Catatan:** Implementasi miniatur kita mengajarkan mekanisme arsitektural CLIP, bukan menyamai kapasitas semantik dunia nyata bobot resmi CLIP.",
        "notes": "Selalu jaga kejujuran klaim bahwa prototipe geometri 4x4 tidak sama dengan model pretrained 400M pasangan."
    },
    {
        "kind": "content",
        "title": "Arsitektur Dual Encoder",
        "body": "```\n[Citra RGB 32x32] -> [TinyImageEncoder] -> [Proj] -> L2Norm -> z_I (32-d)\n                                                                  |\n                                                         Cosine Sim (z_I . z_T) / tau\n                                                                  |\n[Teks Deskripsi]  -> [TinyTextEncoder]  -> [Proj] -> L2Norm -> z_T (32-d)\n```",
        "notes": "Karena Image Encoder dan Text Encoder berjalan terpisah (Dual Encoder), seluruh jutaan embedding gambar dapat diindeks offline di FAISS untuk pencarian super cepat."
    },
    {
        "kind": "content",
        "title": "Penyelarasan Gambar–Teks (Modality Alignment)",
        "body": "- Pelatihan kontrastif menggeser dua himpunan vektor dari *Modality Gap* menuju keselarasan geometris:\n  - Sebelum dilatih: Rata-rata kosinus pasangan cocok $\\approx 0.0$.\n  - Setelah 35 epoch pelatihan (`clip_miniature.py`): Loss turun `1.6808` $\\to$ **`0.0717`**.\n  - Rata-rata Kosinus Pasangan Positif: **`+0.9753`**.\n  - Rata-rata Kosinus Pasangan Negatif: **`-0.0650`** (Margin keselarasan = **`+1.0403`**!).",
        "notes": "Margin +1.04 memisahkan konsep cocok dan tidak cocok secara tegas di ruang embedding 32-dimensi."
    },
    {
        "kind": "content",
        "title": "Prototipe Miniature Dual Encoder (36.7K Param)",
        "body": "- Dataset Sintetis Terkontrol (`paired_dataset_generator.py`): 16 konsep (4 warna $\\times$ 4 bentuk) pada citra `32x32x3` dengan variasi acak ukuran serta pergeseran posisi.\n- **Pemecahan Ketat Train/Test:** 320 pasangan latih dan **96 pasangan uji tak terlihat** (6 varian offset/noise baru per konsep) agar evaluasi menguji generalisasi posisi & warna, bukan sekadar menghafal gambar latih.",
        "notes": "Pemisahan train/test pada varian posisi memastikan encoder benar-benar belajar fitur warna & bentuk yang invarian terhadap translasi ringan."
    },
    {
        "kind": "code",
        "title": "Alur Pelatihan Kontrastif (PyTorch CPU)",
        "body": "```python\n# Unique-concept batch (16 konsep per batch, 0 false negatives)\nemb_i = model.encode_image(batch_imgs)  # (16, 32) L2-normalized\nemb_t = model.encode_text(batch_toks)   # (16, 32) L2-normalized\n\nlogits = (emb_i @ emb_t.T) / 0.15\nlabels = torch.arange(len(batch_imgs))\nloss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))\nloss.backward()\n```",
        "notes": "Pelatihan 35 epoch selesai dalam kurang dari 2 detik pada CPU biasa."
    },
    {
        "kind": "content",
        "title": "Matriks Kesamaan Silang $16 \\times 16$",
        "body": "- Visualisasi matriks kesamaan `similarity_matrix.png` menunjukkan diagonal utama berwarna merah menyala (**skor $\\approx +0.97$**) sementara seluruh kotak di luar diagonal berwarna pucat/biru (**skor $\\approx -0.06$**).\n- Membuktikan bahwa gambar `red circle` hanya sejajar kuat dengan teks `\"red circle\"`, bukan `\"red square\"` atau `\"blue circle\"`.",
        "notes": "Tekankan pentingnya penguasaan komposisional (kombinasi atribut warna + bentuk secara bersamaan)."
    },
    {
        "kind": "content",
        "title": "Image-to-Text Retrieval (`I -> T`)",
        "body": "- Diberikan gambar uji tak terlihat `#00` (`\"red circle\"`), `image_to_text_retrieval.py` menghitung kesamaan kosinus terhadap 16 deskripsi kanonik:\n  - **Rank 1:** `\"red circle\"` (`+0.985+`) **[MATCH]**\n  - **Rank 2:** Konsep warna atau bentuk berdekatan dengan skor jauh lebih rendah.\n- Berguna untuk pemberian tag otomatis (*auto-captioning / tagging*) katalog foto produk.",
        "notes": "Semua evaluasi menggunakan citra baru dari test_ds yang tidak pernah dilihat selama pelatihan."
    },
    {
        "kind": "content",
        "title": "Text-to-Image Retrieval (`T -> I`)",
        "body": "- Diberikan kueri teks seperti `\"blue square\"` atau `\"yellow rectangle\"`, `text_to_image_retrieval.py` menelusuri seluruh **96 gambar uji tak terlihat**:\n  - Seluruh 3 gambar peringkat teratas (*Top-3*) untuk setiap kueri cocok persis dengan warna dan bentuk yang diminta!",
        "notes": "Menjadi dasar mesin pencari foto berdasarkan teks (semantic photo search)."
    },
    {
        "kind": "content",
        "title": "Metrik Evaluasi Retrieval Lintas Modalitas (Hasil Aktual)",
        "body": "Evaluasi ketat pada **96 gambar uji tak terlihat** (`cross_modal_evaluation.py`):\n\n| Arah Penelusuran | Recall@1 | Recall@3 | Recall@5 | Median Rank |\n| :--- | :---: | :---: | :---: | :---: |\n| **Image $\\to$ Text (`N=96` -> `16` teks)** | **100.00%** | **100.00%** | **100.00%** | **1.0** |\n| **Text $\\to$ Image (`N=16` -> `96` gambar)** | **100.00%** | **100.00%** | **100.00%** | **1.0** |\n\n*(Mean Precision@6 untuk T $\\to$ I = **100.00%** — seluruh 6 varian uji konsep duduk di peringkat 1–6 teratas!)*",
        "notes": "Tegaskan kembali bahwa 100% ini dicapai pada geometri sintetis terkontrol sebagai pembuktian konsep pipeline, bukan klaim pemahaman dunia nyata."
    },
    {
        "kind": "content",
        "title": "Visual Question Answering (VQA) Konseptual",
        "body": "- **Arsitektur VQA:** Menggabungkan konteks visual citra + pertanyaan bahasa alami untuk menghasilkan jawaban langsung.\n- **Demonstrasi Baseline (`vqa_concept_demo.py`):** Menerapkan ekstraksi atribut visual deterministik (rata-rata warna RGB foreground & rasio keterisian *bounding box*) untuk menjawab pertanyaan warna, bentuk, dan luas piksel.\n- *Label Eksplisit:* Ini adalah inspektor aturan deterministik, **bukan model neural VQA generatif**.",
        "notes": "Misal rasio area/bbox: segitiga ~0.50, lingkaran ~0.78, persegi ~1.0."
    },
    {
        "kind": "content",
        "title": "Menangani Modalitas Hilang (*Missing Modalities*)",
        "body": "- Di lapangan nyata, sensor kamera dapat tertutup lumpur atau saluran teks log gagal terkirim.\n- **Dampak:** Arsitektur terikat erat (*Early Fusion* / *Cross-Attention*) turun tajam ke `25.00%` bila teks hilang.\n- **Mitigasi:** Gunakan **Late Fusion**, **Modality Dropout** saat pelatihan (mengacak penghapusan salah satu modalitas), atau rekonstruksi silang.",
        "notes": "Jangan pernah mengasumsikan seluruh modalitas akan selalu hadir 100% tanpa gangguan di lapangan."
    },
    {
        "kind": "content",
        "title": "Tantangan Teknik Sistem Multimodal",
        "body": "1. **Perbedaan Laju Sampling (*Synchronization*):** Video 30 FPS vs audio 16 kHz vs sensor suhu 1 Hz menuntut penyelarasan temporal.\n2. **Sinyal Bertentangan (*Conflicting Modalities*):** Apa yang harus dipercaya jika sensor suara mendeteksi anomali sedangkan citra kamera tampak bersih?\n3. **Biaya Komputasi:** Menjalankan dua atau tiga *backbone* besar serentak melipatgandakan konsumsi memori & latensi.",
        "notes": "Tantangan ini harus diatasi sebelum menerapkan sistem multimodal ke ranah kritis."
    },
    {
        "kind": "content",
        "title": "Responsible & Safe Multimodal AI",
        "body": "- **Cross-Modal Hallucination:** Model Vision-Language besar sering mengklaim objek tidak ada di gambar karena bias teks (*language priors*).\n- **Dataset Bias & Privacy:** Gabungan wajah (citra) + rekam suara + teks nama melipatgandakan risiko identifikasi pribadi (*PII exposure*).\n- *(Pembahasan mendalam keamanan dan tata kelola dibahas khusus pada Chapter 39).* ",
        "notes": "Pastikan lisensi penggunaan gambar dan teks dipatuhi pada proyek skala produksi."
    },
    {
        "kind": "code",
        "title": "Demonstrasi Praktik Python: Text-to-Image Retrieval",
        "body": "```python\nfrom clip_miniature import load_or_train_clip_miniature\nfrom text_to_image_retrieval import retrieve_images_for_text\n\nmodel, test_ds = load_or_train_clip_miniature(seed=42)\nhits = retrieve_images_for_text(model, test_ds, \"blue square\", top_k=3)\n\nfor h in hits:\n    print(f\"Cosine={h['cosine_sim']:+.4f} | Test Img #{h['image_idx']} ({h['true_caption']})\")\n```",
        "notes": "Mahasiswa dapat langsung menjalankan skrip retrieval untuk menguji kueri warna-bentuk secara interaktif."
    },
    {
        "kind": "case",
        "title": "Studi Kasus Mini Project & Domain Terapan",
        "body": "- **Mini Project Bab 38:** *Building a Miniature Image–Text Retrieval System* (dataset berpasangan, Dual Encoder, InfoNCE, evaluasi Recall@k dua arah, serta analisis kegagalan).\n- **Contoh Terapan:** *Smart Agriculture* (citra daun + sensor tanah), *Telekomunikasi* (log jaringan + laporan gangguan teks), dan *Inspeksi Industri* (citra produk + pembacaan suhu/getaran).",
        "notes": "Rubrik 100 poin: Pemahaman Masalah & Data (15), Arsitektur Multimodal (25), Contrastive Learning (25), Evaluasi Retrieval & Analisis (25), Dokumentasi (10)."
    },
    {
        "kind": "content",
        "title": "Diskusi Reflektif Kelas",
        "body": "1. Mengapa pelatihan *CLIP-style* pada batch yang berisi banyak gambar bermakna identik memerlukan pencegahan *false negatives*, dan bagaimana strategi *hard negative mining* membantu?\n2. Jika merancang sistem inspeksi mesin pabrik dengan kamera serta sensor getaran yang sesekali putus koneksi, arsitektur fusi mana yang Anda pilih dan mengapa?",
        "notes": "Arahkan diskusi pada alasan praktis memilih Late Fusion atau Modality Dropout pada lingkungan sensor tak andal."
    },
    {
        "kind": "summary",
        "title": "Ringkasan Pembelajaran Bab 38",
        "body": "- **Multimodal AI** menyatukan modalitas heterogen ke dalam ruang representasi bermakna untuk memecahkan tugas lintas persepsi.\n- **Strategi Fusi:** *Early* (mudah tapi rapuh modalitas hilang), *Late* (tangguh ensemble), dan *Cross-Attention* ($Q, K, V$ silang beresolusi spasial).\n- **Contrastive Learning (InfoNCE / CLIP)** menyelaraskan pasangan citra–teks ke hipersfer satuan, menghasilkan **100% Recall@1** pada geometri uji tak terlihat.",
        "notes": "Bab 39 selanjutnya membahas AI Safety, Security & Governance secara mendalam menuju Capstone Akhir Bab 40."
    },
    {
        "kind": "content",
        "title": "Referensi Akademik Terverifikasi",
        "body": "1. **Radford, A., et al. (2021).** *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.\n2. **Oord, A. v. d., Li, Y., & Vinyals, O. (2018).** *Representation Learning with Contrastive Predictive Coding (InfoNCE).* arXiv:1807.03748.\n3. **Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2019).** *Multimodal Machine Learning: A Survey and Taxonomy.* IEEE TPAMI.\n4. **Vaswani, A., et al. (2017).** *Attention Is All You Need.* NeurIPS 2017.\n5. **Antol, S., et al. (2015).** *VQA: Visual Question Answering.* ICCV 2015.",
        "notes": "Seluruh referensi merupakan publikasi paling berpengaruh di bidang pembelajaran multimodal modern."
    }
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(script_dir, "deck.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(slides)} slides to {out_file}")


if __name__ == "__main__":
    main()

