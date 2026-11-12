# Panduan Praktik Laboratorium: Multimodal AI Systems
**Artificial Intelligence Learning Hub — Level 4 (Chapter 38)**  
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## 1. DESKRIPSI DAN PRASYARAT PRAKTIKUM

Praktikum ini membimbing mahasiswa membangun, melatih, dan mengevaluasi sistem **Multimodal AI** berbasis CPU lokal secara menyeluruh—mulai dari penciptaan pasangan citra–teks sintetis (`32x32` RGB), perancangan dual encoder ringan (`TinyImageEncoder` dan `TinyTextEncoder`), perbandingan strategi fusi (**Early Fusion**, **Late Fusion**, dan **Cross-Attention**), penurunan matematis **Symmetric InfoNCE / CLIP Loss**, hingga pelatihan miniature dual encoder untuk penelusuran silang dua arah (*Image-to-Text* & *Text-to-Image Retrieval*).

### Prasyarat Pengetahuan
- Arsitektur *Convolutional Neural Networks* (Chapter 24) dan representasi teks *Token Embeddings* (Chapter 27).
- Mekanisme *Scaled Dot-Product Attention* (Chapter 28) serta kesamaan kosinus vektor & pencarian semantik (Chapter 32).

### Lingkungan Eksekusi
- **Bahasa:** Python 3.10+ (Teruji pada Python 3.12.10).
- **Pustaka Utama:** `torch` (`2.15.0+cpu`), `numpy`, `matplotlib`. Seluruh script bersifat deterministik (`seed=42`) dan berjalan cepat di CPU biasa tanpa GPU.

---

## 2. PANDUAN LANGKAH DEMI LANGKAH EKSEKUSI

Seluruh 16 script Python mandiri tersimpan di `content/level-4-advanced-ai/chapter-38/code/`:

### Langkah 1: Mengapa Multimodal AI Diperlukan?
```bash
python content/level-4-advanced-ai/chapter-38/code/multimodal_foundations.py
```
- **Tujuan:** Mendemonstrasikan kasus sinergi lintas modalitas di mana informasi visual tunggal atau teks tunggal saja tidak memadai untuk menentukan peringatan kritis secara akurat.

### Langkah 2: Generator Dataset Pasangan Citra–Teks Terkendali
```bash
python content/level-4-advanced-ai/chapter-38/code/paired_dataset_generator.py
```
- **Tujuan:** Menghasilkan korpus sintetis `32x32` RGB yang memadukan **4 warna** (`red`, `green`, `blue`, `yellow`) $\times$ **4 bentuk** (`circle`, `square`, `triangle`, `rectangle`) = **16 konsep** dengan *strict train/test split* (`320` pasangan latih, `96` pasangan uji tak terlihat) dan menyimpan `data/paired_dataset_info.json`.

### Langkah 3: Arsitektur Ringan Image Encoder (`TinyImageEncoder`)
```bash
python content/level-4-advanced-ai/chapter-38/code/image_encoder.py
```
- **Tujuan:** Memverifikasi CNN ringan (`23,584` parameter) yang menghasilkan representasi spasial `4x4` (`16` token patch berdimensi `64`) untuk *Cross-Attention* maupun vektor global *pooled* `64`-dimensi.

### Langkah 4: Arsitektur Ringan Text Encoder (`TinyTextEncoder`)
```bash
python content/level-4-advanced-ai/chapter-38/code/text_encoder.py
```
- **Tujuan:** Memverifikasi *Embedding + Linear Projection* (`9,024` parameter) yang menghasilkan sekuens token `(B, 2, 64)` dan vektor kalimat *masked mean-pooled* `(B, 64)`.

### Langkah 5: Early Fusion (Feature Concatenation)
```bash
python content/level-4-advanced-ai/chapter-38/code/early_fusion.py
```
- **Tujuan:** Menggabungkan fitur citra & teks (`128`-d MLP), mencatat akurasi lengkap `100.00%`, namun membuktikan kerapuhannya saat teks dihilangkan (akurasi anjlok ke `25.00%`).

### Langkah 6: Late Fusion (Decision-Level Ensemble)
```bash
python content/level-4-advanced-ai/chapter-38/code/late_fusion.py
```
- **Tujuan:** Melatih cabang unimodal terpisah (`0.5 * P_img + 0.5 * P_txt`), mencatat `100.00%` ketika kedua modalitas hadir, serta ketahanan lebih baik saat modalitas teks hilang (`43.75%`) atau citra hilang (`100.00%`).

### Langkah 7: Cross-Attention Fusion & Verifikasi Paritas Tensor
```bash
python content/level-4-advanced-ai/chapter-38/code/cross_attention_fusion.py
```
- **Tujuan:** Membuktikan paritas numerik NumPy vs PyTorch pada matriks *Scaled Dot-Product Cross-Attention* ($Q=\text{Text}$, $K, V=\text{Image Patches}$, baris probabilitas berjumlah tepat `[1.0, 1.0]`), lalu melatih pengklasifikasi fusi penuh.

### Langkah 8: Ruang Embedding Bersama & Fenomena Modality Gap
```bash
python content/level-4-advanced-ai/chapter-38/code/multimodal_embeddings.py
```
- **Tujuan:** Memproyeksikan fitur citra & teks ke hipersfer satuan L2 (`32`-d, norm `1.0`) serta mengukur *Modality Gap* sebelum pelatihan kontrastif (rata-rata cosine similarity pasangan cocok hanya `0.2650`).

### Langkah 9: Matematika Symmetric InfoNCE / CLIP Loss
```bash
python content/level-4-advanced-ai/chapter-38/code/contrastive_loss.py
```
- **Tujuan:** Memverifikasi perhitungan manual matriks `3x3` ($\tau = 0.5$): `Manual I->T Loss = 0.73244`, `Manual T->I Loss = 0.75112`, dan rata-rata simetris `0.74178` yang persis identik dengan PyTorch `F.cross_entropy`.

### Langkah 10: Pelatihan Miniature Dual Encoder (Gaya CLIP)
```bash
python content/level-4-advanced-ai/chapter-38/code/clip_miniature.py
```
- **Tujuan:** Melatih `MiniatureDualEncoder` (`36,704` parameter) dengan *unique-concept batch sampling* (mencegah *false negatives*), menurunkan loss dari `1.6808` (Epoch 1) ke `0.0717` (Epoch 35), mencapai rata-rata kesamaan pasangan positif `+0.9753`, negatif `-0.0650` (margin `+1.0403`), serta menyimpan bobot `experiments/clip_miniature.pt`.

### Langkah 11: Visualisasi Matriks Kesamaan $16\times 16$
```bash
python content/level-4-advanced-ai/chapter-38/code/similarity_matrix.py
```
- **Tujuan:** Menghitung matriks cosine similarity lintas seluruh `16` konsep tak terlihat dan menyimpan heatmap `experiments/similarity_matrix.png`.

### Langkah 12: Text-to-Image Retrieval
```bash
python content/level-4-advanced-ai/chapter-38/code/text_to_image_retrieval.py
```
- **Tujuan:** Mencari Top-3 gambar paling cocok dari `96` gambar uji tak terlihat berdasarkan kueri teks seperti `'red circle'` dan `'yellow triangle'`.

### Langkah 13: Image-to-Text Retrieval
```bash
python content/level-4-advanced-ai/chapter-38/code/image_to_text_retrieval.py
```
- **Tujuan:** Menjodohkan gambar uji tak terlihat dengan `16` kandidat deskripsi teks kanonik melalui peringkat cosine similarity.

### Langkah 14: Evaluasi Kuantitatif Cross-Modal Retrieval
```bash
python content/level-4-advanced-ai/chapter-38/code/cross_modal_evaluation.py
```
- **Tujuan:** Mengukur metrik retrieval dua arah murni pada `96` sampel uji tak terlihat (`I->T`: `Recall@1/3/5 = 100.00%`, `Median Rank = 1.0`; `T->I`: `Recall@1/3/5 = 100.00%`, `Precision@6 = 100.00%`, `Median Rank = 1.0`).

### Langkah 15: Demo Konseptual VQA Deterministik
```bash
python content/level-4-advanced-ai/chapter-38/code/vqa_concept_demo.py
```
- **Tujuan:** Menjawab pertanyaan warna, bentuk (rasio pengisian *bounding-box*), dan eksistensi objek secara deterministik—dengan penegasan jujur bahwa inspektor aturan ini **bukan** model neural generatif VQA bebas.

### Langkah 16: Benchmark Strategi Fusi & Ketahanan Hilang Modalitas
```bash
python content/level-4-advanced-ai/chapter-38/code/multimodal_comparison.py
```
- **Tujuan:** Membandingkan **Early Fusion**, **Late Fusion**, dan **Cross-Attention** dalam kondisi lengkap maupun saat satu modalitas terputus, dan menyimpan grafik `experiments/multimodal_fusion_comparison.png`.

---

## 3. COMPANION NOTEBOOK & ARTEFAK DATA

Buka dan jalankan sel demi sel notebook pendamping pada:
- `content/level-4-advanced-ai/chapter-38/notebooks/chapter_38_companion.ipynb`
- Metadata dataset sintetis: `content/level-4-advanced-ai/chapter-38/data/paired_dataset_info.json`
- Grafik hasil eksperimen:
  - `content/level-4-advanced-ai/chapter-38/experiments/similarity_matrix.png`
  - `content/level-4-advanced-ai/chapter-38/experiments/multimodal_fusion_comparison.png`

---

## 4. LATIHAN BERTINGKAT DAN PEMBAHASAN

### A. Tingkat Dasar (Foundations)
1. **Soal:** Jelaskan perbedaan mendasar antara **Alignment** (penyelarasan ruang laten bersama) dan **Fusion** (penggabungan fitur untuk prediksi tugas tunggal) dalam sistem multimodal.
   - **Pembahasan:** **Alignment** memetakan input dari modalitas berbeda (misalnya citra dan teks) ke ruang vektor terpadu agar pasangan yang bermakna sama berdekatan secara geometris (contoh: CLIP untuk retrieval). **Fusion** memadukan sinyal dari kedua modalitas ke dalam satu representasi atau ensemble gabungan guna memecahkan tugas hilir seperti klasifikasi atau VQA.
2. **Soal:** Mengapa **Late Fusion** (`0.5 * P_img + 0.5 * P_txt`) mempertahankan akurasi yang jauh lebih baik saat salah satu sensor/modalitas putus dibandingkan **Early Fusion** pada eksperimen `multimodal_comparison.py`?
   - **Pembahasan:** Pada **Early Fusion**, bobot MLP dilatih secara bersamaan menerima `[z_img ; z_txt]` sehingga mengandalkan ko-aktivasi kedua cabang. Ketika vektor teks dinolkan (*missing text*), distribusi aktivasi lapisan pertama bergeser ekstrem (akurasi turun ke `25.00%`). Sebaliknya, pada **Late Fusion**, tiap cabang dilatih independen; jika salah satu modalitas mati, probabilitas modalitas tersisa tetap terkalibrasi (`43.75%` hanya citra, `100.00%` hanya teks).

### B. Tingkat Menengah (Implementation & Tensor Analysis)
3. **Soal:** Dalam fungsi `CrossAttentionFusionClassifier` (`cross_attention_fusion.py`), dari mana asal tensor $Q$, $K$, dan $V$, apa dimensinya pada batch $B=4$, dan mengapa hasil *Cross-Attention* tidak sama dengan sekadar `torch.cat`?
   - **Pembahasan:** Tensor $Q$ berasal dari representasi token teks `txt_seq` berukuran `(4, 2, 64)`, sedangkan $K$ dan $V$ berasal dari peta *spatial patch* citra `img_patches` (`4x4 = 16` patch) berukuran `(4, 16, 64)`. Operasi $\text{softmax}(QK^\top / \sqrt{d_k}) V$ menghasilkan pembobotan dinamis di mana setiap token kata (misalnya `'red'` atau `'circle'`) memilih patch gambar spesifik yang paling relevan, alih-alih menggabungkan statistik global secara kaku.
4. **Soal:** Pada perhitungan manual `contrastive_loss.py` untuk matriks `3x3` dengan suhu $\tau = 0.5$, diperoleh `Manual I->T Loss = 0.73244` dan `Manual T->I Loss = 0.75112`. Hitung total **Symmetric InfoNCE Loss** dan jelaskan dampak kehadiran *false negatives* jika satu mini-batch memuat dua sampel `'red circle'` yang berbeda posisi.
   - **Pembahasan:** $\mathcal{L}_{\text{sym}} = \frac{1}{2}(0.73244 + 0.75112) = 0.74178$. Jika satu mini-batch mengandung dua pasangan `'red circle'`, indeks diagonal $i = j$ menarik pasangan sendiri sementara elemen off-diagonal antara dua `'red circle'` dianggap kelas negatif oleh `F.cross_entropy`, sehingga mendorong representasi konsep yang identik agar saling menjauh (*gradient conflict*). Solusinya di `clip_miniature.py` adalah *unique-concept batch sampling*.

### C. Tingkat Lanjut (Critical Thinking & Engineering Limitations)
5. **Soal:** `MiniatureDualEncoder` meraih `Recall@1 = 100.00%` pada `96` citra uji tak terlihat setelah 35 epoch. Mengapa mahasiswa dilarang mengklaim angka ini setara dengan performa model CLIP skala produksi pada foto dunia nyata (*open-world*)?
   - **Pembahasan:** Korpus kita memiliki himpunan tertutup kecil (**16 konsep** diskrit warna $\times$ bentuk pada latar gelap terisolasi), tanpa oklusi rumit, variasi pencahayaan ekstrem, latar belakang alami yang berisik, maupun polisemi bahasa alami. Pada korpus dunia nyata seperti WIT/LAION (ratusan juta pasangan), diperlukan ribuan konsep tak terhingga, arsitektur ViT/Transformer jauh lebih besar, *augmentasi* kuat, dan teknik *hard-negative mining*.

---

## 5. MINI PROJECT: BUILDING A MINIATURE IMAGE-TEXT RETRIEVAL SYSTEM

### Deskripsi Tugas
Rancang, latih, dan audit sistem **Miniature Cross-Modal Image-Text Retrieval** Anda sendiri dengan memperluas dataset geometris menjadi minimal **5 warna $\times$ 4 bentuk = 20 konsep** (atau menambahkan atribut ukuran `small`/`large` menjadi **32 konsep**) serta mengevaluasi sensitivitas suhu $\tau \in \{0.07, 0.2, 0.5\}$ terhadap kurva konvergensi dan metrik retrieval tak terlihat.

### 16 Artefak Deliverable Wajib
1. Script generator dataset yang diperluas beserta pemecahan *strict unseen test split*.
2. Berkas JSON statistik dataset (`paired_dataset_info.json` versi perluasan).
3. Implementasi bersih `ImageEncoder` beserta unit test dimensi spasial & global.
4. Implementasi bersih `TextEncoder` beserta unit test *masked mean-pooling*.
5. Skrip pembuktian paritas NumPy vs PyTorch untuk *Symmetric InfoNCE Loss*.
6. Skrip pelatihan *Dual Encoder* dengan *unique-concept batch sampling* (zero false negatives).
7. Checkpoint bobot model terlatih (`experiments/*.pt`).
8. Eksperimen ablasi suhu $\tau \in \{0.07, 0.2, 0.5\}$ beserta perbandingan loss & margin positif-negatif.
9. Skrip visualisasi matriks kesamaan kosinus lengkap (`heatmap.png`).
10. Skrip interaktif *Text-to-Image Retrieval* (Top-5 hasil + label `[MATCH]`/`[MISS]`).
11. Skrip interaktif *Image-to-Text Retrieval* (Top-3 deskripsi kandidat).
12. Evaluasi dua arah lengkap: `Recall@1`, `Recall@3`, `Recall@5`, `Precision@k`, dan `Median Rank`.
13. Uji perbandingan minimal dua arsitektur fusi (**Early** vs **Late** vs **Cross-Attention**) saat satu modalitas diputus (*zeroed out*).
14. Analisis kasus gagal (*error analysis*) pada variasi noise visual tinggi atau kombinasi warna–bentuk berdekatan.
15. Notebook Jupyter terstruktur yang menjalankan ringkasan seluruh eksperimen tanpa error.
16. Laporan singkat (1–2 halaman) yang memaparkan batasan realistis sistem miniatur dibandingkan model *foundation multimodal* modern.

### Rubrik Penilaian (Total 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Keunggulan |
| :--- | :---: | :--- |
| **1. Pemahaman Konsep & Arsitektur Multimodal** | **15 poin** | Penjelasan jernih mengenai *Representation Alignment*, *Modality Gap*, perbedaan *Early/Late/Cross-Attention*, serta batasan model miniatur vs CLIP nyata. |
| **2. Implementasi Dataset, Encoders & Fusion** | **25 poin** | Kode PyTorch bersih tanpa kebocoran data latih–uji, dimensi tensor `(B, 16, D)` dan `(B, L, D)` tepat, serta uji ketahanan hilang modalitas berjalan benar. |
| **3. Pelatihan Kontrastif & Loss Verification** | **25 poin** | *Symmetric InfoNCE* tepat secara matematis (paritas NumPy–PyTorch tercapai), implementasi L2-normalisasi konsisten, serta strategi *unique-concept batching* mencegah *false negatives*. |
| **4. Evaluasi Retrieval & Visualisasi** | **25 poin** | Pelaporan lengkap `Recall@1/3/5`, `Precision@k`, dan `Median Rank` pada data uji tak terlihat, dilengkapi visualisasi heatmap matriks kesamaan dan ablasi suhu $\tau$. |
| **5. Analisis Kritis & Reprodusibilitas** | **10 poin** | *Random seed* terdokumentasi (`seed=42`), tidak ada klaim berlebihan, dan diskusi jujur mengenai kegagalan maupun batasan domain sintetis. |

---

## 6. REFERENSI AKADEMIK

1. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). *Multimodal Machine Learning: A Survey and Taxonomy*. IEEE TPAMI. https://arxiv.org/abs/1705.09406
2. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML. https://arxiv.org/abs/2103.00020
3. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
4. Li, J., et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. ICML. https://arxiv.org/abs/2201.12086
5. Antol, S., et al. (2015). *VQA: Visual Question Answering*. ICCV. https://arxiv.org/abs/1505.00468

