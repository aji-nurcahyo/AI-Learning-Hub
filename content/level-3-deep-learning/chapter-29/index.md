# Bab 29: Large Language Models & Generative AI

## Arsitektur Autoregresif, Next-Token Prediction, Pre-training, Instruction Tuning, LoRA PEFT, Strategi Sampling, In-Context Learning, RAG, dan Tata Kelola Guardrails

---

## 1. Pendahuluan & Spektrum Generative AI

Pada Bab 27 dan Bab 28, kita telah membedah representasi bahasa alami dan mekanisme revolusioner *Attention Mechanism* serta arsitektur *Transformer*. Pada bab ini, kita melangkah ke puncak perkembangan kecerdasan buatan abad ke-21: **Large Language Models (LLM)** dan **Generative Artificial Intelligence (Generative AI)**.

```
+---------------------------------------------------------------------------------------------------------+
|                               SPEKTRUM UTAMA KELUARGA MODEL GENERATIF                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. AUTOREGRESSIVE TRANSFORMERS (LLM)       2. DIFFUSION MODELS                                        |
|      +--------------------------------+         +--------------------------------+                      |
|      | Teks: "Kecerdasan buatan..."   |         | Derau Gauss -> Rekonstruksi    |                      |
|      | ──► [Next-Token Prediction]    |         | ──► Citra Fotorealistik / Video|                      |
|      +--------------------------------+         +--------------------------------+                      |
|      Contoh: GPT-4, LLaMA-3, Claude             Contoh: Stable Diffusion, Midjourney, Sora              |
|                                                                                                         |
|   3. VARIATIONAL AUTOENCODERS (VAE)          4. GENERATIVE ADVERSARIAL NETWORKS (GAN)                   |
|      +--------------------------------+         +--------------------------------+                      |
|      | Pemetaan ke Ruang Laten Normal |         | Pertarungan Generator vs       |                      |
|      | ──► Rekonstruksi Kontinu Data  |         | Discriminator (Minimax Game)   |                      |
|      +--------------------------------+         +--------------------------------+                      |
|      Contoh: Kompresi Representasi Laten        Contoh: StyleGAN, CycleGAN                              |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Pemodelan Diskriminatif vs Pemodelan Generatif
Secara formal, perbedaan kedua paradigma dapat dirumuskan melalui aljabar probabilitas:
- **Model Diskriminatif** memodelkan probabilitas bersyarat $P(Y \mid X)$: mencari batas pemisah (*decision boundary*) untuk menentukan label kelas target $Y$ berdasarkan masukan $X$ (*e.g.*, mengklasifikasikan apakah ulasan bersentimen positif atau negatif).
- **Model Generatif** memodelkan probabilitas data $P(X)$ atau probabilitas bersama $P(X, Y)$: mempelajari seluruh distribusi fitur data masukan untuk membangkitkan sampel-sampel baru yang valid dan berdistribusi serupa (*e.g.*, menulis esai ulasan baru yang meyakinkan).

---

## 2. Fondasi Language Modeling: Next-Token Prediction

Inti dari setiap model bahasa (Language Model) adalah menetapkan distribusi probabilitas atas rangkaian token kata $\mathbf{W} = (w_1, w_2, \dots, w_T)$.

```
+---------------------------------------------------------------------------------------------------------+
|                                ALUR PREDIKSI NEXT-TOKEN DALAM LLM                                       |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Prompt Konteks: "Kecerdasan buatan berkembang dengan sangat"                                          |
|                                                                                                         |
|   1. Tokenisasi & Embedding      : [102, 451, 890, 23, 712] ──► Tensor Representasi (T x d_model)       |
|                                                                                                         |
|   2. Transformer Decoder Stacks  : Lapisan Masked Multi-Head Attention + Feed-Forward Blocks            |
|                                                                                                         |
|   3. LM Head Proyeksi Linier     : Matriks Logits Mentah z_t berdimensi (|V| = 32.000)                  |
|                                                                                                         |
|   4. Fungsi Softmax Normalisasi  : P(w_t | Konteks) = exp(z_i) / sum(exp(z_j))                         |
|                                    - "cepat" : P = 0.52                                                 |
|                                    - "pesat" : P = 0.38                                                 |
|                                    - "hebat" : P = 0.08                                                 |
|                                    - "kucing": P = 0.0001                                               |
|                                                                                                         |
|   5. Strategi Sampling           : Memilih satu token berdasarkan distribusi -> Terpilih: "cepat"       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Aturan Perkalian Probabilitas (Chain Rule)
Berdasarkan aturan rantai probabilitas, probabilitas suatu kalimat utuh adalah perkalian probabilitas bersyarat dari setiap kata yang bergantung pada seluruh kata-kata sebelumnya:

$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

LLM tidak "berpikir" seperti manusia, melainkan mengeksekusi inferensi statistik berkecepatan tinggi: *berdasarkan pola triliunan teks yang telah dipelajari, token apa yang paling masuk akal untuk menyambung kalimat ini?*

### Tokenisasi Subword (BPE)
Bahasa manusia kaya akan morfologi imbuhan dan kata majemuk. Pendekatan tokenisasi subword (*Byte-Pair Encoding* / WordPiece) menjadi standar emas industri karena mampu:
1. Memecah kata baru di luar kamus (*Out-of-Vocabulary* / OOV) menjadi sub-satuan leksikal bermakna (*e.g.*, *'pembelajaran'* $\rightarrow$ `['pem', '##belajar', '##an']`).
2. Menjaga ukuran kosakata tetap efisien (biasanya $|V| \in [32.000, 128.000]$ token).
3. Mengonversi teks menjadi deretan bilangan bulat diskret (**Token IDs**).

---

## 3. Objektif Pelatihan & Metrik Perplexity

Pelatihan model bahasa menggunakan fungsi rugi **Cross-Entropy Loss**, yang mengukur jarak antara distribusi probabilitas yang diprediksi model $\hat{\mathbf{y}}_t$ terhadap distribusi *ground truth* satu-panas (*one-hot*) $\mathbf{y}_t^*$:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \ln P(w_t^* \mid w_{<t})$$

Nilai ini identik dengan rata-rata **Negative Log-Likelihood (NLL)**.

```
+---------------------------------------------------------------------------------------------------------+
|                               KONSEP DAN MAKNA FISIS METRIK PERPLEXITY (PPL)                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Formula: PPL = exp(Cross-Entropy Loss)                                                                |
|                                                                                                         |
|   1. MODEL SEMPURNA (Teoretis)   : Loss = 0.00 nat  ──► PPL = exp(0) = 1.0                              |
|      -> Model memiliki kepastian mutlak (0% kebingungan).                                               |
|                                                                                                         |
|   2. MODEL TERLATIH BAIK (LLM)   : Loss = 1.20 nat  ──► PPL = exp(1.2) = 3.32                           |
|      -> Rata-rata model hanya ragu-ragu di antara 3 hingga 4 pilihan kata potensial.                    |
|                                                                                                         |
|   3. MODEL ACAK (Untrained)      : Loss = ln(|V|)   ──► PPL = |V| (misal 32.000)                        |
|      -> Model menebak secara buta setara dengan melempar dadu bersisi 32.000.                           |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Siklus Hidup Pelatihan LLM Modern

Proses rekayasa model bahasa mutakhir dari teks mentah hingga menjadi asisten interaktif terpercaya melalui tiga tahapan besar:

```
+---------------------------------------------------------------------------------------------------------+
|                                    TIGA TAHAP PELATIHAN LLM MODERN                                      |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   TAHAP 1: PRE-TRAINING (Pondasi Representasi Dunia)                                                    |
|   - Data: Triliunan token teks web publik (Wikipedia, buku, repositori kode, artikel).                  |
|   - Komputasi: Ribuan klaster GPU beroperasi paralel selama berbulan-bulan (biaya $10M - $100M+).       |
|   - Hasil: Base Pretrained Model (Mampu melengkapi dokumen, namun belum patuh instruksi).               |
|            │                                                                                            |
|            ▼                                                                                            |
|   TAHAP 2: INSTRUCTION TUNING / SFT (Supervised Fine-Tuning)                                            |
|   - Data: Puluhan ribu dialog instruksi berkualitas tinggi: {"prompt": "...", "response": "..."}.      |
|   - Pembelajaran: Mengajarkan model format asisten dialog (System, User, Assistant).                    |
|   - Hasil: Instruction-Tuned Model (Mampu mengikuti perintah dan menjawab pertanyaan).                 |
|            │                                                                                            |
|            ▼                                                                                            |
|   TAHAP 3: ALIGNMENT (Penyelarasan Nilai & Keselamatan)                                                 |
|   - Metode: Reinforcement Learning from Human Feedback (RLHF) atau Direct Preference Optimization (DPO).|
|   - Kriteria 3H: Helpful (Membantu), Honest (Jujur/Faktual), Harmless (Tidak Berbahaya/Aman).           |
|   - Hasil: Production-Ready Aligned Assistant (e.g., ChatGPT, Claude, LLaMA-Instruct).                  |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 5. Fine-Tuning Efisien: Low-Rank Adaptation (LoRA)

Melakukan *Full Fine-Tuning* pada model bahasa dengan 70 miliar parameter memerlukan pembaruan seluruh 70 miliar bobot, menuntut ratusan gigabyte memori GPU hanya untuk menyimpan status optimizer (Adam).

```
+---------------------------------------------------------------------------------------------------------+
|                                MEKANISME DEKOMPOSISI MATRIKS LoRA                                       |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                      Masukan x (1 x d)                                                                  |
|                         ┬                                                                               |
|                         │                                                                               |
|            ┌────────────┴────────────┐                                                                  |
|            │ (Jalur Model Dasar)     │ (Jalur Adaptor LoRA)                                             |
|            ▼                         ▼                                                                  |
|      [ Bobot Dasar W_0 ]       [ Matriks A (r x d) ]  <-- Inisialisasi Gaussian                         |
|        (d x d) BEKU                  │                                                                  |
|      (Requires_Grad=False)           ▼                                                                  |
|            │                   [ Matriks B (d x r) ]  <-- Inisialisasi Nol                              |
|            │                         │                                                                  |
|            │                         ▼                                                                  |
|            │                   [ Skala (alpha / r) ]                                                    |
|            │                         │                                                                  |
|            └────────────┬────────────┘                                                                  |
|                         ▼                                                                               |
|                   Penjumlahan (+) ──► Keluaran h = x W_0 + (alpha / r) * x (B A)^T                      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Formulasi Matematis LoRA (Hu et al., 2021)
LoRA membekukan matriks bobot terlatih $W_0 \in \mathbb{R}^{d \times k}$ dan membatasi pembaruan bobot $\Delta W$ ke dalam representasi berperingkat rendah (*low intrinsic rank*):

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

di mana:
- $A \in \mathbb{R}^{r \times k}$ diinisialisasi menggunakan distribusi normal acak $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$ diinisialisasi dengan nilai **tepat nol** ($B = 0$).
- Rank $r \ll \min(d, k)$ (umumnya $r \in [4, 16]$).
- $\alpha$ adalah konstanta penskalaan (biasanya $\alpha = 2r$).

Karena $B$ bernilai nol pada awal pelatihan, $\Delta W = B \cdot A = 0$, performa awal model dasar terlindungi 100%. LoRA mereduksi parameter terlatih dan memori gradien hingga **$>95\%$**, memungkinkan fine-tuning model LLM pada komputer GPU konsumen standar.

---

## 6. Strategi Decoding & Sampling Pembangkitan Teks

Setelah model menghasilkan distribusi probabilitas, bagaimana kita memilih token berikutnya? Terdapat empat strategi utama:

```
+---------------------------------------------------------------------------------------------------------+
|                                STRATEGI DECODING PEMBANGKITAN TEKS                                      |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. GREEDY DECODING            : w_t = argmax P(w | C)                                                 |
|                                   - Sepenuhnya deterministik, cepat, namun kaku dan rentan looping.     |
|                                                                                                         |
|   2. TEMPERATURE SAMPLING       : P(w_i) = exp(z_i / T) / sum(exp(z_j / T))                             |
|                                   - T < 1.0 (Dingin): Mempertajam puncak (konservatif & faktual).       |
|                                   - T > 1.0 (Panas) : Meratakan kurva (kreatif & beragam).              |
|                                                                                                         |
|   3. TOP-K SAMPLING             : Memotong ekor distribusi hanya menyisakan k token teratas.            |
|                                   - Membuang token aneh, namun ukuran pool tidak adaptif.               |
|                                                                                                         |
|   4. TOP-P (NUCLEUS) SAMPLING   : Menyisakan subset terkecil dengan probabilitas kumulatif <= p.        |
|                                   - Adaptif dinamis: pool menyusut saat yakin, melebar saat ragu.       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 7. In-Context Learning & Prompt Engineering

Salah satu fenomena paling menakjubkan dari model bahasa skala besar adalah kemampuannya melakukan adaptasi tugas baru secara instan **tanpa memperbarui bobot model sama sekali** (*In-Context Learning* / ICL).

```
+---------------------------------------------------------------------------------------------------------+
|                                POLA IN-CONTEXT LEARNING & PROMPTING                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. ZERO-SHOT PROMPTING                                                                                |
|      Instruksi langsung tanpa contoh:                                                                   |
|      "Terjemahkan ke bahasa Indonesia: Machine learning is powerful."                                    |
|                                                                                                         |
|   2. FEW-SHOT PROMPTING (Brown et al., 2020)                                                            |
|      Menyertakan 2 - 5 demonstrasi pasangan (Input -> Output) untuk mengunci format dan penalaran:      |
|      "Input: Buruk -> Sentimen: Negatif"                                                                |
|      "Input: Bagus -> Sentimen: Positif"                                                                |
|      "Input: Memuaskan -> Sentimen:" ──► Model memprediksi: "Positif"                                   |
|                                                                                                         |
|   3. STRUCTURED JSON PROMPTING                                                                          |
|      Memaksa model mematuhi skema data yang dapat diuraikan oleh backend pemrograman:                   |
|      {"urgensi": "TINGGI", "kategori": "BILLING", "status": "PENDING"}                                  |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 8. Grounding & Retrieval-Augmented Generation (RAG)

Model bahasa murni bersifat *parametric memory*: seluruh pengetahuannya terkunci pada bobot neural network hasil pelatihan masa lalu (*knowledge cutoff*). Ketika ditanya fakta spesifik yang tidak diketahuinya, model rentan **berhalusinasi** (*hallucination*).

```
+---------------------------------------------------------------------------------------------------------+
|                                 ARSITEKTUR LENGKAP PIPELINE RAG                                         |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|    Pertanyaan Pengguna: "Berapa luas kampus ITERA di Lampung Selatan?"                                  |
|            │                                                                                            |
|            ▼                                                                                            |
|   [ 1. RETRIEVER ] ──► Mencari ke Basis Dokumen Akademik Internal ITERA (TF-IDF / Vector Search)        |
|            │                                                                                            |
|            ▼                                                                                            |
|   [ Dokumen Relevan Ditemukan ]: "Buku Panduan: Kampus ITERA memiliki luas lahan 287 hektar..."         |
|            │                                                                                            |
|            ▼                                                                                            |
|   [ 2. CONTEXT AUGMENTATION ]: Menyusun Grounded Prompt                                                 |
|        "Jawab pertanyaan HANYA berdasarkan dokumen berikut:                                             |
|         Dokumen: [Luas lahan 287 hektar...]                                                             |
|         Pertanyaan: Berapa luas kampus ITERA?"                                                          |
|            │                                                                                            |
|            ▼                                                                                            |
|   [ 3. LLM GENERATION ]: "Berdasarkan Buku Panduan, luas kampus ITERA adalah 287 hektar."               |
|        -> Faktual, Terverifikasi, dan Memiliki Atribusi Sumber yang Sah!                                |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 9. Taksonomi Halusinasi & Evaluasi Keandalan

Halusinasi terjadi ketika model menghasilkan pernyataan yang terdengar sangat meyakinkan dan fasih secara tata bahasa, namun sepenuhnya salah atau tidak didukung fakta.

| Jenis Halusinasi | Definisi Operasional | Contoh Kasus Nyata |
|---|---|---|
| **Intrinsic Hallucination** | Pernyataan yang secara langsung bertentangan dengan dokumen konteks yang disajikan. | Konteks: *"Pasien alergi penisilin"*, Model: *"Berikan injeksi penisilin"*. |
| **Extrinsic Hallucination** | Pernyataan yang mengarang fakta baru di luar dokumen konteks yang tidak dapat diverifikasi. | Model mengarang judul jurnal ilmiah atau nomor registrasi undang-undang fiktif. |

### Metrik Groundedness Score:
$$\text{Groundedness Score} = \frac{\text{Jumlah Klaim yang Terverifikasi Didukung Dokumen}}{\text{Total Klaim yang Dihasilkan}}$$
$$\text{Hallucination Rate} = 1.0 - \text{Groundedness Score}$$

---

## 10. Arsitektur Guardrails: Keamanan Masukan & Keluaran

Sistem kecerdasan buatan komersial tidak boleh menghubungkan pengguna langsung ke model mentah tanpa mekanisme pengamanan berlapis (*Defense in Depth*):

```
+---------------------------------------------------------------------------------------------------------+
|                                ARSITEKTUR KEAMANAN SISTEM GUARDRAILS                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|      [ Pengguna ]                                                                                       |
|           │                                                                                             |
|           ▼                                                                                             |
|   [ 1. INPUT GUARDRAILS ]                                                                               |
|        - Deteksi Prompt Injection ("Abaikan instruksi sistem...")                                       |
|        - Penyaringan Konten Berbahaya & Ujaran Kebencian                                                |
|        - Pembatasan Topik di Luar Cakupan (Out-of-Scope)                                                |
|           │ (Lolos)                                                                                     |
|           ▼                                                                                             |
|   [ 2. LLM REASONING CORE ]                                                                             |
|        - Model Transformer + Basis Konteks RAG                                                          |
|           │                                                                                             |
|           ▼                                                                                             |
|   [ 3. OUTPUT GUARDRAILS ]                                                                              |
|        - Redaksi Otomatis Data Pribadi / PII (NIK KTP, Email, Nomor Telepon)                            |
|        - Pemeriksaan Keselarasan Faktual (Hallucination Checker)                                        |
|        - Pemaksaan Skema Format JSON                                                                    |
|           │ (Aman)                                                                                      |
|           ▼                                                                                             |
|      [ Respon Akhir Aman Diterima Pengguna ]                                                            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 11. Tantangan Evaluasi Model Bahasa

Mengevaluasi LLM jauh lebih rumit daripada tugas supervised learning klasik karena bahasa bersifat terbuka (*open-ended*):

1. **Metrik Tumpang-Tindih Leksikal (BLEU / ROUGE)**: Mengukur kesamaan n-gram; murah secara komputasi namun gagal menangkap parafrasa semantik (*e.g.*, *"Aji belajar AI"* dan *"Aji mendalami kecerdasan buatan"* memiliki makna identik namun skor BLEU rendah).
2. **Exact Match & Structured Accuracy**: Cocok untuk tugas ekstraksi entitas, klasifikasi sentimen, dan pemenuhan skema JSON.
3. **LLM-as-a-Judge**: Menggunakan model bahasa tingkat frontier yang lebih kuat (*e.g.*, GPT-4) untuk mengevaluasi koherensi, kebenaran logika, dan kesopanan model yang lebih kecil berdasarkan rubrik terstandar.
4. **Human Evaluation**: Standar tertinggi (*gold standard*) audit kebenaran faktual, namun membutuhkan waktu dan biaya ahli yang besar.

---

## 12. Prinsip Responsible Generative AI

Penerapan Generative AI wajib mematuhi standar etika global (NIST AI Risk Management Framework & UNESCO):
1. **Transparansi**: Pengguna harus diinformasikan secara jelas saat berinteraksi dengan agen kecerdasan buatan.
2. **Keadilan (Fairness)**: Mengidentifikasi dan memitigasi bias demografis, gender, dan sosio-kultural yang terserap dari korpus internet.
3. **Privasi Data**: Mencegah kebocoran data rahasia institusi dan hak cipta karya intelektual.
4. **Pengawasan Manusia (*Human-in-the-Loop*)**: Keputusan kritis pada sektor kesehatan, hukum, dan keuangan wajib mempertahankan kewenangan verifikasi manusia.

---

## 13. Ringkasan & Diskusi Reflektif

### Intisari Bab:
- **Generative AI**: Memodelkan distribusi data $P(X)$ untuk menciptakan karya baru, berbeda dengan model diskriminatif $P(Y \mid X)$.
- **Autoregressive LLM**: Memprediksi token berikutnya secara berulang via arsitektur Transformer Decoder-Only.
- **Tiga Tahap Pelatihan**: Pre-training fondasi representasi $\rightarrow$ Instruction Tuning (SFT) pemahaman dialog $\rightarrow$ Alignment (RLHF/DPO) kepatuhan etika 3H.
- **LoRA**: Membekukan bobot dasar dan melatih adaptor rank rendah $B \cdot A$, menghemat memori hingga $>90\%$.
- **Decoding**: Greedy (deterministik), Temperature (kendali entropi), Top-k dan Top-p Nucleus (eliminasi ekor distribusi probabilitas).
- **RAG & Guardrails**: RAG mengaitkan model ke basis dokumen rujukan untuk memitigasi halusinasi, dipagari oleh Guardrails masukan dan keluaran.

### Pertanyaan Reflektif:
1. Jika model LLM dilatih murni untuk memprediksi token berikutnya, bagaimana kemampuan penalaran (*reasoning*) dapat muncul secara tiba-tiba saat ukuran model diperbesar (*emergent abilities*)?
2. Mengapa suhu $T = 0.0$ tidak selalu menjamin jawaban model 100% benar secara faktual?
3. Apa perbedaan fundamental antara memperbarui pengetahuan model via Fine-Tuning versus menyajikan dokumen via RAG?

---

## 14. Latihan Mandiri, Mini Project, Rubrik Penilaian & Referensi

### Soal Latihan Mandiri
Pelajari dan kerjakan soal latihan berjenjang (Tingkat 1 Dasar, Tingkat 2 Menengah, Tingkat 3 Lanjut) dengan kunci jawaban komprehensif pada berkas [practice.md](practice.md).

---

### Mini Project Chapter 29
**Judul**: *"Building and Evaluating a Small Language Generation System"*

**Panduan Lengkap**: Pelajari petunjuk implementasi, deliverable, dan kriteria penilaian pada berkas [practice.md](practice.md).

#### Rubrik Penilaian Proyek (100 Poin):
- **Problem & Data Understanding** (15 Poin): Perumusan masalah pemodelan bahasa lugas, dokumentasi korpus teks transparan, manajemen kosakata diskret rapi dengan token khusus.
- **Language Modeling Implementation** (25 Poin): Implementasi arsitektur Decoder-Only Transformer bersih (*clean code*), causal masking benar, siklus pelatihan CPU konvergen, dan pelaporan Perplexity valid.
- **Generation, Sampling & Grounding** (20 Poin): Perbandingan komprehensif strategi decoding (Greedy vs Temp vs Top-k vs Top-p), demonstrasi grounding fakta via retrieval passage terbukti membatasi jawaban.
- **Hallucination Evaluation & Guardrails** (25 Poin): Audit kuantitatif halusinasi (Groundedness Score & Hallucination Rate) mendalam, implementasi Input Guardrail anti-injection dan Output Guardrail sanitasi PII berfungsi sempurna.
- **Documentation, Ethics & References** (15 Poin): Kode modular terstruktur rapi, replikasi eksperimen 100% deterministik pada CPU, refleksi Responsible AI kritis, dan pencantuman 5 referensi akademik primer.

---

### Referensi Akademik Primer

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).** *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI Technical Report. [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. **Brown, T., Mann, B., Ryder, N., Subbiah, M., et al. (2020).** *Language Models are Few-Shot Learners* (GPT-3). Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 1877–1901. [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** *LoRA: Low-Rank Adaptation of Large Language Models.* International Conference on Learning Representations (ICLR 2022). [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
5. **Lewis, P., Perez, E., Piktus, A., Petroni, F., et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 9459–9474. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
6. **Jurafsky, D., & Martin, J. H. (2024).** *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (3rd ed. draft). Stanford University. [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
7. **ACM / IEEE-CS Joint Task Force on Computing Curricula. (2023).** *Computer Science Curricula 2023 (CS2023): Artificial Intelligence and Machine Learning Curricular Guidelines.* [https://csed.acm.org/](https://csed.acm.org/)

