# Panduan Praktik Laboratorium: Large Language Models & Generative AI

Selamat datang di modul panduan praktik laboratorium **Large Language Models & Generative AI** (Bab 29). Pada sesi laboratorium ini, Anda akan mempraktikkan arsitektur dan ekosistem komputasi model bahasa besar dan kecerdasan buatan generatif berbasis **Python**, **NumPy**, **PyTorch**, dan **Scikit-learn** pada jalur komputasi CPU: mulai dari mekanisme tokenisasi subword BPE, pemetaan Token ID dan manipulasi jendela konteks (*Context Window*), formulasi objektif *Next-Token Prediction* dan evaluasi *Perplexity* ($e^{\text{loss}}$), siklus pembangkitan teks autoregresif langkah-demi-langkah, analisis trade-off strategi decoding (Greedy vs Temperature Scaling vs Top-k vs Nucleus Top-p Sampling), rekayasa adaptasi efisien parameter via dekomposisi matriks LoRA (*Low-Rank Adaptation*), perancangan prompt terstruktur *In-Context Learning* (Zero-Shot, Few-Shot, JSON Schema), rancang bangun konsep *Retrieval-Augmented Generation* (RAG) untuk grounding fakta, kerangka kerja audit dan deteksi halusinasi, sistem pertahanan keamanan Guardrails (pencegahan *Prompt Injection* dan redaksi data pribadi PII), hingga perakitan dan evaluasi model *Decoder-Only Tiny Transformer* mandiri secara deterministik dan terverifikasi.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- **Python**: Versi 3.10+ (Direkomendasikan Python 3.12).
- **Pustaka Utama**: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `scikit-learn>=1.3.0` (teruji 1.9.1), `numpy>=2.0.0` (teruji 2.5.4).
- **Akselerator Perangkat Keras**: Bersifat opsional. Seluruh 14 skrip Python, modul LoRA, demonstrasi RAG, dan model TinyGPT dirancang khusus agar dapat dieksekusi secara instan pada CPU komputer standar dalam hitungan detik tanpa mewajibkan GPU, tanpa model unduhan raksasa, dan tanpa API berbayar.

### Struktur Direktori Bab 29:
```
content/level-3-deep-learning/chapter-29/
├── code/
│   ├── README.md
│   ├── tokenization_demo.py
│   ├── next_token_prediction.py
│   ├── language_model_loss.py
│   ├── autoregressive_generation.py
│   ├── greedy_decoding.py
│   ├── temperature_sampling.py
│   ├── top_k_sampling.py
│   ├── top_p_sampling.py
│   ├── lora_matrix_demo.py
│   ├── prompting_examples.py
│   ├── toy_retrieval_grounding.py
│   ├── hallucination_evaluation.py
│   ├── output_validation.py
│   └── tiny_transformer_generation.py
├── experiments/
│   └── results_ch29.json
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_29_companion.ipynb
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

### Langkah 1: Tokenisasi Subword & Manajemen Jendela Konteks
Pelajari bagaimana teks mentah dipecah menjadi token-token leksikal diskret dan dipetakan ke Token ID:
```bash
python content/level-3-deep-learning/chapter-29/code/tokenization_demo.py
```
- **Fokus Pengamatan**:
  - Amati segmentasi subword berbasis konsep BPE untuk memecah kata berafiks (*e.g.*, *'bantuan'* $\rightarrow$ `['bantu', '##an']`).
  - Pahami peran token khusus: `<pad>` (0), `<unk>` (1), `<bos>` (2), `<eos>` (3).
  - Perhatikan pembuatan *Attention Mask* (1 untuk token riil, 0 untuk token `<pad>`).

### Langkah 2: Next-Token Prediction & Probabilitas Bersyarat
Pahami bagaimana proyeksi logits dikonversi menjadi distribusi probabilitas kondisional $P(w_t \mid w_{<t})$:
```bash
python content/level-3-deep-learning/chapter-29/code/next_token_prediction.py
```
- **Fokus Pengamatan**:
  - Amati proyeksi vektor tersembunyi teratas ke ruang kosakata berukuran $|V|$.
  - Perhatikan normalisasi Softmax: jumlah total probabilitas seluruh kosakata tepat bernilai $1.000000$.
  - Buktikan bahwa LLM memodelkan probabilitas bersyarat kontekstual dan bukan sekadar frekuensi global.

### Langkah 3: Perhitungan Loss & Perplexity Model Bahasa
Kalkulasikan Cross-Entropy Loss, NLL, dan Perplexity secara analitis:
```bash
python content/level-3-deep-learning/chapter-29/code/language_model_loss.py
```
- **Fokus Pengamatan**:
  - Telusuri perhitungan Negative Log-Likelihood pada setiap langkah waktu: $\text{NLL}_t = -\ln P(w_t^*)$.
  - Hitung metrik Perplexity: $\text{PPL} = \exp(\text{CrossEntropy})$.
  - Bandingkan model terlatih baik ($\text{PPL} \approx 1.4$) terhadap model acak ($\text{PPL} \approx 9.0$).

### Langkah 4: Siklus Pembangkitan Autoregresif Langkah-demi-Langkah
Simulasikan proses pembangkitan teks rekursif:
```bash
python content/level-3-deep-learning/chapter-29/code/autoregressive_generation.py
```
- **Fokus Pengamatan**:
  - Amati bagaimana token yang baru saja diprediksi disisipkan kembali ke dalam jendela konteks masukan.
  - Periksa kondisi terminasi: proses berhenti otomatis saat model membangkitkan token `<eos>` atau mencapai batas `max_new_tokens`.

### Langkah 5: Greedy Decoding & Jebakan Pengulangan (Repetition Loop)
Pelajari sifat deterministik argmax dan kelemahan fatalnya:
```bash
python content/level-3-deep-learning/chapter-29/code/greedy_decoding.py
```
- **Fokus Pengamatan**:
  - Buktikan bahwa 3 kali eksekusi berturut-turut menghasilkan 100% token yang identik.
  - Amati fenomena *degenerate repetition*: model terjebak dalam pengulangan frasa tanpa akhir (*e.g.*, *"sangat hebat dan sangat hebat dan..."*).

### Langkah 6: Penskalaan Temperature & Pengendalian Entropi
Kendalikan tingkat kreativitas dan ketajaman distribusi probabilitas:
```bash
python content/level-3-deep-learning/chapter-29/code/temperature_sampling.py
```
- **Fokus Pengamatan**:
  - Analisis efek suhu dingin ($T=0.2$): probabilitas memusat pada kata puncak (entropi Shannon rendah).
  - Analisis efek suhu netral ($T=1.0$) vs suhu panas ($T=1.8$): distribusi mendekati seragam.
  - Uji Monte Carlo ($N=2000$ sampel): amati variasi kata terpilih pada suhu berbeda.

### Langkah 7: Pemangkasan Kosakata Top-k Sampling
Eliminasi ekor distribusi probabilitas yang tidak relevan:
```bash
python content/level-3-deep-learning/chapter-29/code/top_k_sampling.py
```
- **Fokus Pengamatan**:
  - Amati substitusi nilai logits di luar $k$ teratas menjadi $-\infty$.
  - Periksa renormalisasi probabilitas: buktikan bahwa hanya $k$ token aktif yang bersaing.
  - Bandingkan $k=1$ (identik greedy), $k=3$, dan $k=5$.

### Langkah 8: Nucleus Top-p Sampling Adaptif Dinamis
Pelajari strategi pemotongan probabilitas kumulatif:
```bash
python content/level-3-deep-learning/chapter-29/code/top_p_sampling.py
```
- **Fokus Pengamatan**:
  - Buktikan adaptabilitas dinamis: pada saat model sangat yakin (*peaky*), ukuran kandidat menyusut otomatis menjadi 1–2 token.
  - Pada saat model ragu-ragu (*flat*), ukuran kandidat melebar otomatis mencakup banyak variasi kata yang masuk akal.

### Langkah 9: Low-Rank Adaptation (LoRA) dari Dasar PyTorch
Rancang adaptor efisien parameter dan buktikan penghematan memori:
```bash
python content/level-3-deep-learning/chapter-29/code/lora_matrix_demo.py
```
- **Fokus Pengamatan**:
  - Periksa pembekuan model dasar: `base_layer.weight.requires_grad = False`.
  - Telusuri dekomposisi matriks $\Delta W = \frac{\alpha}{r} (B \cdot A)$ dengan $r=4$.
  - Buktikan penghematan parameter: **96.88% parameter dihemat** berbanding fine-tuning penuh.
  - Verifikasi aliran gradien: hanya adaptor $A$ dan $B$ yang menerima gradien selama *backpropagation*.

### Langkah 10: In-Context Learning & Format Terstruktur
Rancang prompt Zero-Shot, Few-Shot, dan pemaksaan format JSON:
```bash
python content/level-3-deep-learning/chapter-29/code/prompting_examples.py
```
- **Fokus Pengamatan**:
  - Pahami bagaimana demonstrasi Few-Shot mengunci gaya dan format luaran tanpa mengubah bobot model.
  - Rancang prompt ekstraksi entitas informasi dan tiket komplain terstruktur.

### Langkah 11: Grounding Fakta via Retrieval-Augmented Generation (RAG)
Bangun sistem tanya-jawab berdasar fakta dokumen:
```bash
python content/level-3-deep-learning/chapter-29/code/toy_retrieval_grounding.py
```
- **Fokus Pengamatan**:
  - Telusuri tahap Retrieval menggunakan TF-IDF pada basis pengetahuan dokumen akademik ITERA.
  - Amati penyusunan *Grounded Prompt* yang membatasi jawaban hanya pada teks rujukan.
  - Bandingkan jawaban berbasis RAG (faktual & teratribusi) vs jawaban tanpa RAG (halusinasi angka).

### Langkah 12: Kerangka Kerja Evaluasi Halusinasi
Kuantifikasi kebenaran faktual luaran model bahasa:
```bash
python content/level-3-deep-learning/chapter-29/code/hallucination_evaluation.py
```
- **Fokus Pengamatan**:
  - Dekomposisi respon ke dalam klaim-klaim individual diskret.
  - Verifikasi klaim terhadap dokumen acuan (*Ground Truth*).
  - Hitung metrik *Groundedness Score* dan *Hallucination Rate*.

### Langkah 13: Arsitektur Guardrails (Pertahanan Masukan & Keluaran)
Terapkan prinsip *Defense-in-Depth* pada ekosistem LLM:
```bash
python content/level-3-deep-learning/chapter-29/code/output_validation.py
```
- **Fokus Pengamatan**:
  - *Input Guardrail*: Deteksi dan penolakan upaya *Prompt Injection* (*"abaikan instruksi sebelumnya"*).
  - *Output Guardrail*: Penyensoran otomatis data pribadi PII (NIK KTP, Email, Nomor Telepon).
  - Pemaksaan validasi skema JSON luaran.

### Langkah 14: Model Tiny Transformer Generation Utuh
Latih dan jalankan model Transformer Decoder-Only mini pada CPU:
```bash
python content/level-3-deep-learning/chapter-29/code/tiny_transformer_generation.py
```
- **Fokus Pengamatan**:
  - Model mini TinyGPT: 2 blok Transformer, 2 kepala atensi, $d_{\text{model}}=32$ (~11.600 parameter).
  - Latih pada korpus mini edukasi dalam ~0.3 detik pada CPU (loss turun stabil).
  - Hasilkan kalimat baru secara autoregresif menggunakan temperature dan top-k.

---

## 3. Soal Latihan Mandiri Terstruktur

### Tingkat 1: Fondasi Matematika & Konseptual (Dasar)
1. **Perhitungan Probabilitas Rantai Kalimat**:
   Misalkan sebuah model bahasa menetapkan probabilitas bersyarat berikut:
   - $P(\text{'kecerdasan'} \mid \text{'<bos>'}) = 0.40$
   - $P(\text{'buatan'} \mid \text{'<bos>', 'kecerdasan'}) = 0.80$
   - $P(\text{'cerdas'} \mid \text{'<bos>', 'kecerdasan', 'buatan'}) = 0.25$
   - $P(\text{'<eos>'} \mid \text{'<bos>', 'kecerdasan', 'buatan', 'cerdas'}) = 0.50$
   Hitung probabilitas bersama (*joint probability*) dari kalimat *"kecerdasan buatan cerdas"*!
   - *Jawaban*:
     - Berdasarkan aturan rantai probabilitas:
       $$P(W) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_{1:2}) \cdot P(w_4 \mid w_{1:3})$$
     - $P(W) = 0.40 \times 0.80 \times 0.25 \times 0.50 = 0.0400$ ($4.0\%$).

2. **Perbedaan Mendasar Discriminative vs Generative AI**:
   Jelaskan secara matematis perbedaan antara model klasifikasi ResNet pada Bab 24 dengan model bahasa GPT pada Bab 29!
   - *Jawaban*:
     - ResNet adalah **model diskriminatif** yang memodelkan probabilitas posterior $P(Y \mid X)$, yaitu mencari batas keputusan untuk membedakan label $Y$ berdasarkan citra masukan $X$.
     - GPT adalah **model generatif** yang memodelkan distribusi probabilitas gabungan atau data $P(X) = \prod P(x_t \mid x_{<t})$, yang mampu mensimulasikan proses pembentukan data baru berdistribusi serupa.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Halusinasi pada Aplikasi Customer Service**:
   Sebuah model LLM diimplementasikan untuk menjawab pertanyaan pelanggan mengenai jam operasional toko. Meskipun dalam prompt telah dituliskan *"Toko buka jam 08.00 - 17.00"*, model terkadang menjawab *"Toko kami buka 24 jam setiap hari"*. Jelaskan mengapa hal ini terjadi dan bagaimana memperbaikinya!
   - *Jawaban*:
     - *Akar Masalah*: Model mengalami *prior knowledge dominance* (pengetahuan umum internet saat pre-training mengenai toko online 24 jam mendominasi konteks prompt) atau suhu pembangkitan (*temperature*) disetel terlalu tinggi ($T \ge 1.0$) sehingga token berkemungkinan acak terpilih.
     - *Solusi Perbaikan*:
       1. Turunkan nilai Temperature mendekati nol ($T = 0.0$ atau $0.1$) untuk memaksa decoding greedy yang kaku.
       2. Terapkan instruksi batas negatif yang tegas: *"Hanya gunakan informasi jam operasional di atas. Dilarang mengasumsikan toko buka 24 jam."*
       3. Pasang *Output Guardrail* berbasis regex/pencocokan fakta untuk memverifikasi angka jam sebelum teks dikirimkan ke pelanggan.

2. **Dilema LoRA: Pemilihan Rank $r$ dan Alpha $\alpha$**:
   Jelaskan dampak pemilihan rank $r$ yang terlalu kecil ($r=1$) versus terlalu besar ($r=128$) terhadap performa dan efisiensi memori fine-tuning LoRA!
   - *Jawaban*:
     - Jika $r=1$: Model mengalami *underfitting* representasi; kapasitas adaptasi terlalu rendah untuk mempelajari domain spesifik yang kompleks.
     - Jika $r=128$: Efisiensi memori hilang; parameter membengkak mendekati full fine-tuning, meningkatkan risiko *overfitting* dan memperlambat proses pelatihan.
     - Standar praktis: Rentang $r \in [4, 16]$ dengan skala $\alpha = 2r$ terbukti secara empiris memberikan titik kompromi optimal untuk mayoritas tugas NLP hilir.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Analisis Matematis Efisiensi Parameter LoRA**:
   Misalkan sebuah model LLM memiliki dimensi representasi $d_{\text{model}} = 4.096$ dan memiliki 32 lapisan Transformer. Setiap lapisan memiliki 4 matriks proyeksi atensi ($W_Q, W_K, W_V, W_O$) berukuran $(4.096 \times 4.096)$. Hitung total parameter bobot atensi model dasar, dan buktikan berapa persen penghematan parameter jika seluruh proyeksi tersebut dipasangi adaptor LoRA berperingkat $r=8$!
   - *Jawaban*:
     - **Parameter Model Dasar per Lapisan**:
       $$4 \times (4.096 \times 4.096) = 4 \times 16.777.216 = 67.108.864 \text{ bobot}$$
     - **Total Parameter Dasar untuk 32 Lapisan**:
       $$32 \times 67.108.864 = 2.147.483.648 \text{ bobot} \approx 2.15 \text{ Miliar Parameter}$$
     - **Parameter LoRA per Matriks Proyeksi ($r=8$)**:
       $$\text{Param}(A) + \text{Param}(B) = (8 \times 4.096) + (4.096 \times 8) = 32.768 + 32.768 = 65.536 \text{ bobot}$$
     - **Total Parameter LoRA untuk 32 Lapisan**:
       $$32 \times 4 \times 65.536 = 128 \times 65.536 = 8.388.608 \text{ bobot} \approx 8.39 \text{ Juta Parameter}$$
     - **Persentase Penghematan Parameter**:
       $$\text{Hemat} = \left(1 - \frac{8.388.608}{2.147.483.648}\right) \times 100\% = (1 - 0.00390625) \times 100\% = 99.61\%!$$
     - *Kesimpulan*: LoRA mereduksi parameter terlatih sebesar **$99.61\%$**, memungkinkan fine-tuning model miliaran parameter pada perangkat keras konsumen dengan VRAM terbatas.

---

## 4. Mini Project Chapter 29

### Judul Tugas Proyek:
**"Building and Evaluating a Small Language Generation System"**

### Deskripsi Tugas:
Rancang, bangun, verifikasi, dan evaluasi sistem pembangkitan bahasa mini berbasis **Python**, **NumPy**, dan **PyTorch** pada CPU yang menerapkan siklus hidup pemodelan generatif modern:
1. **Definisi Masalah & Korpus**: Tentukan domain teks spesifik (misal ringkasan berita akademik, pepatah, atau panduan FAQ kampus) dan dokumentasikan kosakata korpus.
2. **Tokenisasi & Representasi**: Bangun modul tokenisasi terstandar dengan penanganan token khusus (`<bos>`, `<eos>`, `<pad>`).
3. **Arsitektur Model Bahasa**: Bangun model Decoder-Only Transformer mini (TinyGPT) dengan Causal Self-Attention.
4. **Pelatihan & Loss Tracking**: Latih model menggunakan Cross-Entropy Loss dan laporkan kurva penurunan Perplexity pada CPU.
5. **Eksperimen Sampling**: Bandingkan luaran pembangkitan teks di bawah 4 konfigurasi: Greedy, Temperature rendah ($T=0.3$), Temperature tinggi ($T=1.5$), dan Top-p Nucleus ($p=0.85$).
6. **Implementasi Grounding Sederhana**: Integrasikan modul temu-kembali (retrieval) passage bukti fakta.
7. **Audit Halusinasi**: Hitung *Groundedness Score* dan identifikasi klaim yang tidak didukung data sumber.
8. **Pertahanan Guardrails**: Pasang penyaring masukan prompt injection dan sanitasi keluaran data pribadi (PII).
9. **Responsible AI**: Susun analisis risiko bias dan batas keandalan sistem.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Problem & Data Understanding** | 15 Poin | Perumusan masalah pemodelan bahasa lugas, dokumentasi korpus teks transparan, manajemen kosakata diskret rapi dengan token-token khusus. |
| **2. Language Modeling Implementation** | 25 Poin | Implementasi arsitektur Decoder-Only Transformer bersih (*clean code*), causal masking segitiga bawah benar, siklus pelatihan CPU konvergen, dan pelaporan Perplexity valid. |
| **3. Generation, Sampling & Grounding** | 20 Poin | Perbandingan komprehensif strategi decoding (Greedy vs Temp vs Top-k vs Top-p), demonstrasi grounding fakta via retrieval passage terbukti membatasi jawaban. |
| **4. Hallucination Evaluation & Guardrails** | 25 Poin | Audit kuantitatif halusinasi (Groundedness Score & Hallucination Rate) mendalam, implementasi Input Guardrail anti-injection dan Output Guardrail sanitasi PII berfungsi sempurna. |
| **5. Documentation, Ethics & References** | 15 Poin | Kode modular terstruktur rapi, replikasi eksperimen 100% deterministik pada CPU, refleksi Responsible AI kritis, dan pencantuman 5 referensi akademik primer. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 14 skrip Python di direktori `content/level-3-deep-learning/chapter-29/code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Tokenisasi, kosakata, dan context window tervalidasi pada `tokenization_demo.py`.
- [ ] Distribusi probabilitas conditional next-token terbukti memenuhi $\sum P = 1.0$ pada `next_token_prediction.py`.
- [ ] Perhitungan Cross-Entropy Loss dan Perplexity tervalidasi pada `language_model_loss.py`.
- [ ] Siklus autoregresif berhenti tepat saat token `<eos>` pada `autoregressive_generation.py`.
- [ ] Sifat deterministik dan repetisi greedy decoding terbukti pada `greedy_decoding.py`.
- [ ] Pengendalian entropi via Temperature scaling tervalidasi pada `temperature_sampling.py`.
- [ ] Pemangkasan ekor distribusi Top-k tervalidasi pada `top_k_sampling.py`.
- [ ] Adaptabilitas dinamis pool Top-p tervalidasi pada `top_p_sampling.py`.
- [ ] Penghematan parameter LoRA (>90%) dan isolasi gradien terbukti pada `lora_matrix_demo.py`.
- [ ] Pola Zero-Shot, Few-Shot, dan format JSON terstruktur teruji pada `prompting_examples.py`.
- [ ] Pipeline RAG memitigasi halusinasi via bukti passage pada `toy_retrieval_grounding.py`.
- [ ] Audit kuantitatif Groundedness Score terverifikasi pada `hallucination_evaluation.py`.
- [ ] Guardrails masukan (anti-injection) dan keluaran (redaksi PII) aktif pada `output_validation.py`.
- [ ] Model Tiny Transformer Decoder-Only konvergen dan membangkitkan teks pada `tiny_transformer_generation.py`.
- [ ] Notebook pendamping `chapter_29_companion.ipynb` tervalidasi bersih dalam format v4.
- [ ] Slide Reveal.js `deck.json` memuat 33 slide semantik terstruktur rapi.

