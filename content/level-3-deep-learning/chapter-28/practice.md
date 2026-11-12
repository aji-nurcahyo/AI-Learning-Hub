# Panduan Praktik Laboratorium: Attention & Transformer Architecture

Selamat datang di modul panduan praktik laboratorium **Attention & Transformer Architecture** (Bab 28). Pada sesi laboratorium ini, Anda akan mempraktikkan fondasi revolusioner mekanisme atensi (*Attention Mechanism*) dan arsitektur **Transformer** berbasis **Python**, **NumPy**, dan **PyTorch** pada jalur komputasi CPU: mulai dari perumusan aljabar linier Query-Key-Value, pembuktian analitis penskalaan variansi $\sqrt{d_k}$, stabilitas numerik fungsi Softmax (*shift-max trick*), mekanika Self-Attention dan Cross-Attention, perancangan Padding Mask dan Causal Mask segitiga bawah, implementasi Multi-Head Attention dari dasar tanpa pustaka (*from scratch*), pembedahan `nn.MultiheadAttention` PyTorch, pembangkitan Positional Encoding sinusoidal (Vaswani et al., 2017), penyusunan blok pembangun lengkap Transformer Encoder dan Transformer Decoder dengan koneksi residu dan Layer Normalization, orkestrasi arsitektur Seq2Seq Mini Transformer, hingga pengujian deterministik verifikasi kesetaraan numerik ketat (deviasi $< 1.0 \times 10^{-6}$).

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- **Python**: Versi 3.10+ (Direkomendasikan Python 3.12).
- **Pustaka Utama**: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `numpy>=2.0.0` (teruji pada NumPy 2.5.4).
- **Akselerator Perangkat Keras**: Bersifat opsional. Seluruh 13 skrip Python, modul Encoder-Decoder, dan pengujian kesetaraan numerik dirancang khusus agar dapat dieksekusi secara instan dan deterministik pada CPU komputer standar dalam hitungan detik tanpa mewajibkan GPU.

### Struktur Direktori Bab 28:
```
content/level-3-deep-learning/chapter-28/
├── code/
│   ├── README.md
│   ├── dot_product_attention.py
│   ├── scaled_dot_product_attention.py
│   ├── stable_attention_softmax.py
│   ├── self_attention.py
│   ├── cross_attention.py
│   ├── attention_masks.py
│   ├── multi_head_attention_numpy.py
│   ├── multi_head_attention_torch.py
│   ├── positional_encoding.py
│   ├── transformer_encoder.py
│   ├── transformer_decoder.py
│   ├── transformer_architecture_demo.py
│   └── attention_numerical_verification.py
├── experiments/
│   └── results_ch28.json
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_28_companion.ipynb
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

### Langkah 1: Anatomi Query-Key-Value & Dot-Product Attention Dasar
Pelajari peran analogi pustaka (*search query*, *catalog key*, *content value*) dan perhitungan perkalian titik matriks:
```bash
python content/level-3-deep-learning/chapter-28/code/dot_product_attention.py
```
- **Fokus Pengamatan**:
  - Amati bagaimana skor keselarasan dihitung melalui perkalian matriks $S = Q K^T$.
  - Perhatikan transformasi baris skor menjadi distribusi probabilitas stokastik menggunakan Softmax ($\sum_j A_{ij} = 1.0$).
  - Periksa agregasi nilai $O = A V$ sebagai penjumlahan berbobot adaptif dari representasi Value.

### Langkah 2: Mengapa Perlu Faktor Penskalaan $\sqrt{d_k}$?
Buktikan secara empiris fenomena lonjakan variansi dan kepunahan gradien (*gradient vanishing*) pada dimensi tinggi:
```bash
python content/level-3-deep-learning/chapter-28/code/scaled_dot_product_attention.py
```
- **Fokus Pengamatan**:
  - Jalankan simulasi Monte Carlo $10.000$ sampel: amati bahwa hasil kali titik vektor acak berdimensi $d_k=64$ memiliki variansi sebesar $\approx 64.92$.
  - Perhatikan bahwa pembagian dengan $\sqrt{d_k} = 8$ menormalkan variansi kembali ke $\approx 1.01$.
  - Uji perbandingan bobot Softmax: tanpa penskalaan, skor ekstrem menghasilkan distribusi tajam *winner-takes-all* (elemen non-maksimum bernilai $10^{-9}$ hingga $10^{-44}$), mematikan perambatan gradien *backpropagation*.

### Langkah 3: Stabilitas Numerik Softmax (*Shift-Max Trick*)
Pelajari kegagalan aritmetika komputer floating-point dan implementasi fungsi Softmax anti-overflow:
```bash
python content/level-3-deep-learning/chapter-28/code/stable_attention_softmax.py
```
- **Fokus Pengamatan**:
  - Berikan input ekstrem seperti $[1000.0, 1005.0, 995.0]$.
  - Amati bagaimana Softmax naif menghasilkan `nan` (*Not a Number*) akibat $e^{1000}$ melampaui batas representasi IEEE 754 float64 ($\approx 10^{308}$).
  - Buktikan bahwa trik penggeseran nilai maksimum $\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$ menghasilkan probabilitas yang valid dan stabil secara deterministik.

### Langkah 4: Rancang Bangun Self-Attention dengan Matriks Proyeksi
Eksplorasi interaksi kontekstual intra-sekuens dengan matriks proyeksi terpelajari:
```bash
python content/level-3-deep-learning/chapter-28/code/self_attention.py
```
- **Fokus Pengamatan**:
  - Amati transformasi input $\mathbf{X}$ berdimensi $(B, T, d_{\text{in}})$ menjadi $Q, K, V$ berdimensi $(B, T, d_{\text{attn}})$ menggunakan $W_Q, W_K, W_V$.
  - Pelajari peran matriks proyeksi keluaran $W_O$ dalam memetakan konteks kembali ke dimensi representasi asli.
  - Perhatikan bagaimana representasi akhir setiap token merupakan perpaduan informasi dari seluruh token dalam kalimat yang sama.

### Langkah 5: Mekanisme Cross-Attention (Decoder-Encoder Interplay)
Implementasikan jembatan informasi antara sekuens Decoder dan memori representasi Encoder:
```bash
python content/level-3-deep-learning/chapter-28/code/cross_attention.py
```
- **Fokus Pengamatan**:
  - Amati perbedaan dimensi sekuens: Query berasal dari Decoder ($T_{\text{dec}} = 3$), sedangkan Key dan Value berasal dari Encoder ($T_{\text{enc}} = 4$).
  - Perhatikan bentuk matriks bobot atensi $(B, T_{\text{dec}}, T_{\text{enc}})$, di mana setiap token luaran Decoder bebas mengonsultasikan seluruh posisi masukan Encoder.

### Langkah 6: Rancang Bangun Attention Masks (Padding & Causal)
Terapkan penyaringan posisi ilegal agar representasi model mematuhi integritas linguistik dan kausalitas waktu:
```bash
python content/level-3-deep-learning/chapter-28/code/attention_masks.py
```
- **Fokus Pengamatan**:
  - Amati substitusi skor posisi terlarang dengan nilai penalti negatif raksasa ($-10^9$ atau $-\infty$) sebelum dievaluasi oleh fungsi Softmax ($e^{-\infty} = 0$).
  - Periksa *Padding Mask*: buktikan bahwa posisi token `<pad>` menghasilkan bobot atensi tepat $0.0000$.
  - Periksa *Causal Mask*: amati matriks segitiga bawah (*lower-triangular matrix*), di mana token pada posisi $i$ dilarang keras mengamati token masa depan pada posisi $j > i$.

### Langkah 7: Algoritma Multi-Head Attention dari Dasar NumPy
Bangun modul Multi-Head Attention paralel tanpa ketergantungan pada pustaka deep learning:
```bash
python content/level-3-deep-learning/chapter-28/code/multi_head_attention_numpy.py
```
- **Fokus Pengamatan**:
  - Pelajari teknik manipulasi bentuk tensor (*tensor reshaping & axis swapping*): membagi dimensi $d_{\text{model}}$ menjadi $h$ kepala masing-masing berdimensi $d_k$ sehingga menghasilkan tensor 4D `(Batch, Heads, SeqLen, d_k)`.
  - Amati bagaimana operasi *scaled dot-product attention* dieksekusi secara paralel di seluruh kepala dalam satu operasi aljabar linier matriks.
  - Amati rekombinasi tensor (*concatenation*) dan proyeksi akhir via matriks $W_O$.
  - Periksa diversitas atensi: buktikan bahwa kepala yang berbeda memberikan bobot fokus pada posisi token yang berbeda (*subspace specialization*).

### Langkah 8: Anatomi PyTorch nn.MultiheadAttention
Bedah representasi internal modul MHA standar produksi PyTorch:
```bash
python content/level-3-deep-learning/chapter-28/code/multi_head_attention_torch.py
```
- **Fokus Pengamatan**:
  - Pahami alasan PyTorch menggabungkan $W_Q, W_K, W_V$ ke dalam satu parameter tunggal `in_proj_weight` berdimensi $(3 \cdot d_{\text{model}}, d_{\text{model}})$ demi efisiensi transfer memori GPU/CPU.
  - Verifikasi rumus analitis jumlah parameter:
    $$\text{Parameter MHA} = 4 \cdot d_{\text{model}}^2 + 4 \cdot d_{\text{model}}$$
    (Untuk $d_{\text{model}}=16$: $4 \times 256 + 64 = 1.088$ parameter).
  - Terapkan parameter `batch_first=True` agar orientasi dimensi tensor konsisten `(Batch, SeqLen, Dim)`.

### Langkah 9: Positional Encoding Sinusoidal (Vaswani et al., 2017)
Buktikan sifat invarian permutasi dari Self-Attention dan suntikkan urutan posisi via fungsi trigonometri:
```bash
python content/level-3-deep-learning/chapter-28/code/positional_encoding.py
```
- **Fokus Pengamatan**:
  - Uji sifat invarian permutasi: jika urutan baris masukan diacak, keluaran Self-Attention murni ikut teracak persis sama tanpa kesadaran sintaksis urutan kalimat.
  - Implementasikan formulasi sinusoidal:
    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
  - Buktikan bahwa nilai kesamaan kosinus antar-vektor posisi yang berdekatan ($pos=1$ dan $pos=2$: $0.8841$) jauh lebih tinggi daripada posisi yang berjauhan ($pos=1$ dan $pos=8$: $0.2156$).
  - Buktikan identitas trigonometri pergeseran linier: posisi $pos + k$ dapat dinyatakan sebagai transformasi linier dari posisi $pos$.

### Langkah 10: Rancang Bangun Blok Transformer Encoder
Integrasikan MHA, Residual Connection, Layer Normalization, dan Feed-Forward Network:
```bash
python content/level-3-deep-learning/chapter-28/code/transformer_encoder.py
```
- **Fokus Pengamatan**:
  - Amati arsitektur Post-LN / Pre-LN: $x = \text{LayerNorm}(x + \text{MHA}(x))$ dan $x = \text{LayerNorm}(x + \text{FFN}(x))$.
  - Periksa kestabilan statistik Layer Normalization: rata-rata per token stabil di $\mu \approx 0.0$ dan standar deviasi $\sigma \approx 1.0$.
  - Hitung parameter analitis: untuk $d_{\text{model}}=16, d_{\text{ff}}=32$, total parameter blok Encoder adalah tepat $3.280$.

### Langkah 11: Rancang Bangun Blok Transformer Decoder
Susun blok Decoder lengkap dengan Masked MHA, Cross-Attention, dan FFN:
```bash
python content/level-3-deep-learning/chapter-28/code/transformer_decoder.py
```
- **Fokus Pengamatan**:
  - Telusuri 3 sub-lapisan bertingkat: Masked Self-Attention $\rightarrow$ Cross-Attention $\rightarrow$ Position-wise Feed-Forward.
  - Amati penerapan Causal Mask pada Masked Self-Attention untuk memelihara sifat autoregresif selama proses *decoding*.
  - Hitung parameter analitis: untuk konfigurasi yang sama, blok Decoder memiliki tepat $4.400$ parameter (memuat 2 modul MHA dan 3 LayerNorm).

### Langkah 12: Orkestrasi Seq2Seq Mini Transformer Lengkap
Rakit arsitektur Seq2Seq terpadu dan ukur profil latensi inferensi pada CPU:
```bash
python content/level-3-deep-learning/chapter-28/code/transformer_architecture_demo.py
```
- **Fokus Pengamatan**:
  - Amati alur ujung-ke-ujung (*end-to-end*): Token Embedding $\rightarrow$ Positional Encoding $\rightarrow$ Encoder Stacks $\rightarrow$ Decoder Stacks $\rightarrow$ Linear Projection Head $\rightarrow$ Logits Kosakata.
  - Catat latensi komputasi inferensi lintasan maju (*forward pass*) pada CPU: rata-rata hanya $\approx 1.34$ ms per batch.

### Langkah 13: Audit Verifikasi Kesetaraan Numerik Ketat
Buktikan integritas ilmiah implementasi scratch Anda terhadap standar industri PyTorch:
```bash
python content/level-3-deep-learning/chapter-28/code/attention_numerical_verification.py
```
- **Fokus Pengamatan**:
  - Amati sinkronisasi deterministik seluruh bobot dan bias antara modul NumPy mandiri dan `nn.MultiheadAttention` PyTorch.
  - Periksa hasil pengujian dua skenario utama:
    - *Kasus A (Unmasked)*: Deviasi maksimum tensor keluaran $7.82 \times 10^{-8}$ dan peta atensi $1.19 \times 10^{-7}$.
    - *Kasus B (Causal Masked)*: Deviasi maksimum tensor keluaran $7.60 \times 10^{-8}$ dan peta atensi $6.40 \times 10^{-8}$.
  - Pastikan seluruh deviasi berada jauh di bawah ambang batas toleransi ketat $1.0 \times 10^{-6}$ (**Audit 100% Lulus**).

---

## 3. Soal Latihan Mandiri Terstruktur

### Tingkat 1: Fondasi Matematika & Konseptual (Dasar)
1. **Derivasi Variansi Produk Titik**:
   Misalkan $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ adalah dua vektor acak di mana setiap elemen $q_i$ dan $k_j$ berdistribusi independen dan identik (i.i.d.) dengan ekspektasi $\mathbb{E}[q_i] = \mathbb{E}[k_j] = 0$ dan variansi $\operatorname{Var}(q_i) = \operatorname{Var}(k_j) = 1$. Buktikan bahwa:
   $$\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$$
   - *Jawaban*:
     - Hasil kali titik didefinisikan sebagai $S = \sum_{i=1}^{d_k} q_i k_i$.
     - Karena $q_i$ dan $k_i$ independen dengan rata-rata 0, maka $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$.
     - Variansi dari perkalian dua variabel acak independen ber-mean 0 adalah:
       $$\operatorname{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = \operatorname{Var}(q_i)\operatorname{Var}(k_i) = 1 \cdot 1 = 1$$
     - Karena setiap suku dalam penjumlahan saling independen:
       $$\operatorname{Var}(S) = \operatorname{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k$$
     - *Q.E.D.* Terbukti bahwa pembagian dengan $\sqrt{d_k}$ menghasilkan variansi $\operatorname{Var}(S / \sqrt{d_k}) = \frac{d_k}{(\sqrt{d_k})^2} = 1.0$.

2. **Perhitungan Parameter Multi-Head Attention**:
   Sebuah model bahasa menggunakan representasi token berdimensi $d_{\text{model}} = 512$ dan $h = 8$ kepala atensi. Hitung dimensi proyeksi per kepala $d_k$ serta total parameter terpelajari (bobot + bias) dari lapisan Multi-Head Attention tersebut!
   - *Jawaban*:
     - Dimensi per kepala: $d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$.
     - Matriks proyeksi:
       - $W_Q, W_K, W_V \in \mathbb{R}^{512 \times 512}$ (masing-masing $512 \times 512 = 262.144$ bobot).
       - Matriks keluaran $W_O \in \mathbb{R}^{512 \times 512}$ ($262.144$ bobot).
       - Total bobot matriks: $4 \times (512 \times 512) = 4 \times 262.144 = 1.048.576$ parameter.
       - Total vektor bias: $4 \times 512 = 2.048$ parameter.
       - Total parameter keseluruhan: $1.048.576 + 2.048 = 1.050.624$ parameter.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Galat NaN pada Self-Attention Kustom**:
   Seorang praktisi mengimplementasikan fungsi atensi sendiri dengan rumus:
   ```python
   def my_attention(Q, K, V):
       scores = Q @ K.T / np.sqrt(Q.shape[-1])
       weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
       return weights @ V
   ```
   Saat model menerima sekuens panjang dengan nilai skor $S_{ij} \approx 800$, fungsi di atas mengembalikan tensor berisi nilai `nan`. Jelaskan akar masalah komputasionalnya dan perbaiki fungsi tersebut!
   - *Jawaban*:
     - *Akar Masalah*: Angka $e^{800}$ melampaui rentang representasi standar *double precision* IEEE 754 (batas maksimum float64 adalah $\approx 1.79 \times 10^{308}$). Akibatnya, `np.exp(800)` menghasilkan `inf`. Operasi pembagian berikutnya `inf / inf` menghasilkan nilai tak-terdefinisi `nan`.
     - *Solusi Perbaikan*: Terapkan *shift-max trick* (stabilitas numerik):
       ```python
       def my_attention_stable(Q, K, V):
           scores = Q @ K.T / np.sqrt(Q.shape[-1])
           scores_max = np.max(scores, axis=-1, keepdims=True)
           exp_scores = np.exp(scores - scores_max)
           weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
           return weights @ V
       ```

2. **Dilema Causal Masking pada Pelatihan Autoregresif**:
   Jelaskan apa yang akan terjadi jika seorang perekayasa lupa menerapkan *Causal Mask* pada lapisan Self-Attention Decoder saat melatih model penerjemahan mesin atau model bahasa seperti GPT!
   - *Jawaban*:
     - Tanpa Causal Mask, lapisan Self-Attention Decoder menjadi dua arah (*bidirectional*). Setiap token pada langkah waktu $t$ memiliki akses langsung untuk memperhatikan token masa depan $t+1, t+2, \dots$.
     - Model akan mengalami *label leakage* sepele: ia hanya belajar menyalin token target yang berada di depannya tanpa mempelajari relasi semantik dan pemodelan probabilitas transisi autoregresif $P(w_t \mid w_1, \dots, w_{t-1})$.
     - Selama evaluasi dan inferensi waktu-nyata (*test time* / *generation*), token masa depan belum ada, sehingga performa model akan runtuh drastis (hancur total) dan menghasilkan luaran yang berulang atau tidak masuk akal.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Analisis Kompleksitas Komputasi & Memori ($O(T^2)$ vs $O(T)$)**:
   Bandingkan kompleksitas waktu komputasi per lapisan dan kompleksitas jejak memori antara Recurrent Neural Network (LSTM) dan Self-Attention (Transformer) terhadap panjang sekuens $T$ dan dimensi representasi $d$. Jelaskan implikasi arsitekturalnya pada pemrosesan dokumen panjang (misal $T = 32.000$ token) dan solusi kontemporer yang diusulkan oleh komunitas riset!
   - *Jawaban*:
     - *Perbandingan Kompleksitas*:
       - **Kompleksitas Komputasi per Layer**:
         - LSTM: $O(T \cdot d^2)$ — didominasi oleh perkalian bobot gerbang berukuran $4(d \times d)$ yang dieksekusi sebanyak $T$ langkah waktu.
         - Self-Attention: $O(T^2 \cdot d + T \cdot d^2)$ — matriks skor $Q K^T$ berdimensi $(T \times T)$ membutuhkan $O(T^2 \cdot d)$ operasi perkalian titik, sedangkan proyeksi linier membutuhkan $O(T \cdot d^2)$.
       - **Operasi Minimum Sekuensial**:
         - LSTM: $O(T)$ langkah sekuensial tak-terelakkan (karena $\mathbf{h}_t$ bergantung mutlak pada $\mathbf{h}_{t-1}$).
         - Transformer: $O(1)$ langkah sekuensial (seluruh matriks $Q, K, V$ dan atensi dikomputasi simultan secara paralel di GPU/TPU).
       - **Jarak Jalur Maksimum Ketergantungan**:
         - LSTM: $O(T)$ (informasi terhambat dan tereduksi saat merambat melalui $T$ langkah waktu).
         - Transformer: $O(1)$ (interaksi langsung antar sembarang pasangan token melalui satu operasi dot product).
     - *Implikasi pada Sekuens Panjang ($T = 32.000$)*:
       - Menyimpan matriks bobot atensi $(T \times T) = 32.000 \times 32.000 = 1.024.000.000$ elemen float32 memerlukan $\approx 4.096$ MB ($\approx 4$ GB) memori **hanya untuk satu kepala dan satu lapisan**. Untuk model 32 kepala dan 32 lapisan, kebutuhan memori melonjak hingga ratusan gigabyte, menyebabkan kehabisan memori (*Out Of Memory* / OOM).
     - *Solusi Kontemporer*:
       - **FlashAttention** (Dao et al.): Mengoptimalkan komputasi softmax menggunakan teknik *tiling* berbasis SRAM pada GPU tanpa perlu menulis matriks $(T \times T)$ secara eksplisit ke HBM, menghemat memori dari $O(T^2)$ menjadi $O(T)$.
       - **Linear Attention & State Space Models (SSM)**: Menghilangkan perkalian $Q K^T$ eksplisit melalui aproksimasi kernel atau arsitektur modern seperti Mamba/S4 yang memulihkan kompleksitas waktu dan memori linier $O(T)$.

---

## 4. Mini Project Chapter 28

### Judul Tugas Proyek:
**"Implementing Attention and a Small Transformer from Scratch"**

### Deskripsi Tugas:
Rancang, bangun, verifikasi, dan analisis modul mekanisme atensi dan arsitektur mini Transformer lengkap dari dasar (*from scratch*) menggunakan **Python**, **NumPy**, dan **PyTorch** pada CPU. Anda diminta untuk:
1. Mengimplementasikan modul *Scaled Dot-Product Attention* mandiri berbasis NumPy yang dilengkapi dengan stabilisasi numerik *shift-max trick* dan dukungan masker komposit (*padding* + *causal*).
2. Membangun modul *Multi-Head Attention* (MHA) dari nol yang mampu memproyeksikan masukan ke dalam beberapa kepala paralel, mengagregasikan representasi, dan memproyeksikannya kembali melalui matriks keluaran $W_O$.
3. Mengembangkan fungsi pembangkit *Positional Encoding Sinusoidal* (Vaswani et al., 2017) dan membuktikan secara empiris sifat kemiripan kosinus representasi posisi berdekatan vs berjauhan.
4. Merakit blok pembangun terintegrasi: *Transformer Encoder Block* dan *Transformer Decoder Block* yang menggabungkan MHA, Residual Skip Connection, Layer Normalization, dan Position-wise Feed-Forward Network.
5. Membangun arsitektur terpadu *Seq2Seq Mini Transformer* dan melakukan simulasi lintasan maju (*forward pass*) untuk memetakan token masukan ke probabilitas kosakata luaran pada perangkat CPU.
6. Melakukan audit verifikasi numerik ketat (*numerical equivalence audit*): membandingkan keluaran tensor dan peta bobot atensi implementasi NumPy mandiri Anda terhadap `nn.MultiheadAttention` PyTorch deterministik di bawah toleransi ketat deviasi $< 1.0 \times 10^{-6}$.
7. Menganalisis perbedaan performa komputasi, trade-off arsitektur (Encoder-Only vs Decoder-Only vs Encoder-Decoder), dan kelemahan komputasi kuadratik $O(T^2)$.
8. Menyusun dokumentasi akademik yang mencerminkan pertimbangan etika kecerdasan buatan (*Responsible AI*), privasi data pada model bahasa, dan mencantumkan minimal 5 referensi ilmiah primer.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Theoretical Understanding & Scaled Dot-Product Attention** | 20 Poin | Penjelasan matematis peran Query-Key-Value lugas, pembuktian kebutuhan penskalaan variansi $\sqrt{d_k}$ jelas, implementasi *shift-max trick* terbukti anti-overflow, dan penanganan masker padding/kausal tepat. |
| **2. Multi-Head Attention & Positional Encoding Implementation** | 25 Poin | Implementasi MHA NumPy dari dasar bersih dan modular (*clean code*), pemisahan dan penggabungan kepala paralel tepat, implementasi formula sinusoidal PE benar, serta bukti empiris sifat geometris posisi valid. |
| **3. Integrated Transformer Blocks & Seq2Seq Architecture** | 25 Poin | Blok Encoder dan Decoder tersusun rapi dengan koneksi residu, Layer Normalization, dan FFN; arsitektur Seq2Seq Mini Transformer berfungsi sempurna pada forward pass CPU tanpa galat dimensi tensor. |
| **4. Rigorous Numerical Verification & Profiling** | 15 Poin | Pengujian audit numerik komparatif terhadap PyTorch `nn.MultiheadAttention` menghasilkan deviasi absolut $< 1.0 \times 10^{-6}$ secara konsisten, disertai profil latensi komputasi CPU dan pembedahan alokasi parameter analitis. |
| **5. Technical Report, Responsible AI & Academic References** | 15 Poin | Laporan ditulis secara ilmiah dan komprehensif, analisis arsitektur (Encoder vs Decoder vs Seq2Seq) mendalam, pembahasan etika dan batas komputasi terartikulasi baik, serta menyertakan minimal 5 referensi akademik primer terverifikasi. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 13 skrip Python di direktori `content/level-3-deep-learning/chapter-28/code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Peran matriks Query, Key, Value dan pembobotan atensi terverifikasi pada `dot_product_attention.py`.
- [ ] Penskalaan variansi $\sqrt{d_k}$ (stabilisasi dari $\approx 64$ ke $\approx 1.0$) tervalidasi pada `scaled_dot_product_attention.py`.
- [ ] Stabilitas numerik fungsi Softmax (*shift-max trick*) terbukti mencegah overflow `nan` pada `stable_attention_softmax.py`.
- [ ] Transformasi proyeksi linier $W_Q, W_K, W_V, W_O$ pada Self-Attention bekerja mulus pada `self_attention.py`.
- [ ] Interaksi asimetris Decoder-Encoder ($T_{\text{dec}} \neq T_{\text{enc}}$) terverifikasi pada `cross_attention.py`.
- [ ] Padding Mask dan Causal Mask segitiga bawah terbukti menyaring bobot terlarang menjadi $0.0000$ pada `attention_masks.py`.
- [ ] Modul Multi-Head Attention murni NumPy dari dasar menghasilkan representasi multi-kepala paralel pada `multi_head_attention_numpy.py`.
- [ ] Struktur gabungan `in_proj_weight` dan total parameter PyTorch `nn.MultiheadAttention` terbedah pada `multi_head_attention_torch.py`.
- [ ] Pembangkitan Positional Encoding sinusoidal terbukti memberikan kedekatan kosinus pada posisi berdekatan pada `positional_encoding.py`.
- [ ] Blok Transformer Encoder lengkap dengan LayerNorm dan Residual Connection berjalan pada `transformer_encoder.py`.
- [ ] Blok Transformer Decoder lengkap dengan Masked MHA dan Cross-Attention teruji pada `transformer_decoder.py`.
- [ ] Orkestrasi Seq2Seq Mini Transformer terpadu mencatatkan profil latensi inferensi CPU pada `transformer_architecture_demo.py`.
- [ ] Audit verifikasi kesetaraan numerik NumPy vs PyTorch lolos dengan deviasi $< 1.0 \times 10^{-6}$ pada `attention_numerical_verification.py`.
- [ ] Notebook pendamping `chapter_28_companion.ipynb` tervalidasi bersih dalam format v4.
- [ ] Slide Reveal.js `deck.json` memuat 35 slide semantik terstruktur rapi.

