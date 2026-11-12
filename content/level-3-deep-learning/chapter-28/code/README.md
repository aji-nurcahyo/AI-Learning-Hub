# Katalog Kode Praktik Chapter 28: Attention & Transformer Architecture

Direktori ini berisi seluruh 13 skrip Python mandiri (*runnable scripts*) untuk **Chapter 28: Attention & Transformer Architecture**, dirancang untuk dieksekusi secara instan dan deterministik pada CPU tanpa ketergantungan perangkat keras GPU atau unduhan bobot model LLM eksternal.

---

## 1. Daftar Skrip Praktik

| No | Berkas Skrip | Topik Pembelajaran Utama | Library Utama | Output / Perilaku yang Diharapkan |
|---|---|---|---|---|
| 1 | `dot_product_attention.py` | Konsep peran Query (Q), Key (K), dan Value (V), matriks skor mentah $S = Q K^T$ | `numpy` | Matriks skor dot-product, bobot atensi softmax, vektor representasi konteks |
| 2 | `scaled_dot_product_attention.py` | Formula $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, normalisasi variansi $O(1)$ | `numpy` | Bukti empiris variansi skor dinormalkan dari $d_k=64$ ke $1.01$, worked example |
| 3 | `stable_attention_softmax.py` | Trik pergeseran maksimum $\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$ pencegah overflow | `numpy` | Demonstrasi naif vs stabil: pencegahan `NaN`/`Inf` pada logit besar |
| 4 | `self_attention.py` | Modul Self-Attention dengan proyeksi linier $W_Q, W_K, W_V, W_O$, relasi interaksi antar-token | `numpy` | Matriks atensi interaksi token-ke-token, preservasi dimensi input $d_{\text{model}}$ |
| 5 | `cross_attention.py` | Cross-Attention Decoder-ke-Encoder, penanganan panjang sekuens asimetris $T_{\text{dec}} \ne T_{\text{enc}}$ | `numpy` | Distribusi bobot atensi per token target terhadap seluruh token sumber |
| 6 | `attention_masks.py` | Padding Mask (penghilangan token `<pad>`) dan Causal Mask segitiga bawah (autoregresif) | `numpy`, `torch` | Pembuktian matematis posisi masa depan / padding menerima bobot atensi tepat $0.0000$ |
| 7 | `multi_head_attention_numpy.py` | Implementasi Multi-Head Attention lengkap dari dasar (scratch NumPy), pemecahan dan penggabungan head | `numpy` | Atensi paralel 4D tensor, pembuktian spesialisasi pola representasi antar-head |
| 8 | `multi_head_attention_torch.py` | Pembedahan mendalam PyTorch `nn.MultiheadAttention`, struktur `in_proj_weight`, key_padding_mask, causal mask | `torch` | Verifikasi parameter $4 d^2 + 4 d$, forward pass dengan masked atensi |
| 9 | `positional_encoding.py` | Sinusoidal Positional Encoding (Vaswani et al. 2017), bukti invarian permutasi Self-Attention | `numpy`, `torch` | Matriks PE, properti sinus/kosinus pada pos 0, penggabungan vektor posisi |
| 10 | `transformer_encoder.py` | Blok utuh Transformer Encoder (MHA + Add & Norm + Feed-Forward Network + Add & Norm) | `torch`, `numpy` | Verifikasi parameter analitis (3.280 param), normalisasi layer (mean ~0, std ~1) |
| 11 | `transformer_decoder.py` | Blok utuh Transformer Decoder (Masked MHA + Add & Norm + Cross-Attention + Add & Norm + FFN + Add & Norm) | `torch`, `numpy` | Verifikasi parameter analitis (4.400 param), forward pass dengan causal masking |
| 12 | `transformer_architecture_demo.py` | Mini Transformer terpadu (Encoder-Decoder Seq2Seq) lengkap dari embedding hingga logits | `torch` | Forward pass inferensi CPU terukur (~1.34 ms/batch), bentuk tensor logits target |
| 13 | `attention_numerical_verification.py` | Pengujian kesetaraan numerik ketat NumPy scratch vs PyTorch MHA (unmasked & causal mask) | `numpy`, `torch` | Deviasi numerik maksimum $< 1 \times 10^{-7}$ (terbukti MATCH 100% identik), log JSON |

---

## 2. Cara Menjalankan Skrip

Seluruh skrip dapat dijalankan langsung melalui terminal menggunakan lingkungan virtual Python:

```bash
# Menjalankan verifikasi formula Scaled Dot-Product Attention
python content/level-3-deep-learning/chapter-28/code/scaled_dot_product_attention.py

# Menjalankan demonstrasi Multi-Head Attention dari dasar NumPy
python content/level-3-deep-learning/chapter-28/code/multi_head_attention_numpy.py

# Menjalankan demonstrasi arsitektur Mini Transformer lengkap
python content/level-3-deep-learning/chapter-28/code/transformer_architecture_demo.py

# Menjalankan audit kesetaraan numerik ketat terhadap PyTorch
python content/level-3-deep-learning/chapter-28/code/attention_numerical_verification.py
```

Log hasil audit verifikasi numerik tersimpan secara otomatis dalam format JSON di:
`content/level-3-deep-learning/chapter-28/experiments/results_ch28.json`.

