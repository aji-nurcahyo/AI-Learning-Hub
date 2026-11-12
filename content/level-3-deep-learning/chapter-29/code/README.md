# Katalog Kode & Panduan Praktik Laboratorium: Bab 29
## Large Language Models & Generative AI

Direktori ini memuat 14 skrip Python mandiri (*self-contained runnable scripts*) yang mendemonstrasikan seluruh fondasi teoritis dan praktis **Large Language Models (LLM)** dan **Generative AI** pada lingkungan CPU komputer standar:

---

### Daftar Skrip & Fokus Algoritma

| No | Berkas Skrip | Fokus Materi & Algoritma | Pustaka Kunci |
|---|---|---|---|
| 1 | `tokenization_demo.py` | Word vs Char vs Subword BPE, pemetaan ID, token khusus, padding & context window. | `re`, `typing` |
| 2 | `next_token_prediction.py` | Distribusi probabilitas kondisional $P(w_t \mid w_{<t})$, proyeksi logits, Softmax. | `torch`, `torch.nn.functional` |
| 3 | `language_model_loss.py` | Perhitungan Cross-Entropy Loss, Negative Log-Likelihood (NLL), dan Perplexity ($e^{\text{loss}}$). | `math`, `torch` |
| 4 | `autoregressive_generation.py` | Siklus pembangkitan autoregresif rekursif, umpan balik konteks, kondisi henti `<eos>`. | `torch`, `typing` |
| 5 | `greedy_decoding.py` | Strategi decoding greedy ($\operatorname{argmax}$), sifat deterministik, dan jebakan pengulangan (*loop*). | `torch` |
| 6 | `temperature_sampling.py` | Penskalaan logits dengan suhu ($z/T$), analisis entropi Shannon, perataan probabilitas. | `torch` |
| 7 | `top_k_sampling.py` | Penyaringan $k$ token teratas, pemotongan ekor distribusi panjang, renormalisasi Softmax. | `torch` |
| 8 | `top_p_sampling.py` | Nucleus sampling berbasis ambang kumulatif $p$, ukuran kandidat pool adaptif dinamis. | `torch` |
| 9 | `lora_matrix_demo.py` | Low-Rank Adaptation (LoRA): pembekuan $W_0$, adaptor $B \cdot A$, penghematan >90% parameter. | `torch.nn` |
| 10 | `prompting_examples.py` | In-Context Learning: Pola Zero-Shot, Few-Shot, ekstraksi entitas, dan skema JSON. | `json`, `typing` |
| 11 | `toy_retrieval_grounding.py` | RAG concept: temu-kembali TF-IDF, grounded prompt, perbandingan fakta vs halusinasi. | `sklearn.feature_extraction` |
| 12 | `hallucination_evaluation.py` | Kerangka kerja audit halusinasi ekstrinsik/intrinsik, dekomposisi klaim, groundedness score. | `typing` |
| 13 | `output_validation.py` | Arsitektur Guardrails: pencegahan prompt injection, redaksi PII (NIK, Email), validasi JSON. | `re`, `json` |
| 14 | `tiny_transformer_generation.py` | Arsitektur lengkap Decoder-Only Transformer mini, pelatihan toy corpus, generasi autoregresif. | `torch`, `torch.nn` |

---

### Petunjuk Eksekusi

Jalankan seluruh skrip secara mandiri menggunakan interpreter Python di lingkungan virtual repositori:

```bash
# Menjalankan salah satu skrip demonstrasi
python content/level-3-deep-learning/chapter-29/code/tokenization_demo.py
python content/level-3-deep-learning/chapter-29/code/lora_matrix_demo.py
python content/level-3-deep-learning/chapter-29/code/tiny_transformer_generation.py
```

Seluruh skrip dirancang tanpa mewajibkan GPU, tanpa model unduhan raksasa, dan tanpa ketergantungan pada API berbayar.

