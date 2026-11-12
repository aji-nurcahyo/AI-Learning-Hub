# Katalog Kode Praktik Chapter 27: Natural Language Processing Fundamentals

Direktori ini berisi seluruh skrip Python mandiri (*runnable scripts*) untuk **Chapter 27: Natural Language Processing Fundamentals**, dirancang untuk dieksekusi secara instan dan deterministik pada CPU tanpa ketergantungan perangkat keras GPU atau unduhan model eksternal yang masif.

---

## 1. Daftar Skrip Praktik

| No | Berkas Skrip | Topik Pembelajaran Utama | Library Utama | Output / Perilaku yang Diharapkan |
|---|---|---|---|---|
| 1 | `text_preprocessing.py` | Text normalization, case folding, slang normalization, negation-aware stopword removal | Python stdlib, `re`, `string` | Teks bersih dengan pelestarian kata negasi kritis ('tidak', 'bukan') |
| 2 | `tokenization.py` | Regex tokenization, vocabulary mapping, token `<pad>`, `<unk>`, `<bos>`, `<eos>`, padding & truncation | Python stdlib, `re` | Vektor sekuens integer terstandar panjang tepat `max_len` |
| 3 | `bag_of_words.py` | Count vectorization dari dasar (Scratch), term frequency matrix, sparsity analysis, order invariance limitation | `numpy`, `scikit-learn` | Matriks frekuensi dokumen $\times$ kata, persentase sparsity, demonstrasi hilangnya urutan kata |
| 4 | `tfidf_representation.py` | Step-by-step TF-IDF calculation, formula smoothed Scikit-learn, normalisasi L2 | `numpy`, `scikit-learn` | Pembuktian kesetaraan numerik identik (< 1e-6) terhadap `TfidfVectorizer` |
| 5 | `ngram_features.py` | Ekstraksi sliding window Unigram, Bigram, Trigram, penangkapan konteks negasi lokal, analisis ledakan kosakata | `scikit-learn` | Fitur 'tidak ramah' sebagai prediktor mandiri, perbandingan ukuran kosakata |
| 6 | `text_classification.py` | End-to-end text classification pipeline, isolasi data split 3-way, Scikit-learn Pipeline (Tfidf + NB/LR) | `scikit-learn` | Evaluasi akurasi, Macro F1, matriks konfusi, dan analisis galat |
| 7 | `word_embeddings.py` | Sparse one-hot vs dense distributed embeddings, tabel bobot $E \in \mathbb{R}^{V \times D}$, embedding lookup NumPy vs PyTorch | `numpy`, `torch` | Vektor padat per kata, verifikasi deviasi numerik 0.00e+00 terhadap `nn.Embedding` |
| 8 | `cosine_similarity.py` | Formulasi geometris Cosine Similarity $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$, nearest neighbors query, limitasi semantik | `numpy` | Pembuktian sudut sejajar (+1.0), tegak lurus (0.0), berlawanan (-1.0), dan kueri tetangga terdekat |
| 9 | `simple_rnn.py` | Forward pass Simple RNN (Elman) rekuren $h_t = \tanh(x_t W_{xh}^T + h_{t-1} W_{hh}^T + b)$ dari dasar NumPy | `numpy`, `torch` | Verifikasi kesetaraan output dan final hidden state terhadap `torch.nn.RNN` (< 1e-6) |
| 10 | `lstm_sequence_model.py` | Long Short-Term Memory (LSTM), Cell State $c_t$, Hidden State $h_t$, Forget/Input/Output gates, perhitungan parameter | `torch` | Perhitungan analitis parameter ($4 \times (D \cdot H + H^2 + 2H)$) cocok dengan PyTorch `nn.LSTM` |
| 11 | `gru_sequence_model.py` | Gated Recurrent Unit (GRU), Update gate $z_t$, Reset gate $r_t$, efisiensi parameter ~25% vs LSTM | `torch` | Pembuktian rasio parameter 75.0% terhadap LSTM pada dimensi terkalibrasi |
| 12 | `rnn_lstm_gru_comparison.py` | Benchmarking komparatif Simple RNN vs LSTM vs GRU pada tugas klasifikasi teks sekuensial sintetis | `torch`, `numpy` | Laporan komparasi: parameter, runtime, loss, val accuracy, dan pemantauan norma gradien |

---

## 2. Cara Menjalankan Skrip

Seluruh skrip dapat dijalankan langsung melalui terminal menggunakan lingkungan virtual Python:

```bash
# Menjalankan pembersihan dan normalisasi teks
python content/level-3-deep-learning/chapter-27/code/text_preprocessing.py

# Menjalankan tokenisasi dan kosakata
python content/level-3-deep-learning/chapter-27/code/tokenization.py

# Menjalankan verifikasi analitis TF-IDF
python content/level-3-deep-learning/chapter-27/code/tfidf_representation.py

# Menjalankan benchmarking komparasi sekuensial RNN vs LSTM vs GRU
python content/level-3-deep-learning/chapter-27/code/rnn_lstm_gru_comparison.py
```

Log hasil eksperimen tersimpan secara otomatis dalam format JSON di:
`content/level-3-deep-learning/chapter-27/experiments/results_ch27.json`.

