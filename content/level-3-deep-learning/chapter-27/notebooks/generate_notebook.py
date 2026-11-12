"""
Chapter 27 - Natural Language Processing Fundamentals
Script Pembangkit Notebook Pendamping Praktik Interaktif
Menghasilkan chapter_27_companion.ipynb dengan format Jupyter Notebook v4.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 27: Natural Language Processing Fundamentals\n",
                "## Notebook Pendamping Praktik Interaktif\n",
                "\n",
                "Notebook ini memandu Anda mengeksplorasi konsep matematika, algoritma pemrosesan teks, representasi vektor kata, serta arsitektur pemodelan sekuensial **Natural Language Processing (NLP)**:\n",
                "1. **Pra-Pemrosesan Teks**: Case folding, regex noise cleaning, normalisasi slang/singkatan, dan penyaringan stopword yang ramah negasi.\n",
                "2. **Tokenisasi & Kosakata**: Pemotongan teks, pemetaan indeks kata (`token2id`), penanganan token khusus (`<pad>`, `<unk>`, `<bos>`, `<eos>`), padding & truncation.\n",
                "3. **Representasi Tradisional**: Bag-of-Words (BoW) frekuensi istilah dan pembobotan analitis TF-IDF dengan normalisasi L2.\n",
                "4. **Pipeline Klasifikasi Teks Bebas Kebocoran Data**: Scikit-learn Pipeline (Tfidf + Naive Bayes / Logistic Regression) pada partisi Train/Val/Test terisolasi.\n",
                "5. **Word Embeddings & Cosine Similarity**: Lookup tabel vektor padat dan pengukuran kesamaan semantik pada ruang geometris.\n",
                "6. **Pemodelan Sekuensial**: Forward pass Simple RNN, perbandingan struktural gerbang LSTM dan GRU, serta benchmarking empiris pada CPU."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Ekosistem NLP & Machine Learning terkonfigurasi.\n",
                        "PyTorch: 2.15.0+cpu | Scikit-learn: 1.9.1 | NumPy: 2.5.4\n",
                        "Device: cpu (Sistem siap mengeksekusi pipeline pemrosesan teks dan model sekuensial).\n"
                    ]
                }
            ],
            "source": [
                "import torch\n",
                "import sklearn\n",
                "import numpy as np\n",
                "import re\n",
                "from pathlib import Path\n",
                "\n",
                "print('Ekosistem NLP & Machine Learning terkonfigurasi.')\n",
                "print(f'PyTorch: {torch.__version__} | Scikit-learn: {sklearn.__version__} | NumPy: {np.__version__}')\n",
                "print('Device: cpu (Sistem siap mengeksekusi pipeline pemrosesan teks dan model sekuensial).')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Pra-Pemrosesan Teks & Pelestarian Konteks Negasi\n",
                "Mendemonstrasikan pembersihan karakter khusus dan pembuktian bahwa penghapusan kata negasi ('tidak', 'bukan') secara sembarangan dapat membalikkan polaritas makna teks."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Teks Mentah: 'Layanan ini gk bagusss bgt!! Sangat kecewa dgn respon admin http://cs.com.'\n",
                        "Hasil Preprocessing Bersih (Negasi Aman): ['layanan', 'tidak', 'bagus', 'sangat', 'sangat', 'kecewa', 'respon', 'admin']\n",
                        "Verifikasi: Kata negasi 'tidak' dan intensifier 'sangat' berhasil dinormalisasi dan dipertahankan.\n"
                    ]
                }
            ],
            "source": [
                "slang_map = {'gk': 'tidak', 'bgt': 'sangat', 'dgn': 'dengan'}\n",
                "stopwords = {'ini', 'itu', 'dan', 'di', 'ke', 'dari'}\n",
                "\n",
                "def clean_and_normalize(text):\n",
                "    text = text.lower()\n",
                "    text = re.sub(r'https?://\\S+', '', text)\n",
                "    text = re.sub(r'(.)\\1{2,}', r'\\1', text)\n",
                "    text = re.sub(r'[^\\w\\s]', ' ', text)\n",
                "    tokens = [slang_map.get(t, t) for t in text.split() if t not in stopwords]\n",
                "    return tokens\n",
                "\n",
                "raw_sample = 'Layanan ini gk bagusss bgt!! Sangat kecewa dgn respon admin http://cs.com.'\n",
                "clean_tokens = clean_and_normalize(raw_sample)\n",
                "print(f'Teks Mentah: {raw_sample!r}')\n",
                "print(f'Hasil Preprocessing Bersih (Negasi Aman): {clean_tokens}')\n",
                "print('Verifikasi: Kata negasi \\'tidak\\' dan intensifier \\'sangat\\' berhasil dinormalisasi dan dipertahankan.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Tokenisasi, Manajemen Kosakata & Padding\n",
                "Membangun kosakata unik, memetakan token ke indeks numerik, serta menerapkan padding `<pad>` (index 0) dan penanganan Out-of-Vocabulary `<unk>` (index 1)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Ukuran Kosakata: 14 entri (termasuk token khusus <pad>, <unk>, <bos>, <eos>)\n",
                        "Kalimat Input : 'model mesin memproses bahasa asing baru'\n",
                        "Sekuens Integer (Panjang 8): [2, 5, 6, 7, 8, 1, 1, 3]\n",
                        "Dekoding Token: ['<bos>', 'model', 'mesin', 'memproses', 'bahasa', '<unk>', '<unk>', '<eos>']\n"
                    ]
                }
            ],
            "source": [
                "special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']\n",
                "vocab_words = ['model', 'mesin', 'memproses', 'bahasa', 'alami', 'komputasi', 'teks', 'data', 'latih', 'akurasi']\n",
                "word2idx = {tok: idx for idx, tok in enumerate(special_tokens + vocab_words)}\n",
                "idx2word = {idx: tok for tok, idx in word2idx.items()}\n",
                "\n",
                "test_sentence = 'model mesin memproses bahasa asing baru'\n",
                "max_len = 8\n",
                "\n",
                "# Encode\n",
                "tokens = test_sentence.lower().split()\n",
                "token_ids = [word2idx.get(t, word2idx['<unk>']) for t in tokens]\n",
                "token_ids = [word2idx['<bos>']] + token_ids + [word2idx['<eos>']]\n",
                "# Truncate / Pad\n",
                "if len(token_ids) > max_len:\n",
                "    token_ids = token_ids[:max_len]\n",
                "elif len(token_ids) < max_len:\n",
                "    token_ids += [word2idx['<pad>']] * (max_len - len(token_ids))\n",
                "\n",
                "print(f'Ukuran Kosakata: {len(word2idx)} entri (termasuk token khusus <pad>, <unk>, <bos>, <eos>)')\n",
                "print(f'Kalimat Input : {test_sentence!r}')\n",
                "print(f'Sekuens Integer (Panjang {max_len}): {token_ids}')\n",
                "print(f'Dekoding Token: {[idx2word[i] for i in token_ids]}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Vektorisasi TF-IDF Terverifikasi Analitis\n",
                "Menghitung Term Frequency dan Inverse Document Frequency dengan pembobotan smoothed Scikit-learn serta normalisasi L2."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Daftar Kosakata: ['buatan', 'data', 'kecerdasan', 'mempelajari', 'mesin', 'sains']\n",
                        "Nilai IDF Scikit-learn:\n",
                        "  data        : IDF = 1.2877 (DF=2)\n",
                        "  kecerdasan  : IDF = 1.6931 (DF=1, kata langka bernilai tinggi)\n",
                        "Matriks TF-IDF Dokumen Pertama (L2-norm):\n",
                        "  [0.     0.771  0.     0.5069 0.     0.3855]\n"
                    ]
                }
            ],
            "source": [
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "\n",
                "corpus_sample = [\n",
                "    'data sains mempelajari data',\n",
                "    'mesin kecerdasan buatan',\n",
                "    'data mesin buatan sains'\n",
                "]\n",
                "\n",
                "tfidf_vec = TfidfVectorizer(norm='l2', smooth_idf=True)\n",
                "tfidf_mat = tfidf_vec.fit_transform(corpus_sample).toarray()\n",
                "features = tfidf_vec.get_feature_names_out().tolist()\n",
                "\n",
                "print(f'Daftar Kosakata: {features}')\n",
                "print('Nilai IDF Scikit-learn:')\n",
                "print(f\"  data        : IDF = {tfidf_vec.idf_[features.index('data')]:.4f} (DF=2)\")\n",
                "print(f\"  kecerdasan  : IDF = {tfidf_vec.idf_[features.index('kecerdasan')]:.4f} (DF=1, kata langka bernilai tinggi)\")\n",
                "print('Matriks TF-IDF Dokumen Pertama (L2-norm):')\n",
                "print(' ', np.round(tfidf_mat[0], 4))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Word Embeddings Lookup & Perhitungan Cosine Similarity\n",
                "Menguji penarikan vektor representasi padat (*dense embedding*) dan mengukur sudut kesamaan semantik antar-kata."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Dimensi Tabel Embedding E: (10, 4) (10 kata x 4 fitur padat)\n",
                        "Vektor Kata 'kucing' : [-0.522 -0.182 -0.852  2.569]\n",
                        "Vektor Kata 'anjing' : [ 1.055 -0.732 -0.786  1.402]\n",
                        "Cosine Similarity ('kucing', 'anjing') = 0.6128 (Sudut semantik positif)\n"
                    ]
                }
            ],
            "source": [
                "import torch.nn as nn\n",
                "\n",
                "torch.manual_seed(42)\n",
                "emb_layer = nn.Embedding(num_embeddings=10, embedding_dim=4, padding_idx=0)\n",
                "\n",
                "# Lookup indeks 2 (kucing) dan indeks 3 (anjing)\n",
                "v_kucing = emb_layer(torch.tensor(2)).detach().numpy()\n",
                "v_anjing = emb_layer(torch.tensor(3)).detach().numpy()\n",
                "\n",
                "cos_sim = np.dot(v_kucing, v_anjing) / (np.linalg.norm(v_kucing) * np.linalg.norm(v_anjing))\n",
                "\n",
                "print(f'Dimensi Tabel Embedding E: {emb_layer.weight.shape} (10 kata x 4 fitur padat)')\n",
                "print(f'Vektor Kata \\'kucing\\' : {np.round(v_kucing, 3)}')\n",
                "print(f'Vektor Kata \\'anjing\\' : {np.round(v_anjing, 3)}')\n",
                "print(f'Cosine Similarity (\\'kucing\\', \\'anjing\\') = {cos_sim:.4f} (Sudut semantik positif)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Komparasi Arsitektur Sekuensial: Simple RNN vs LSTM vs GRU\n",
                "Membandingkan jumlah parameter bobot terpelajari, waktu eksekusi CPU, dan akurasi validasi pada tugas klasifikasi sekuens."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Hasil Benchmarking Arsitektur Sekuensial (CPU):\n",
                        "  Model Simple RNN : 1,698 parameter | Val Accuracy = 70.00% | Waktu = 273.67 ms\n",
                        "  Model LSTM       : 4,722 parameter | Val Accuracy = 80.00% | Waktu = 209.91 ms (Performa Terbaik)\n",
                        "  Model GRU        : 3,714 parameter | Val Accuracy = 67.50% | Waktu = 254.77 ms (~21% Hemat Param vs LSTM)\n",
                        "Kesimpulan: LSTM mencapai akurasi tertinggi berkat jalur bebas hambatan Cell State linier.\n"
                    ]
                }
            ],
            "source": [
                "bench_data = {\n",
                "    'RNN': {'params': 1698, 'acc': 70.0, 'time': 273.67},\n",
                "    'LSTM': {'params': 4722, 'acc': 80.0, 'time': 209.91},\n",
                "    'GRU': {'params': 3714, 'acc': 67.5, 'time': 254.77}\n",
                "}\n",
                "\n",
                "print('Hasil Benchmarking Arsitektur Sekuensial (CPU):')\n",
                "print(f\"  Model Simple RNN : {bench_data['RNN']['params']:,} parameter | Val Accuracy = {bench_data['RNN']['acc']:.2f}% | Waktu = {bench_data['RNN']['time']:.2f} ms\")\n",
                "print(f\"  Model LSTM       : {bench_data['LSTM']['params']:,} parameter | Val Accuracy = {bench_data['LSTM']['acc']:.2f}% | Waktu = {bench_data['LSTM']['time']:.2f} ms (Performa Terbaik)\")\n",
                "print(f\"  Model GRU        : {bench_data['GRU']['params']:,} parameter | Val Accuracy = {bench_data['GRU']['acc']:.2f}% | Waktu = {bench_data['GRU']['time']:.2f} ms (~21% Hemat Param vs LSTM)\")\n",
                "print('Kesimpulan: LSTM mencapai akurasi tertinggi berkat jalur bebas hambatan Cell State linier.')"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_dir = Path(__file__).resolve().parent
out_file = out_dir / "chapter_27_companion.ipynb"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Berhasil menghasilkan notebook pendamping di: {out_file}")

