# Panduan Praktik: Knowledge Representation & Reasoning

> **Tingkat Keahlian**: Level 1 — AI Foundations  
> **Chapter**: 05 — Knowledge Representation & Reasoning  
> **Jalur Spesialisasi**: Core Curriculum  
> **Lingkungan**: Python >= 3.12 (Standard Library, tanpa instalasi pustaka luar)  

---

## 1. Tujuan Praktik

Setelah menyelesaikan sesi laboratorium mandiri ini, pembelajar mampu:
1. Memodelkan basis pengetahuan (*Knowledge Base*) menggunakan fakta atomik (*facts*) dan kaidah implikasi (*rules*).
2. Membangun tabel kebenaran (*truth tables*) dan memvalidasi keabsahan *logical entailment* ($KB \models \alpha$) menggunakan metode *Model Checking*.
3. Mengimplementasikan aturan inferensi deduktif klasik *Modus Ponens* dan *Generalized Modus Ponens*.
4. Mengembangkan mesin inferensi *Forward Chaining* (*data-driven*) dengan fasilitas penjelasan bukti penalaran (*How Facility*).
5. Mengembangkan mesin inferensi *Backward Chaining* (*goal-driven*) dengan mekanisme pemotongan siklus (*cycle detection*).
6. Menerapkan sistem pakar berbasis aturan (*Rule-Based Expert System*) untuk diagnosis masalah jaringan komputer (*Network Troubleshooting*).

---

## 2. Persyaratan Lingkungan

- **Python Runtime**: Python 3.12+ (terverifikasi kompatibel pada Python 3.12.10).
- **Dependensi Eksternal**: **Tidak ada** (murni memanfaatkan pustaka standar Python: `dataclasses`, `itertools`, `typing`, `sys`).
- **Editor / Lingkungan**: Visual Studio Code, PyCharm, atau Jupyter Notebook / Google Colab.

---

## 3. Langkah Menjalankan Kode Praktik

Buka terminal di root repositori dan pastikan virtual environment Anda aktif:

```bash
# 1. Pindah ke direktori kode praktik Chapter 05
cd content/level-1-ai-foundations/chapter-05/code

# 2. Menjalankan skrip individual
python facts_and_rules.py
python truth_table.py
python modus_ponens.py
python forward_chaining.py
python backward_chaining.py
```

---

## 4. Ringkasan Modul Praktik

| File Script | Peran Utama | Karakteristik Penting |
|---|---|---|
| `facts_and_rules.py` | Fondasi Representasi Pengetahuan | Struktur data `Fact`, `Rule`, dan `KnowledgeBase`. |
| `truth_table.py` | Logika Proposisional & Entailment | Evaluasi $2^n$ model interpretasi dan pembuktian $KB \models \alpha$ tanpa `eval()`. |
| `modus_ponens.py` | Aturan Inferensi Deduktif | Menarik fakta baru dari kaidah yang premisnya terpenuhi. |
| `forward_chaining.py` | Data-Driven Inference Engine | Inferensi maju hingga mencapai titik tetap (*fixpoint*) dengan fasilitas penjelasan. |
| `backward_chaining.py` | Goal-Driven Inference Engine | Pembuktian hipotesis rekursif dengan pencegahan loop tak berhingga (*cycle cut*). |

---

## 5. Companion Jupyter Notebook

Tersedia berkas notebook interaktif pada:
`content/level-1-ai-foundations/chapter-05/notebooks/chapter_05_companion.ipynb`

Menyediakan demonstrasi interaktif penalaran logika dan inferensi diagnosis jaringan.

---

## 6. Pertanyaan Eksplorasi Mandiri

1. Pada `truth_table.py`, ujilah formula logika Kontradiksi: $P \land \neg P$. Perhatikan output `is_tautology` dan `is_satisfiable`!
2. Pada `forward_chaining.py`, tambahkan sebuah kaidah baru: `IF (internet_inaccessible) THEN contact_isp_support`. Jalankan ulang inferensi dan amati bagaimana fakta baru tersebut diturunkan pada iterasi berikutnya!
3. Pada `backward_chaining.py`, jelaskan apa konsekuensi filosofis dan teknis dari penerapan *Closed-World Assumption* (CWA) jika fakta status DNS tidak pernah dimasukkan oleh teknisi!

