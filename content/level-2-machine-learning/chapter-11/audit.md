# Audit Publikasi Konten — Chapter 11: Introduction to Machine Learning

Dokumen ini mencatat audit editorial, validasi struktural, integrasi platform, dan verifikasi eksekusi kode Python untuk **Chapter 11: Introduction to Machine Learning**. Bab ini merupakan gerbang pembuka dari **Level 2 — Machine Learning** pada kurikulum Artificial Intelligence Learning Hub.

---

## 1. Identitas & Metadata Chapter

- **ID Chapter**: 11
- **Slug**: `chapter-11`
- **Tingkat Kurikulum**: Level 2 — Machine Learning (Bab Pembuka Level 2)
- **Judul Resmi**: `Introduction to Machine Learning`
- **Status Publikasi**: `published`
- **Prerequisites**: `[10]` (AI Foundations Capstone)
- **Tinjauan Kualitas**: `reviewed: true`, `executionVerified: true`
- **Target Pembelajaran**: 8 measurable objectives berbasis Taksonomi Bloom.
- **Daftar Pustaka Primer**: 5 referensi akademik terverifikasi (Russell & Norvig/Pearson, Mitchell/McGraw-Hill, Hastie et al./Springer, Goodfellow et al./MIT Press, ACM/IEEE-CS CS2023).

---

## 2. Kelengkapan Deliverables Konten

| Komponen | Jalur Berkas | Keterangan & Verifikasi |
| :--- | :--- | :--- |
| **Modul Utama** | `content/level-2-machine-learning/chapter-11/index.md` | Single Source of Truth: 28 sub-topik materi, 12 diagram alur konseptual, pembedaan formal Mitchell vs Samuel, 6 latihan berjenjang, Mini Project Identifikasi Masalah & Studi Baseline, rubrik 100 poin, dan 5 referensi primer. |
| **Slide Presentasi** | `content/level-2-machine-learning/chapter-11/slides/deck.json` | 28 slide Reveal.js terstruktur mengikuti silabus resmi, tata letak kartu dua kolom untuk split slide, blok kode ringkas $\le 18$ baris, panjang body $\le 1000$ karakter, dan speaker notes mendalam untuk dosen. |
| **Panduan Praktikum** | `content/level-2-machine-learning/chapter-11/practice.md` | Panduan praktikum laboratorium langkah demi langkah untuk 7 skrip Python mandiri, studi komparasi rule-based vs ML, eksplorasi overfitting, dan evaluasi baseline. |
| **Companion Notebook** | `content/level-2-machine-learning/chapter-11/notebooks/chapter_11_companion.ipynb` | Notebook interaktif Jupyter format v4 (11 sel) dengan luaran aktual yang tereksekusi tanpa galat. |
| **Katalog Kode Sumber** | `content/level-2-machine-learning/chapter-11/code/README.md` | Dokumentasi 7 skrip Python mandiri, petunjuk eksekusi, dan tabel hasil verifikasi numerik aktual. |

---

## 3. Hasil Verifikasi Eksekusi Skrip Python (3.12.10)

| File Skrip | Fokus Konseptual | Hasil Eksekusi Aktual | Exit Code |
| :--- | :--- | :--- | :---: |
| `rule_based_vs_learning.py` | Rule-Based vs Data-Driven | Rule-based gagal pada variasi sinonim (50.0%); Naive Bayes memetakan bobot statistik dari data latih. | 0 |
| `features_and_labels.py` | Matriks X dan Vektor Target y | Matriks $X$ 2D berdimensi $(10, 3)$, target regresi kontinu $y_{\text{reg}}$ $(10,)$, target klasifikasi biner $y_{\text{clf}}$ $(10,)$. | 0 |
| `supervised_learning_demo.py` | Regresi & Klasifikasi | Regresi daya server MSE: 0.32 ($R^2=0.9999$); Klasifikasi anomali suhu akurasi latih 100.0%. | 0 |
| `unsupervised_learning_demo.py` | K-Means Clustering | K-Means ($K=3$) mengelompokkan 12 profil energi gedung kampus tanpa label dengan inersia 889.50. | 0 |
| `training_testing_demo.py` | Generalisasi vs Hafalan | Model penghafal ($k=1$) akurasi latih 100% namun akurasi uji anjlok ke 90.0%; $k=5$ menggeneralisasi stabil 100.0% di data uji. | 0 |
| `overfitting_underfitting_demo.py` | Bias–Variance Trade-off | Derajat 1: Train MSE 0.1869 / Test MSE 0.2172 (Underfit); Derajat 3: 0.0218 / 0.0424 (Optimal); Derajat 15: 0.0144 / 0.0475 (Overfit). | 0 |
| `baseline_comparison.py` | Baseline Model pada Imbalance | DummyClassifier akurasi mentah 95.00% (Balanced Acc 50.00% - gagal mendeteksi fraud); LogisticRegression Balanced Acc 98.42%. | 0 |

