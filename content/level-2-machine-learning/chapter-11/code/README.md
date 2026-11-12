# Panduan Praktik Kode: Chapter 11 — Introduction to Machine Learning

Direktori ini memuat 7 skrip Python mandiri yang mendemonstrasikan konsep-konsep pokok pengantar Machine Learning menggunakan Scikit-learn, NumPy, dan Pandas.

---

## 1. Inventaris Modul Kode

| Nama File | Fokus Konseptual | Deskripsi Singkat |
| :--- | :--- | :--- |
| `rule_based_vs_learning.py` | Paradigma AI | Komparasi sistem berbasis aturan (*if-else*) vs *Machine Learning* pada klasifikasi spam. |
| `features_and_labels.py` | Struktur Data ML | Pembagian Matriks Fitur ($X \in \mathbb{R}^{N \times D}$) dan Vektor Target ($y \in \mathbb{R}^N$). |
| `supervised_learning_demo.py` | Pembelajaran Terbimbing | Demonstrasi tugas Regresi (output kontinu) dan Klasifikasi (output diskrit). |
| `unsupervised_learning_demo.py` | Pembelajaran Tak Terbimbing | Menemukan struktur klaster tersembunyi pada data profil energi tanpa label bimbingan. |
| `training_testing_demo.py` | Validasi & Generalisasi | Mengapa akurasi data latih 100% pada model penghafal (*memorizer*) tidak menjamin generalisasi data uji. |
| `overfitting_underfitting_demo.py` | Bias–Variance Trade-off | Spektrum *Underfitting* (derajat 1), *Good Fit* (derajat 3), dan *Overfitting* (derajat 15). |
| `baseline_comparison.py` | Evaluasi Objektif | Mengapa baseline model sepele (*DummyClassifier*) mutlak diperlukan untuk mengevaluasi akurasi model. |

---

## 2. Persyaratan & Lingkungan Eksekusi

Pastikan virtual environment telah aktif:
```powershell
.\.venv\Scripts\Activate.ps1
```

Pustaka yang digunakan:
- `python >= 3.10`
- `scikit-learn >= 1.3.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`

---

## 3. Cara Menjalankan Skrip

Setiap skrip dapat dijalankan secara mandiri:
```bash
python content/level-2-machine-learning/chapter-11/code/rule_based_vs_learning.py
python content/level-2-machine-learning/chapter-11/code/features_and_labels.py
python content/level-2-machine-learning/chapter-11/code/supervised_learning_demo.py
python content/level-2-machine-learning/chapter-11/code/unsupervised_learning_demo.py
python content/level-2-machine-learning/chapter-11/code/training_testing_demo.py
python content/level-2-machine-learning/chapter-11/code/overfitting_underfitting_demo.py
python content/level-2-machine-learning/chapter-11/code/baseline_comparison.py
```

---

## 4. Tabel Verifikasi Hasil Numerik Aktual

| Skrip | Metrik / Pengamatan Kunci | Nilai Hasil Eksekusi | Status Validasi |
| :--- | :--- | :--- | :---: |
| `rule_based_vs_learning.py` | Akurasi Rule-Based vs ML | Rule-Based: 50.0% vs ML: 100.0% | **TERVERIFIKASI** |
| `features_and_labels.py` | Dimensi Matriks X & Target y | $X: (10, 3)$, $y_{\text{reg}}: (10,)$, $y_{\text{clf}}: (10,)$ | **TERVERIFIKASI** |
| `supervised_learning_demo.py` | Regresi MSE / Klasifikasi Akurasi | Regresi MSE: 0.17 ($R^2=0.9996$) / Akurasi: 100% | **TERVERIFIKASI** |
| `unsupervised_learning_demo.py` | Inersia Klaster K-Means (K=3) | Inersia: 984.75, 3 klaster terpisah | **TERVERIFIKASI** |
| `training_testing_demo.py` | Generalisasi k-NN ($k=1$ vs $k=5$) | $k=1$: Train 100%, Test 80.0% vs $k=5$: Train 88.6%, Test 86.7% | **TERVERIFIKASI** |
| `overfitting_underfitting_demo.py` | MSE Latih vs Uji (Deg 1, 3, 15) | Deg 1: 0.222 / 0.189; Deg 3: 0.015 / 0.021; Deg 15: 0.006 / 218.4 | **TERVERIFIKASI** |
| `baseline_comparison.py` | Akurasi vs Balanced Accuracy | Dummy: 95.0% (Bal: 50.0%) vs ML: 97.0% (Bal: 94.7%) | **TERVERIFIKASI** |

