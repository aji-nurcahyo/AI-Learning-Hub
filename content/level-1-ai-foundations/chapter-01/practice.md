# Praktik & Laboratorium: Chapter 01 — Introduction to Artificial Intelligence

Halaman ini menyediakan seluruh materi praktik langsung, contoh kode Python runnable, interaktif Jupyter Notebook, petunjuk eksekusi, serta verifikasi output aktual untuk Chapter 01.

---

## 1. Tujuan Praktik

Setelah menyelesaikan sesi laboratorium ini, Anda diharapkan mampu:
1. Menjalankan dan menganalisis sistem rekomendasi berbasis aturan deterministik (*Rule-Based System*).
2. Membuktikan secara eksperimental terjadinya *False Positive* dan *False Negative* pada klasifikasi berbasis kata kunci (*keyword matching*).
3. Menguji kerapuhan pencocokan eksak (*exact matching*) saat berhadapan dengan variasi bahasa alami manusia.
4. Menghitung metrik akurasi secara eksplisit menggunakan logika dasar Python tanpa library *black-box*.
5. Menghitung dan menginterpretasikan matriks kontingensi biner (*confusion counts*: TP, FP, TN, FN) pada skenario data tidak seimbang (*class imbalance*).

---

## 2. Persyaratan Lingkungan (Environment)

- **Python Runtime**: Python 3.11 atau Python 3.12 (teruji pada Python 3.12.10).
- **Pustaka Pihak Ketiga**: Tidak ada! Seluruh contoh skrip menggunakan *Python Standard Library* murni (`sys`, `typing`, `json`, `io`).
- **Jupyter Environment**: VS Code Jupyter Extension atau Google Colab untuk membuka file `.ipynb`.
- **Lokasi Kerja**: Buka direktori proyek menggunakan menu *Open Folder* di VS Code (lihat panduan [Getting Started GS-03](#getting-started/gs-03)).

---

## 3. Langkah Menjalankan Praktik

1. Buka terminal pada direktori `content/level-1-ai-foundations/chapter-01/code/`.
2. Aktifkan virtual environment Anda (lihat panduan [Getting Started GS-04](#getting-started/gs-04)).
3. Eksekusi skrip Python satu per satu menggunakan perintah:
   ```bash
   python 01_rule_based_recommender.py
   python 02_keyword_text_classifier.py
   python 03_exact_matching_vs_variation.py
   python 04_explicit_accuracy_calc.py
   python 05_simple_confusion_analysis.py
   ```
4. Untuk eksplorasi interaktif berbasis sel, buka berkas notebook:
   `content/level-1-ai-foundations/chapter-01/notebooks/chapter_01_companion.ipynb`
   Lalu jalankan perintah **Run All** pada kernel bersih.

---

## 4. Daftar Contoh Kode Python Runnable

Setiap contoh kode di bawah ini dapat disalin langsung atau dijalankan dari folder `code/`:

### Contoh 1: Sistem Rekomendasi Berbasis Aturan (`01_rule_based_recommender.py`)
Mendemonstrasikan logika penentuan keputusan deterministik menggunakan aturan `if-elif-else`.

```python
from typing import Dict, Any, List

def recommend_study_path(profile: Dict[str, Any]) -> List[str]:
    interest = profile.get("interest", "").lower()
    exp = profile.get("experience_level", "").lower()
    recs = []

    if exp in ["pemula", "beginner"]:
        recs.append("Fondasi Python & Struktur Data Dasar")

    if interest == "computer vision":
        recs.append("CNN & Pengolahan Citra" if exp == "menengah" else "Dasar Matriks Citra")
    elif interest == "natural language processing":
        recs.append("LLM & Fine-Tuning" if exp == "menengah" else "Pengantar Tokenisasi Teks")
    else:
        recs.append("Dasar Algoritma AI & Representasi Pengetahuan")

    recs.append("Prinsip Responsible AI & Mitigasi Bias")
    return recs

# Uji profil mahasiswa
mhs = {"user": "Budi", "interest": "computer vision", "experience_level": "pemula"}
print(f"Rekomendasi untuk {mhs['user']}:", recommend_study_path(mhs))
```

*Output Aktual Terverifikasi:*
```text
Rekomendasi untuk Budi: ['Fondasi Python & Struktur Data Dasar', 'Dasar Matriks Citra', 'Prinsip Responsible AI & Mitigasi Bias']
```

---

### Contoh 2: Klasifikasi Teks Kata Kunci & Analisis Kasus Salah (`02_keyword_text_classifier.py`)
Mendemonstrasikan terjadinya False Positive (pesan akademik normal dianggap spam) dan False Negative (pesan penipuan lolos).

```python
SPAM_KEYWORDS = {"hadiah", "gratis", "klaim", "menang", "promo", "transfer"}

def classify_message_by_keyword(text: str) -> str:
    text_lower = text.lower()
    for kw in SPAM_KEYWORDS:
        if kw in text_lower:
            return "SPAM"
    return "HAM"

# Contoh kasus uji:
# Kasus A: "Buku referensi kuliah ini gratis dipinjam dari perpustakaan." (HAM) -> Terprediksi SPAM (False Positive!)
# Kasus B: "Tolong kirimkan sejumlah dana darurat ke rekening ini." (SPAM) -> Terprediksi HAM (False Negative!)
```

*Wawasan Pembelajaran:* Kata kunci mengabaikan semantik kalimat. Konteks "gratis di perpustakaan" disalahartikan sebagai spam karena ketiadaan pemodelan semantik.

---

### Contoh 3: Exact Matching vs Variasi Input Alami (`03_exact_matching_vs_variation.py`)
Membuktikan kerapuhan aturan string matching ketika menghadapi salah ketik (*typo*), huruf kapital, tanda baca, dan variasi frasa sinonim.

| Kueri Masukan | Hasil Pencocokan | Status | Keterangan |
|---|---|---|---|
| `kapan jadwal uas` | Ditemukan | **SUKSES** | Persis sesuai entri database |
| `Kapan jadwal UAS?` | Tidak Diketahui | **GAGAL** | Variasi kapitalisasi & tanda tanya |
| `kapan jadual uas` | Tidak Diketahui | **GAGAL** | Variasi salah ketik (*typo*) |
| `kapan ujian akhir semester dimulai` | Tidak Diketahui | **GAGAL** | Variasi sinonim formal |

*Wawasan Pembelajaran:* Exact matching gagal pada 3 dari 4 variasi kalimat umum. Inilah alasan mengapa kita membutuhkan representasi vektor semantik (*embeddings*).

---

### Contoh 4: Kalkulasi Eksplisit Akurasi (`04_explicit_accuracy_calc.py`)
Menghitung rumus akurasi secara transparan dari daftar label aktual dan prediksi:

$$\text{Akurasi} = \frac{\sum [y_{\text{true}} == y_{\text{pred}}]}{N}$$

```python
y_ground_truth = [1, 0, 0, 1, 0, 0, 1, 1, 0, 1]  # 1 = Spam, 0 = Ham
y_model_pred   = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]

correct = sum(1 for yt, yp in zip(y_ground_truth, y_model_pred) if yt == yp)
total = len(y_ground_truth)
acc = correct / total
print(f"Prediksi benar: {correct}/{total} -> Akurasi: {acc * 100:.1f}%")
```

*Output Aktual Terverifikasi:*
```text
Prediksi benar: 8/10 -> Akurasi: 80.0%
```

---

### Contoh 5: Analisis Matriks Kontingensi / Confusion Counts (`05_simple_confusion_analysis.py`)
Membongkar angka akurasi ke dalam 4 kuadran pada skenario deteksi penipuan (*fraud*):

| Kelas | Prediksi Positif (Fraud) | Prediksi Negatif (Normal) |
|---|---|---|
| **Aktual Positif (Fraud)** | $\text{TP} = 2$ | $\text{FN} = 2$ |
| **Aktual Negatif (Normal)** | $\text{FP} = 2$ | $\text{TN} = 14$ |

*Output Analisis:*
- Akurasi keseluruhan: **80.0%**
- False Negative: **2 dari 4 transaksi fraud nyata lolos tak terdeteksi (50% kegagalan tangkap!)**
- Pelajaran penting: Dalam aplikasi keamanan atau medis, False Negative membawa dampak kerugian yang jauh lebih fatal daripada sekadar angka persentase akurasi global.

---

## 5. Interactive Companion Notebook

File notebook pendamping tersimpan di:
[`notebooks/chapter_01_companion.ipynb`](file:///d:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-1-ai-foundations/chapter-01/notebooks/chapter_01_companion.ipynb)

Karakteristik Notebook:
- **Format**: nbformat 4 (Python 3 kernel).
- **Struktur**: Menyusun kelima eksperimen di atas secara sekuensial disertai sel Markdown penjelasan dan pertanyaan eksplorasi interaktif.
- **Status Eksekusi**: Seluruh sel telah dieksekusi secara bersih dan menyimpan output stream riil.

---

## 6. Tata Kelola Dataset & Lisensi

- **Status Dataset**: Seluruh data yang digunakan dalam latihan Chapter 01 ini adalah **data sintetis ilustratif** yang didefinisikan langsung di dalam kode untuk tujuan pengajaran.
- **Privasi & Keamanan**: Tidak ada data pribadi, kredensial, atau kunci rahasia (*API keys*) yang disimpan.
- **Lisensi**: Kode dan materi latihan dirilis terbuka di bawah lisensi repositori untuk tujuan edukasi nirlaba.

---

## 7. Keterbatasan Praktik

- Model klasifikasi pada chapter ini belum menggunakan optimasi gradien atau bobot tensor yang dilatih dari jutaan sampel; fokus utama adalah pemahaman logika evaluasi dan penalaran sistem cerdas.
- Pelatihan model Machine Learning skala riil dengan Scikit-learn akan dimulai secara sistematis pada Level 2.

