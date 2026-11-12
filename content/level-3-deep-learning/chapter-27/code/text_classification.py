"""
Chapter 27 - Natural Language Processing Fundamentals
Script 6: End-to-End Traditional Text Classification Pipeline

Membangun alur kerja klasifikasi teks (analisis sentimen pelanggan) terstruktur:
1. Partisi data 3-way (Train 70%, Val 15%, Test 15%) bebas kebocoran data (zero data leakage).
2. Fitting TfidfVectorizer HANYA pada data latih (training data only).
3. Pembuatan Scikit-learn Pipeline untuk MultinomialNB dan LogisticRegression.
4. Evaluasi komparatif metrik (Akurasi, Macro F1, Matriks Konfusi) terhadap Dummy Baseline.
5. Analisis galat (error analysis) pada sampel prediksi yang meleset.
"""

import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def create_sentiment_dataset() -> Tuple[List[str], List[int], List[str]]:
    """
    Menghasilkan dataset ulasan pelanggan sintetis deterministik.
    Kelas: 0 (Negatif), 1 (Netral), 2 (Positif).
    """
    texts = [
        # Negatif (0)
        "Barang rusak saat tiba, kemasan penyok dan tidak berfungsi sama sekali",
        "Kecewa berat dengan pelayanan toko ini, respon sangat lambat dan kasar",
        "Kualitas produk sangat buruk, bahan tipis mudah sobek, rugi beli",
        "Pengiriman terlambat satu minggu, kurir tidak ramah dan barang cacat",
        "Aplikasi sering crash dan error terus saat mau transaksi pembayaran",
        "Baterai cepat panas dan boros sekali, baru pakai satu jam sudah habis",
        "Tidak sesuai deskripsi foto, warnanya kusam dan ukurannya terlalu kecil",
        "Sangat menyesal belanja di sini, tidak akan pernah pesan lagi",
        "Pelayanan komplain sangat mengecewakan, tidak ada solusi yang diberikan",
        "Harga sangat mahal tapi kualitas sangat rendah, tidak direkomendasikan",
        # Netral (1)
        "Barang sudah sampai sesuai pesanan, standar saja tidak ada yang istimewa",
        "Fungsi produk berjalan normal, pengiriman tidak terlalu cepat atau lambat",
        "Kualitas lumayan untuk harga segini, packing kardus biasa saja",
        "Ukuran baju pas di badan, bahan cukup nyaman dipakai harian",
        "Warna agak sedikit berbeda dari foto namun masih dalam batas wajar",
        "Respon penjual standar, produk berfungsi sebagaimana mestinya",
        "Pengemasan rapi dan aman, estimasi waktu tiba tepat waktu",
        "Barang original buatan pabrik, garansi resmi satu tahun",
        "Kualitas audio biasa saja, bass tidak terlalu menonjol",
        "Aplikasi berfungsi normal setelah diupdate ke versi terbaru",
        # Positif (2)
        "Produk luar biasa bagus, kualitas material premium dan sangat awet",
        "Pelayanan penjual sangat ramah, respon cepat dan sangat membantu",
        "Pengiriman super kilat, pesan pagi sore sudah sampai rumah dengan aman",
        "Sangat puas dengan performa laptop ini, cepat dan lancar untuk coding",
        "Desain elegan dan mewah, sangat direkomendasikan untuk dibeli",
        "Kamera jernih sekali baik siang maupun malam, hasilnya memuaskan",
        "Harga terjangkau dengan fitur lengkap, toko terpercaya bintang lima",
        "Baterai sangat awet tahan dua hari pemakaian penuh, mantap",
        "Packing sangat tebal dengan bubble wrap berlapis, barang mulus sempurna",
        "Pelayanan garansi ramah dan cepat, barang langsung diganti baru",
    ]
    labels = [0] * 10 + [1] * 10 + [2] * 10
    class_names = ["Negatif", "Netral", "Positif"]
    return texts, labels, class_names


def main():
    print("=== Chapter 27: Demo Traditional Text Classification Pipeline ===")

    texts, labels, class_names = create_sentiment_dataset()
    print(f"Total Dataset: {len(texts)} sampel (3 kelas seimbang: 10 Negatif, 10 Netral, 10 Positif)")

    # 1. Partisi Data Terisolasi 3-Way (Train 70%, Val 15%, Test 15%)
    # Menggunakan stratifikasi agar proporsi kelas seimbang di setiap split
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Partisi Data: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

    # 2. Pembuatan Model Pipeline (Tfidf + Estimator)
    # Pipeline menjamin TfidfVectorizer hanya di-fit pada X_train_raw tanpa kebocoran data
    models = {
        "Dummy Baseline": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", DummyClassifier(strategy="most_frequent"))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", MultinomialNB(alpha=1.0))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(C=1.0, max_iter=200, random_state=42))
        ])
    }

    # 3. Pelatihan dan Evaluasi pada Validation Set
    print("\n--- Evaluasi Model pada Validation Set ---")
    val_scores = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train_raw, y_train)
        preds_val = pipeline.predict(X_val_raw)
        acc = accuracy_score(y_val, preds_val)
        f1 = f1_score(y_val, preds_val, average="macro")
        val_scores[name] = (acc, f1)
        print(f"  {name:25s} | Akurasi: {acc*100:5.2f}% | Macro F1: {f1:.4f}")

    # Pilih model terbaik berdasarkan F1-score validasi
    best_model_name = max(val_scores, key=lambda k: val_scores[k][1])
    best_pipeline = models[best_model_name]
    print(f"\nModel Terpilih untuk Evaluasi Uji Akhir: '{best_model_name}'")

    # 4. Pengujian Akhir pada Test Set Terisolasi
    preds_test = best_pipeline.predict(X_test_raw)
    test_acc = accuracy_score(y_test, preds_test)
    test_f1 = f1_score(y_test, preds_test, average="macro")
    cm = confusion_matrix(y_test, preds_test)

    print("\n--- Hasil Uji Akhir (Unseen Test Set) ---")
    print(f"Akurasi Test: {test_acc*100:.2f}%")
    print(f"Macro F1    : {test_f1:.4f}")
    print("\nMatriks Konfusi (Confusion Matrix):")
    print(cm)

    # 5. Analisis Galat (Error Analysis)
    print("\n--- Analisis Galat (Error Analysis pada Test Set) ---")
    misclassified = 0
    for text, true_lbl, pred_lbl in zip(X_test_raw, y_test, preds_test):
        if true_lbl != pred_lbl:
            misclassified += 1
            print(f"  [SALAH PREDIKSI]:")
            print(f"    Teks         : '{text}'")
            print(f"    Label Sejati : {class_names[true_lbl]} (Kelas {true_lbl})")
            print(f"    Prediksi     : {class_names[pred_lbl]} (Kelas {pred_lbl})")

    if misclassified == 0:
        print("  Semua sampel test set berhasil diklasifikasikan dengan benar 100%!")

    # Verifikasi assert
    assert test_acc >= 0.60, "Model teks harus melampaui performa tebakan acak (33.3%)"
    print("\nStatus: text_classification.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

