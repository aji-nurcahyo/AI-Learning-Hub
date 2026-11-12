"""
Chapter 11: Introduction to Machine Learning
Script 1: rule_based_vs_learning.py

Tujuan:
Membandingkan pendekatan tradisional berbasis aturan (Rule-Based Systems)
dengan pendekatan Machine Learning pada tugas klasifikasi teks spam sederhana.

Konsep:
- Rule-Based: Aturan if-else dirancang manual oleh manusia. Kaku dan rapuh terhadap variasi kata.
- Machine Learning: Pola asosiasi kata dipelajari secara otomatis dari data berlabel.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def rule_based_classifier(text: str) -> int:
    """Klasifikasi rule-based menggunakan pencocokan kata kunci manual."""
    spam_keywords = ["hadiah", "gratis", "klaim", "menang", "klik di sini", "transfer"]
    text_lower = text.lower()
    for kw in spam_keywords:
        if kw in text_lower:
            return 1  # Spam
    return 0  # Ham (Bukan spam)


def main():
    print("=" * 65)
    print("DEMO: RULE-BASED SYSTEMS VS MACHINE LEARNING")
    print("=" * 65)

    # 1. Dataset latihan teks pendek (Bahasa Indonesia)
    train_texts = [
        "Selamat Anda menang hadiah uang tunai segera klaim sekarang",
        "Dapatkan promo gratis kuota internet klik di sini",
        "Pemberitahuan jadwal kuliah semester genap telah diperbarui",
        "Tolong kirimkan revisi draft proposal skripsi besok pagi",
        "Rapat koordinasi praktikum akan diadakan hari Kamis",
        "Segera transfer uang muka untuk konfirmasi undian mobil",
        "Pengumuman libur nasional hari raya idul fitri kampus",
        "Diskon spesial gratis biaya pendaftaran kursus online"
    ]
    # 1: Spam, 0: Ham
    train_labels = [1, 1, 0, 0, 0, 1, 0, 1]

    # 2. Dataset pengujian (memuat variasi kata dan konteks baru)
    test_texts = [
        "Jadwal pengumpulan tugas kecerdasan buatan dimundurkan",        # Seharusnya Ham (0)
        "Dapatkan bonus tanpa bayar sepeser pun sekarang juga",          # Seharusnya Spam (1) - tanpa kata kunci persis 'gratis'
        "Klaim tiket webinar riset artificial intelligence",            # Seharusnya Ham (0) - memuat kata 'klaim' tapi konteks edukasi
        "Pemenang undian kampus harap menghubungi bagian keuangan",     # Seharusnya Spam (1)
    ]
    test_labels = [0, 1, 0, 1]

    # 3. Evaluasi Pendekatan Rule-Based
    print("\n--- 1. Evaluasi Pendekatan Rule-Based (Manual If-Else) ---")
    rb_preds = [rule_based_classifier(t) for t in test_texts]
    for text, true_lbl, pred in zip(test_texts, test_labels, rb_preds):
        status = "BENAR" if true_lbl == pred else "SALAH"
        print(f"Teks: \"{text}\"")
        print(f"  Target: {true_lbl} | Prediksi Rule-Based: {pred} -> [{status}]")

    rb_acc = accuracy_score(test_labels, rb_preds)
    print(f"Akurasi Rule-Based: {rb_acc * 100:.1f}%")

    # 4. Evaluasi Pendekatan Machine Learning (Multinomial Naive Bayes)
    print("\n--- 2. Evaluasi Pendekatan Machine Learning (Data-Driven) ---")
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    ml_model = MultinomialNB()
    ml_model.fit(X_train, train_labels)

    ml_preds = ml_model.predict(X_test)
    for text, true_lbl, pred in zip(test_texts, test_labels, ml_preds):
        status = "BENAR" if true_lbl == pred else "SALAH"
        print(f"Teks: \"{text}\"")
        print(f"  Target: {true_lbl} | Prediksi ML: {pred} -> [{status}]")

    ml_acc = accuracy_score(test_labels, ml_preds)
    print(f"Akurasi Machine Learning: {ml_acc * 100:.1f}%")

    print("\n--- Kesimpulan Konseptual ---")
    print("Rule-based mudah terkelabui oleh variasi sinonim atau kemunculan kata kunci")
    print("dalam konteks yang berbeda. Machine Learning mempelajari bobot statistik")
    print("seluruh fitur kata secara simultan dari kumpulan data pengalaman.")
    print("=" * 65)


if __name__ == "__main__":
    main()

