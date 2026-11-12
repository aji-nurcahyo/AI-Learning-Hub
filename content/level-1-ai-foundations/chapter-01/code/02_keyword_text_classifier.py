"""
02_keyword_text_classifier.py
Contoh 2: Klasifikasi Teks Berbasis Kata Kunci (Keyword-Based Classifier).

Tujuan:
Menunjukkan cara kerja klasifikasi deterministik berbasis pencarian kata kunci,
sekaligus mendemonstrasikan kerentanan terhadap False Positive dan False Negative.
"""

from typing import List, Dict

SPAM_KEYWORDS = {"hadiah", "gratis", "klaim", "menang", "promo", "transfer"}

def classify_message_by_keyword(text: str) -> str:
    """
    Mengklasifikasikan pesan sebagai 'SPAM' atau 'HAM' (bukan spam)
    hanya berdasarkan kemunculan kata kunci tertentu.
    """
    text_lower = text.lower()
    for kw in SPAM_KEYWORDS:
        if kw in text_lower:
            return "SPAM"
    return "HAM"

if __name__ == "__main__":
    print("=== DEMO: KEYWORD-BASED TEXT CLASSIFIER ===")
    
    # Dataset ilustratif untuk demonstrasi keberhasilan dan kegagalan
    test_cases: List[Dict[str, str]] = [
        {
            "id": "1",
            "text": "Selamat Anda menang undian berhadiah 100 juta!",
            "true_label": "SPAM",
            "keterangan": "True Positive (Berhasil terdeteksi karena ada 'menang' dan 'berhadiah')"
        },
        {
            "id": "2",
            "text": "Besok kita kumpul di perpustakaan jam 10 pagi ya.",
            "true_label": "HAM",
            "keterangan": "True Negative (Pesan normal tanpa kata kunci spam)"
        },
        {
            "id": "3",
            "text": "Buku referensi kuliah ini gratis dipinjam dari perpustakaan.",
            "true_label": "HAM",
            "keterangan": "False Positive! (Pesan akademik normal tapi dianggap SPAM karena kata 'gratis')"
        },
        {
            "id": "4",
            "text": "Tolong kirimkan sejumlah dana darurat ke rekening ini segera.",
            "true_label": "SPAM",
            "keterangan": "False Negative! (Pesan penipuan nyata tapi lolos karena memakai kata 'dana', bukan 'transfer')"
        }
    ]

    for item in test_cases:
        pred = classify_message_by_keyword(item["text"])
        is_correct = pred == item["true_label"]
        status = "BENAR" if is_correct else "SALAH (ERROR)"
        print(f"\n[Kasus {item['id']}] Status: {status}")
        print(f"  Pesan       : \"{item['text']}\"")
        print(f"  Label Asli  : {item['true_label']}")
        print(f"  Prediksi    : {pred}")
        print(f"  Analisis    : {item['keterangan']}")

    print("\n[Catatan Analisis]")
    print("Pencocokan kata kunci mengabaikan konteks semantik kalimat.")
    print("Inilah alasan mengapa Machine Learning berbasis data kontekstual dibutuhkan.")

