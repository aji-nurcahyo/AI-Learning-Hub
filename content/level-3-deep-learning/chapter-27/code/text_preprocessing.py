"""
Chapter 27 - Natural Language Processing Fundamentals
Script 1: Text Preprocessing & Normalization Pipeline

Mendemonstrasikan tahapan pembersihan teks mentah, case folding, normalisasi slang/singkatan,
penanganan tanda baca, serta analisis dampak stopword removal terhadap pemeliharaan konteks negasi.
"""

import re
import string
from typing import List, Dict, Set

# Kamus normalisasi kata tidak baku / singkatan populer (Indonesian slang normalization)
SLANG_DICT: Dict[str, str] = {
    "gk": "tidak",
    "gak": "tidak",
    "tdk": "tidak",
    "bgt": "sangat",
    "banget": "sangat",
    "dgn": "dengan",
    "yg": "yang",
    "utk": "untuk",
    "krn": "karena",
    "bagusss": "bagus",
    "recommended": "direkomendasikan"
}

# Daftar stopword dasar, dengan pengecualian kata negasi kritis agar sentimen tidak terbalik
DEFAULT_STOPWORDS: Set[str] = {
    "yang", "di", "ke", "dari", "ini", "itu", "dan", "atau", "adalah", "pada", "untuk", "dengan"
}
NEGATION_WORDS: Set[str] = {"tidak", "bukan", "belum", "jangan", "kurang", "tanpa"}


def case_folding(text: str) -> str:
    """Mengubah seluruh karakter ke huruf kecil (lowercase)."""
    return text.lower()


def clean_special_characters(text: str, keep_punctuation: bool = False) -> str:
    """Menghapus URL, mention, hashtag, dan angka/karakter non-alfanumerik."""
    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Hapus HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Hapus karakter berulang berlebih (e.g. bagussss -> bagus)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    if not keep_punctuation:
        # Hapus tanda baca
        text = text.translate(str.maketrans("", "", string.punctuation))
    # Hapus whitespace berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slang(tokens: List[str], slang_dict: Dict[str, str] = None) -> List[str]:
    """Mengganti kata slang/singkatan menjadi bentuk baku."""
    if slang_dict is None:
        slang_dict = SLANG_DICT
    return [slang_dict.get(t, t) for t in tokens]


def remove_stopwords(tokens: List[str], preserve_negation: bool = True) -> List[str]:
    """
    Menyaring stopword dengan opsi pelestarian kata negasi (preserve_negation=True).
    Penting untuk analisis sentimen agar frasa 'tidak bagus' tidak tereduksi menjadi 'bagus'.
    """
    filtered = []
    for t in tokens:
        if preserve_negation and t in NEGATION_WORDS:
            filtered.append(t)
        elif t not in DEFAULT_STOPWORDS:
            filtered.append(t)
    return filtered


def preprocess_pipeline(raw_text: str, preserve_negation: bool = True) -> List[str]:
    """Pipeline terintegrasi dari teks mentah ke token bersih."""
    # 1. Case folding
    lowered = case_folding(raw_text)
    # 2. Pembersihan karakter khusus
    cleaned = clean_special_characters(lowered)
    # 3. Tokenisasi berbasis spasi
    tokens = cleaned.split()
    # 4. Normalisasi kata singkatan/slang
    normalized = normalize_slang(tokens)
    # 5. Penyaringan stopword adaptif
    final_tokens = remove_stopwords(normalized, preserve_negation=preserve_negation)
    return final_tokens


def main():
    print("=== Chapter 27: Demo Text Preprocessing Pipeline ===")
    
    sample_texts = [
        "Layanan ini gk bagusss bgt!! Sangat kecewa dgn respon admin http://example.com/cs.",
        "Produk yg ini TIDAK mengecewakan, sangat recommended utk dibeli!",
        "Kualitas kamera biasa saja, tapi baterai awet bgt."
    ]
    
    for idx, text in enumerate(sample_texts, 1):
        print(f"\n[Teks Mentah {idx}]:\n  '{text}'")
        tokens_safe = preprocess_pipeline(text, preserve_negation=True)
        tokens_aggressive = preprocess_pipeline(text, preserve_negation=False)
        
        print(f"  Hasil Preprocessing (Aman Negasi): {tokens_safe}")
        print(f"  Hasil Tanpa Negasi (Bahaya Bias) : {tokens_aggressive}")
        
    # Verifikasi kritis pelestarian negasi
    test_negation = "Makanan ini tidak enak"
    tokens_res = preprocess_pipeline(test_negation, preserve_negation=True)
    assert "tidak" in tokens_res, "Error: Kata negasi 'tidak' tidak boleh terhapus!"
    assert "enak" in tokens_res, "Error: Kata 'enak' harus dipertahankan!"
    print("\n[Audit Negasi]: 'tidak enak' berhasil dipertahankan secara utuh tanpa inversi sentimen.")
    print("Status: text_preprocessing.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

