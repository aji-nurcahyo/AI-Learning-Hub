"""
Chapter 27 - Natural Language Processing Fundamentals
Script 5: N-gram Feature Extraction & Local Context Modeling

Mengekstraksi fitur n-gram (Unigram, Bigram, Trigram) dari teks, mendemonstrasikan
kemampuan n-gram dalam menangkap konteks lokal negasi frasa ('tidak ramah' vs 'ramah'),
serta menganalisis fenomena ledakan ukuran kosakata (Vocabulary Explosion) saat nilai n bertambah.
"""

from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import CountVectorizer


def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Mengekstraksi n-gram menggunakan teknik sliding window sepanjang n elemen."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def format_ngram_list(ngrams: List[Tuple[str, ...]]) -> List[str]:
    """Menggabungkan tupel n-gram menjadi string yang mudah dibaca."""
    return ["_".join(gram) for gram in ngrams]


def main():
    print("=== Chapter 27: Demo N-gram Feature Extraction ===")

    sentence = "pelayanan staf hotel tidak ramah sama sekali"
    tokens = sentence.split()

    print(f"Kalimat Uji: '{sentence}'")
    print(f"Token Kata : {tokens} (Jumlah: {len(tokens)})")

    unigrams = extract_ngrams(tokens, n=1)
    bigrams = extract_ngrams(tokens, n=2)
    trigrams = extract_ngrams(tokens, n=3)

    print("\n1. Ekstraksi Unigram (n=1):")
    print("  ", format_ngram_list(unigrams))

    print("\n2. Ekstraksi Bigram (n=2):")
    print("  ", format_ngram_list(bigrams))

    print("\n3. Ekstraksi Trigram (n=3):")
    print("  ", format_ngram_list(trigrams))

    # Analisis Konteks Negasi
    print("\n[Analisis Konteks Negasi pada Analisis Sentimen]:")
    s_pos = "pelayanan staf ramah sekali"
    s_neg = "pelayanan staf tidak ramah"

    vec_unigram = CountVectorizer(ngram_range=(1, 1))
    vec_bigram = CountVectorizer(ngram_range=(1, 2))

    vec_unigram.fit([s_pos, s_neg])
    vec_bigram.fit([s_pos, s_neg])

    print("  Kosakata Model Unigram (1-gram):")
    print("   ", vec_unigram.get_feature_names_out().tolist())
    print("  -> Perhatikan: Kata 'ramah' muncul pada kedua kalimat, mengaburkan sentimen sebenarnya!")

    print("  Kosakata Model Gabungan Unigram + Bigram (1-2 gram):")
    print("   ", vec_bigram.get_feature_names_out().tolist())
    print("  -> Fitur 'tidak ramah' kini terbentuk sebagai prediktor mandiri pembawa sentimen negatif!")

    # Demonstrasi Ledakan Kosakata (Vocabulary Explosion)
    corpus_sample = [
        "kecerdasan buatan mempelajari representasi pengetahuan dan penalaran logis",
        "pembelajaran mesin dan pemrosesan bahasa alami menggunakan model probabilitas",
        "arsitektur jaringan saraf tiruan memproses data teks dalam urutan temporal",
    ]

    print("\n[Analisis Ledakan Kosakata (Vocabulary Growth)]:")
    for n_min, n_max in [(1, 1), (2, 2), (3, 3), (1, 3)]:
        cv = CountVectorizer(ngram_range=(n_min, n_max))
        cv.fit(corpus_sample)
        print(f"  Rentang ({n_min}, {n_max}): {len(cv.get_feature_names_out()):2d} fitur unik")

    # Verifikasi assert
    assert ("tidak", "ramah") in bigrams, "Bigram ('tidak', 'ramah') harus berhasil terekstraksi"
    assert ("tidak", "ramah", "sama") in trigrams, "Trigram harus berhasil terekstraksi"
    print("\nStatus: ngram_features.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

