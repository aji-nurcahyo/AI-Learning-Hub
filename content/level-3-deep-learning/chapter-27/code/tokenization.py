"""
Chapter 27 - Natural Language Processing Fundamentals
Script 2: Tokenization, Vocabulary Building, Padding & Truncation

Mengimplementasikan tokenisasi berbasis ekspresi reguler, pembangunan kamus kata (Vocabulary),
penanganan token khusus (<pad>, <unk>, <bos>, <eos>), konversi teks ke urutan indeks integer,
serta operasi penyeragaman panjang sekuens (Padding dan Truncation).
"""

import re
from typing import List, Dict, Tuple, Optional


class SimpleTokenizer:
    """Tokenizer berbasis aturan regex dengan manajemen kosakata khusus."""

    PAD_TOKEN = "<pad>"  # Indeks 0: Padding untuk menyamakan panjang batch
    UNK_TOKEN = "<unk>"  # Indeks 1: Out-of-Vocabulary (kata tak dikenal)
    BOS_TOKEN = "<bos>"  # Indeks 2: Beginning-of-Sequence
    EOS_TOKEN = "<eos>"  # Indeks 3: End-of-Sequence

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._initialize_vocab()

    def _initialize_vocab(self):
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def tokenize(self, text: str) -> List[str]:
        """Memecah kalimat menjadi token kata dan tanda baca via regex."""
        text = text.lower().strip()
        # Memisahkan kata atau tanda baca mandiri
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def build_vocab(self, corpus: List[str]):
        """Membangun kosakata dari korpus teks berdasarkan frekuensi kemunculan."""
        freq: Dict[str, int] = {}
        for sentence in corpus:
            for token in self.tokenize(sentence):
                freq[token] = freq.get(token, 0) + 1

        # Tambahkan kata yang memenuhi ambang batas frekuensi
        for token, count in sorted(freq.items(), key=lambda item: -item[1]):
            if count >= self.min_freq and token not in self.word2idx:
                new_idx = len(self.word2idx)
                self.word2idx[token] = new_idx
                self.idx2word[new_idx] = token

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_len: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        """Mengubah teks mentah menjadi sekuens indeks integer (token IDs)."""
        raw_tokens = self.tokenize(text)
        token_ids = [self.word2idx.get(t, self.unk_idx) for t in raw_tokens]

        if add_special_tokens:
            token_ids = (
                [self.word2idx[self.BOS_TOKEN]]
                + token_ids
                + [self.word2idx[self.EOS_TOKEN]]
            )

        # Truncation: Potong jika melebihi panjang maksimum
        if truncation and max_len is not None and len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            if add_special_tokens:
                token_ids[-1] = self.word2idx[self.EOS_TOKEN]

        # Padding: Tambahkan token <pad> jika panjang kurang dari max_len
        if padding and max_len is not None and len(token_ids) < max_len:
            pad_count = max_len - len(token_ids)
            token_ids = token_ids + [self.pad_idx] * pad_count

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Merekontruksi kembali teks dari daftar indeks integer."""
        words = []
        for idx in token_ids:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if skip_special_tokens and word in self.special_tokens:
                continue
            words.append(word)
        return " ".join(words)


def main():
    print("=== Chapter 27: Demo Tokenization & Vocabulary Pipeline ===")

    corpus = [
        "Kecerdasan buatan mempelajari penalaran dan pembelajaran mesin.",
        "Model pembelajaran mesin membutuhkan data pelatihan berkualitas.",
        "Natural language processing memproses teks bahasa manusia.",
    ]

    tokenizer = SimpleTokenizer(min_freq=1)
    tokenizer.build_vocab(corpus)

    print(f"Ukuran Kosakata (Vocabulary Size): {tokenizer.vocab_size} entri")
    print("Contoh entri awal kosakata:")
    for i in range(min(10, tokenizer.vocab_size)):
        print(f"  ID {i:2d} -> '{tokenizer.idx2word[i]}'")

    sample_sentence = "Kecerdasan buatan memproses bahasa asing baru!"
    max_len = 10

    encoded = tokenizer.encode(sample_sentence, max_len=max_len, padding=True, truncation=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    decoded_clean = tokenizer.decode(encoded, skip_special_tokens=True)

    print(f"\nKalimat Uji: '{sample_sentence}'")
    print(f"Hasil Enkoding (max_len={max_len}): {encoded}")
    print(f"Hasil Dekoding Lengkap : '{decoded}'")
    print(f"Hasil Dekoding Bersih   : '{decoded_clean}'")

    # Verifikasi assert
    assert len(encoded) == max_len, f"Panjang sekuens harus tepat {max_len}"
    assert encoded[0] == tokenizer.word2idx["<bos>"], "Token pertama harus <bos>"
    assert tokenizer.unk_idx in encoded, "Kata baru 'asing' harus dipetakan ke <unk>"
    assert encoded[-1] == tokenizer.pad_idx or encoded[-1] == tokenizer.word2idx["<eos>"]

    print("\nStatus: tokenization.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

