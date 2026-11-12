"""
Chapter 29 - Large Language Models & Generative AI
Script 1: Tokenization Fundamentals & Subword Tokenization Concept
Mendemonstrasikan perbedaan word-level, character-level, dan subword tokenization,
pemetaan kosakata (vocabulary to ID), token khusus (<pad>, <unk>, <bos>, <eos>),
serta padding dan truncation untuk LLM context window.
"""

import re
from typing import List, Dict, Tuple


class SimpleSubwordTokenizer:
    """
    Subword tokenizer sederhana berbasis konsep Byte-Pair Encoding (BPE).
    Mendemonstrasikan pemecahan kata majemuk / imbuhan menjadi sub-satuan leksikal.
    """
    def __init__(self, vocab: List[str]):
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = self.special_tokens + vocab
        self.token2id: Dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token: Dict[int, str] = {idx: token for token, idx in self.token2id.items()}

    def tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
        tokens = []
        for word in words:
            if word in self.token2id:
                tokens.append(word)
            else:
                # Coba segmentasi subword sederhana (prefix + root / suffix)
                segmented = False
                for i in range(len(word) - 1, 1, -1):
                    prefix = word[:i]
                    suffix = "##" + word[i:]
                    if prefix in self.token2id and suffix in self.token2id:
                        tokens.extend([prefix, suffix])
                        segmented = True
                        break
                if not segmented:
                    tokens.append("<unk>")
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = ["<bos>"] + tokens + ["<eos>"]
        return [self.token2id.get(t, self.token2id["<unk>"]) for t in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = [self.id2token.get(idx, "<unk>") for idx in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        # Rekonstruksi subword '##'
        text = " ".join(tokens).replace(" ##", "")
        return text

    def pad_and_truncate(self, token_ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
        """Membatasi ke context window dan menambahkan padding mask."""
        if len(token_ids) > max_length:
            truncated = token_ids[:max_length]
            mask = [1] * max_length
            return truncated, mask
        else:
            padded = token_ids + [self.token2id["<pad>"]] * (max_length - len(token_ids))
            mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
            return padded, mask


def main():
    print("=" * 70)
    print("CHAPTER 29: TOKENIZATION FUNDAMENTALS & SUBWORD BPE CONCEPT")
    print("=" * 70)

    # Kosakata dasar bahasa Indonesia terstruktur dengan subword units
    vocab_subwords = [
        "model", "bahasa", "besar", "menghasilkan", "teks", "cerdas",
        "pembelajaran", "mesin", "sangat", "cepat", "bantu", "kerja",
        "manusia", "ajar", "hitung", "##kan", "##nya", "##an", "##i",
        "komputer", "algoritma", "ai", "generatif", "akurasi", "tinggi"
    ]

    tokenizer = SimpleSubwordTokenizer(vocab_subwords)
    print(f"\nUkuran Kosakata (termasuk 4 token khusus): {len(tokenizer.vocab)}")
    print(f"Token Khusus: {tokenizer.special_tokens}")

    # Contoh 1: Kalimat dengan kata dikenal penuh
    kalimat_1 = "Model bahasa besar menghasilkan teks cerdas."
    tokens_1 = tokenizer.tokenize(kalimat_1)
    ids_1 = tokenizer.encode(kalimat_1)
    decoded_1 = tokenizer.decode(ids_1)

    print(f"\n[Kasus 1: Kata Dikenal Penuh]")
    print(f"Kalimat Asli    : '{kalimat_1}'")
    print(f"Token-token     : {tokens_1}")
    print(f"Token IDs (BOS/EOS): {ids_1}")
    print(f"Rekonstruksi Teks: '{decoded_1}'")

    # Contoh 2: Kata berafiks yang disegmentasi subword
    kalimat_2 = "Bantuan komputer mengajarkan ai generatif."
    # 'bantuan' -> 'bantu' + '##an'
    tokenizer.vocab.append("meng##")
    tokenizer.token2id["meng##"] = len(tokenizer.vocab) - 1
    tokenizer.id2token[len(tokenizer.vocab) - 1] = "meng##"

    tokens_2 = tokenizer.tokenize(kalimat_2)
    ids_2 = tokenizer.encode(kalimat_2)
    decoded_2 = tokenizer.decode(ids_2)

    print(f"\n[Kasus 2: Segmentasi Subword (BPE Analogy)]")
    print(f"Kalimat Asli    : '{kalimat_2}'")
    print(f"Token-token     : {tokens_2}")
    print(f"Token IDs       : {ids_2}")
    print(f"Rekonstruksi Teks: '{decoded_2}'")

    # Contoh 3: Context Window, Truncation, dan Attention Mask
    context_window_max = 8
    padded_ids, attn_mask = tokenizer.pad_and_truncate(ids_1, max_length=context_window_max)

    print(f"\n[Kasus 3: Context Window & Attention Mask (Max Length = {context_window_max})]")
    print(f"Token IDs Asli ({len(ids_1)} token): {ids_1}")
    print(f"Padded / Truncated IDs   : {padded_ids}")
    print(f"Attention Mask (1=Data, 0=Pad): {attn_mask}")
    print(f"Token Rekonstruksi       : {[tokenizer.id2token[i] for i in padded_ids]}")

    print("\n[VERIFIKASI ILMIAH]")
    print("1. Subword tokenization memitigasi ledakan OOV dengan memecah kata majemuk/imbuhan.")
    print("2. Token ID merupakan masukan diskret yang dikonversi menjadi embedding vektor dalam LLM.")
    print("3. Attention mask secara ketat mengabaikan komputasi pada token <pad>.")


if __name__ == "__main__":
    main()

