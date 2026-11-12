"""
Chapter 29 - Large Language Models & Generative AI
Script 5: Greedy Decoding & The Repetition Dilemma
Mendemonstrasikan strategi Greedy Decoding (argmax), sifat deterministiknya,
serta kelemahannya: rentan terjebak dalam pengulangan kata tanpa henti (degenerate repetition).
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


class RepetitiveModelSimulator:
    """
    Simulator model yang memodelkan jebakan pengulangan greedy.
    Jika kata tertentu terpilih, model memiliki kecenderungan mengulang frasa sebelumnya.
    """
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}

    def get_logits(self, context_ids: List[int]) -> torch.Tensor:
        torch.manual_seed(100)
        V = len(self.vocab)
        logits = torch.randn(V) * 0.2
        last_word = self.id2token[context_ids[-1]]

        # Pola pengulangan: "sangat" -> "sangat", "hebat" -> "dan", "dan" -> "sangat"
        if last_word == "model":
            logits[self.token2id["ini"]] = 3.5
        elif last_word == "ini":
            logits[self.token2id["bekerja"]] = 3.8
        elif last_word == "bekerja":
            logits[self.token2id["sangat"]] = 4.0
        elif last_word == "sangat":
            logits[self.token2id["hebat"]] = 3.9
        elif last_word == "hebat":
            logits[self.token2id["dan"]] = 4.1
        elif last_word == "dan":
            logits[self.token2id["sangat"]] = 4.2  # Memicu loop pengulangan!
        return logits


def greedy_decode(model: RepetitiveModelSimulator, start_tokens: List[str], max_steps: int = 12) -> List[str]:
    context = [model.token2id[w] for w in start_tokens]
    
    for step in range(max_steps):
        logits = model.get_logits(context)
        # Argmax murni: pilih token dengan probabilitas mutlak tertinggi
        next_id = torch.argmax(logits).item()
        context.append(next_id)

    return [model.id2token[i] for i in context]


def main():
    print("=" * 70)
    print("CHAPTER 29: GREEDY DECODING & DEGENERATE REPETITION")
    print("=" * 70)

    vocab = ["model", "ini", "bekerja", "sangat", "hebat", "dan", "cepat", "pintar", "<eos>"]
    model = RepetitiveModelSimulator(vocab)
    prompt = ["model"]

    print(f"Prompt Masukan: '{' '.join(prompt)}'")

    # Uji Sifat Deterministik: Jalankan 3 kali berturut-turut
    print("\n[UJI DETERMINISTIK: 3 KALI RUNNING]")
    run1 = greedy_decode(model, prompt, max_steps=10)
    run2 = greedy_decode(model, prompt, max_steps=10)
    run3 = greedy_decode(model, prompt, max_steps=10)

    print(f"Run 1: {' '.join(run1)}")
    print(f"Run 2: {' '.join(run2)}")
    print(f"Run 3: {' '.join(run3)}")

    is_identical = (run1 == run2 == run3)
    print(f"\nApakah seluruh luaran 100% identik? -> {is_identical}")

    # Analisis Kelemahan: Terjebak dalam Pola Pengulangan (Loop)
    print("\n[ANALISIS KELEMAHAN: JEBAKAN PENGULANGAN (REPETITION LOOP)]")
    long_run = greedy_decode(model, prompt, max_steps=14)
    print(f"Luaran Panjang: {' '.join(long_run)}")
    
    # Deteksi pengulangan frasa "sangat hebat dan"
    text_str = " ".join(long_run)
    repeat_phrase = "sangat hebat dan"
    count = text_str.count(repeat_phrase)
    print(f"Frasa '{repeat_phrase}' berulang sebanyak: {count} kali!")

    print("\n[KESIMPULAN ILMIAH]")
    print("1. Greedy decoding selalu memilih token dengan probabilitas lokal tertinggi: w_t = argmax P(w | C).")
    print("2. Keunggulan: Sangat cepat, hemat komputasi, dan sepenuhnya deterministik.")
    print("3. Kelemahan fatal: Buta terhadap masa depan global dan mudah terjebak dalam siklus pengulangan tak berujung.")
    print("4. Solusi: Menggunakan teknik Sampling probabilistik (Temperature, Top-k, Top-p).")


if __name__ == "__main__":
    main()

