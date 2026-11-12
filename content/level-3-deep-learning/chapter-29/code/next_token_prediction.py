"""
Chapter 29 - Large Language Models & Generative AI
Script 2: Next-Token Prediction & Conditional Probability
Mendemonstrasikan perumusan probabilistik model bahasa:
P(w_t | w_1, w_2, ..., w_{t-1}) melalui proyeksi logits dan fungsi Softmax.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


def simulate_next_token_logits(context: List[str], vocab: List[str]) -> torch.Tensor:
    """
    Simulasi distribusi logits realistis berdasarkan konteks bahasa Indonesia.
    """
    torch.manual_seed(42)
    # Konteks: "Kecerdasan buatan berkembang dengan sangat"
    # Target paling masuk akal: "cepat", "pesat", "hebat"
    logits = torch.randn(len(vocab)) * 0.8

    # Tingkatkan nilai logits untuk kata-kata kontekstual
    context_str = " ".join(context).lower()
    if "sangat" in context_str:
        if "cepat" in vocab:
            logits[vocab.index("cepat")] += 4.5
        if "pesat" in vocab:
            logits[vocab.index("pesat")] += 4.2
        if "hebat" in vocab:
            logits[vocab.index("hebat")] += 2.8
        if "lambat" in vocab:
            logits[vocab.index("lambat")] += 1.0

    return logits


def main():
    print("=" * 70)
    print("CHAPTER 29: NEXT-TOKEN PREDICTION & CONDITIONAL PROBABILITY")
    print("=" * 70)

    vocab = [
        "cepat", "pesat", "hebat", "lambat", "komputer", "kucing",
        "makan", "tinggi", "luas", "tidur", "hujan", "sepatu"
    ]

    context = ["Kecerdasan", "buatan", "berkembang", "dengan", "sangat"]
    print(f"\nKonteks Masukan: '{' '.join(context)}'")
    print(f"Panjang Konteks: {len(context)} token")
    print(f"Ukuran Kosakata: {len(vocab)} kata")

    # Hitung logits mentah (output linear head sebelum aktivasi)
    logits = simulate_next_token_logits(context, vocab)

    # Ubah logits menjadi distribusi probabilitas kondisional menggunakan Softmax
    probs = F.softmax(logits, dim=-1)

    print("\n[DISTRIBUSI PROBABILITAS NEXT-TOKEN]")
    print(f"{'Kata':<12} | {'Logit (z_i)':<12} | {'Probabilitas P(w_t | C)':<22} | {'Visualisasi'}")
    print("-" * 65)

    # Urutkan berdasarkan probabilitas menurun
    sorted_indices = torch.argsort(probs, descending=True)
    for idx in sorted_indices:
        word = vocab[idx.item()]
        logit_val = logits[idx].item()
        prob_val = probs[idx].item()
        bar = "█" * int(prob_val * 40)
        print(f"{word:<12} | {logit_val:>11.4f} | {prob_val:>21.4f} | {bar}")

    # Verifikasi matematis sifat distribusi probabilitas
    sum_probs = torch.sum(probs).item()
    print("-" * 65)
    print(f"Total Probabilitas Seluruh Kosakata: {sum_probs:.6f} (Wajib = 1.000000)")

    top_token = vocab[sorted_indices[0].item()]
    top_prob = probs[sorted_indices[0]].item()
    print(f"\nPrediksi Token Berikutnya Paling Mungkin: '{top_token}' (P = {top_prob:.2%})")

    print("\n[PENJELASAN KONSEP]")
    print("1. LLM TIDAK sekadar memilih kata yang paling sering muncul secara global.")
    print("2. LLM memodelkan probabilitas bersyarat P(w_t | konteks) secara adaptif.")
    print("3. Kata 'kucing' atau 'sepatu' memperoleh probabilitas mendekati nol karena tidak selaras semantik.")


if __name__ == "__main__":
    main()

