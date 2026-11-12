"""
Chapter 29 - Large Language Models & Generative AI
Script 7: Top-k Sampling & Vocabulary Truncation
Mendemonstrasikan pemangkasan logits dengan strategi Top-k:
Menyaring hanya k token probabilitas teratas, memaksa sisanya menjadi -inf,
dan melakukan renormalisasi Softmax.
"""

import torch
import torch.nn.functional as F


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Menyaring logits sehingga hanya menyisakan k token dengan nilai tertinggi.
    Logits di luar k teratas digantikan dengan -inf.
    """
    assert k > 0, "Nilai k harus lebih besar dari 0."
    v_size = logits.size(-1)
    k = min(k, v_size)

    # Temukan nilai ambang batas pada peringkat k
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[-1]

    # Gantikan seluruh nilai di bawah ambang batas dengan -infinity
    filtered_logits = logits.clone()
    filtered_logits[filtered_logits < threshold] = float("-inf")
    return filtered_logits


def main():
    print("=" * 70)
    print("CHAPTER 29: TOP-K SAMPLING STRATEGY")
    print("=" * 70)

    vocab = ["makan", "minum", "tidur", "lari", "baca", "tulis", "duduk", "nyanyi"]
    logits = torch.tensor([5.0, 4.2, 3.8, 2.5, 1.8, 1.0, 0.2, -1.0])

    print(f"Kosakata Uji ({len(vocab)} kata): {vocab}")
    print(f"Logits Awal: {logits.tolist()}\n")

    k_values = [1, 3, 5, len(vocab)]

    print(f"{'Kata':<10} | {'Logit':<6} | " + " | ".join([f"k = {k:<2} (P)" for k in k_values]))
    print("-" * 65)

    probs_dict = {}
    for k in k_values:
        filtered = top_k_filtering(logits, k)
        probs_dict[k] = F.softmax(filtered, dim=-1)

    for i, word in enumerate(vocab):
        row = f"{word:<10} | {logits[i].item():>6.2f} | "
        for k in k_values:
            p_val = probs_dict[k][i].item()
            row += f"{p_val:9.4f} | "
        print(row)

    print("-" * 65)

    print("\n[VERIFIKASI SIFAT RENORMALISASI PROBABILITAS]")
    for k in k_values:
        sum_p = probs_dict[k].sum().item()
        non_zero_tokens = (probs_dict[k] > 0).sum().item()
        print(f"k = {k:<2} -> Jumlah P = {sum_p:.6f} | Jumlah Token Aktif = {non_zero_tokens}")

    print("\n[ANALISIS TEKNIS]")
    print("1. Pada k = 1: Top-k bertindak persis sama dengan Greedy Decoding (P=1.0 pada kata teratas).")
    print("2. Pada k = 3: Hanya 3 kata teratas ('makan', 'minum', 'tidur') yang bersaing; kata-kata lain dipangkas menjadi P=0.0000.")
    print("3. Keunggulan Top-k: Memotong 'ekor distribusi panjang' (long-tail) yang berpotensi menghasilkan kata absurd/aneh.")
    print("4. Kelemahan Top-k: Nilai k statis tidak adaptif terhadap bentuk distribusi (ketika model sangat yakin, k=5 tetap menyertakan kata yang tidak relevan; ketika model bingung, k=3 memotong kata potensial).")


if __name__ == "__main__":
    main()

