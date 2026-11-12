"""
Chapter 29 - Large Language Models & Generative AI
Script 3: Language Model Loss & Perplexity (PPL)
Mendemonstrasikan perhitungan Cross-Entropy Loss, Negative Log-Likelihood (NLL),
dan metrik Perplexity: PPL = exp(CrossEntropy).
"""

import math
import torch
import torch.nn.functional as F


def calculate_perplexity(loss: float) -> float:
    """Menghitung Perplexity dari Cross-Entropy Loss."""
    return math.exp(loss)


def main():
    print("=" * 70)
    print("CHAPTER 29: LANGUAGE MODEL TRAINING OBJECTIVE & PERPLEXITY")
    print("=" * 70)

    # Kosakata mini
    vocab = ["<pad>", "saya", "belajar", "deep", "learning", "hari", "ini", "senang", "<eos>"]
    V = len(vocab)
    print(f"Ukuran Kosakata (V) = {V}")

    # Sekuens target: "saya" -> "belajar" -> "deep" -> "learning"
    # Target IDs: [1, 2, 3, 4]
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    T = len(targets)

    # Skenario A: Model Terlatih Baik (Memberikan probabilitas tinggi pada target)
    logits_good = torch.zeros(T, V)
    for t, target_idx in enumerate(targets):
        logits_good[t] = torch.randn(V) * 0.5
        logits_good[t, target_idx] = 4.0  # Logit target dominan

    # Skenario B: Model Belum Terlatih / Acak (Distribusi rata / salah prediksi)
    logits_random = torch.zeros(T, V)
    for t in range(T):
        logits_random[t] = torch.randn(V) * 0.5  # Logit acak

    # Hitung Cross-Entropy Loss (NLLLoss over LogSoftmax)
    loss_good = F.cross_entropy(logits_good, targets).item()
    loss_random = F.cross_entropy(logits_random, targets).item()

    ppl_good = calculate_perplexity(loss_good)
    ppl_random = calculate_perplexity(loss_random)

    # Kasus Teoretis Batas: Model Tebak Acak Seragam
    loss_uniform = math.log(V)
    ppl_uniform = V

    print("\n[PERBANDINGAN KINERJA MODEL BAHASA]")
    print(f"{'Metrik / Indikator':<28} | {'Model Bagus (Trained)':<22} | {'Model Acak (Untrained)':<22} | {'Model Seragam (Batas)'}")
    print("-" * 90)
    print(f"{'Cross-Entropy Loss (nat)':<28} | {loss_good:>22.4f} | {loss_random:>22.4f} | {loss_uniform:>22.4f}")
    print(f"{'Perplexity (PPL)':<28} | {ppl_good:>22.4f} | {ppl_random:>22.4f} | {ppl_uniform:>22.4f}")

    print("\n[TELUSUR LANGKAH-DEMI-LANGKAH (WORKED EXAMPLE MODEL BAGUS)]")
    probs_good = F.softmax(logits_good, dim=-1)
    for t, target_idx in enumerate(targets):
        prob_target = probs_good[t, target_idx].item()
        nll_t = -math.log(prob_target)
        word = vocab[target_idx.item()]
        print(f"Langkah {t+1}: Kata '{word}' | P(target) = {prob_target:6.4f} | NLL = -ln(P) = {nll_t:.4f}")

    mean_nll = torch.mean(-torch.log(probs_good[torch.arange(T), targets])).item()
    print("-" * 65)
    print(f"Rata-rata NLL (Cross-Entropy Loss): {mean_nll:.4f}")
    print(f"Perplexity = exp({mean_nll:.4f}) = {math.exp(mean_nll):.4f}")

    print("\n[INTERPRETASI ILMIAH]")
    print("1. Perplexity mengukur tingkat kebingungan model: ekuivalen dengan ukuran pilihan kata yang membingungkan.")
    print("2. PPL = 1.35 berarti model rata-rata hanya ragu-ragu di antara 1.35 pilihan kata.")
    print("3. PPL = 8.65 pada model acak mendekati ukuran kosakata (V=9), artinya model menebak hampir secara buta.")
    print("4. Semakin rendah Cross-Entropy Loss dan Perplexity, semakin unggul kualitas representasi model bahasa.")


if __name__ == "__main__":
    main()

