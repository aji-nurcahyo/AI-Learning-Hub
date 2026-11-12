"""
Chapter 29 - Large Language Models & Generative AI
Script 6: Temperature Scaling & Entropy Control
Mendemonstrasikan penskalaan logits dengan Temperature: P(w_i) = softmax(z_i / T).
Membuktikan bagaimana T rendah mempertajam probabilitas dan T tinggi meratakannya.
"""

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Menerapkan penskalaan temperature pada vektor logits."""
    assert temperature > 0.0, "Temperature wajib bernilai positif (> 0)."
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)


def sample_from_probs(probs: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
    """Mengambil sampel multinomial dari distribusi probabilitas."""
    return torch.multinomial(probs, num_samples=num_samples, replacement=True)


def main():
    print("=" * 70)
    print("CHAPTER 29: TEMPERATURE SAMPLING & PROBABILITY SHAPING")
    print("=" * 70)

    vocab = ["kucing", "anjing", "kelinci", "burung", "ikan"]
    # Logits mentah awal
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5])

    temperatures = [0.2, 0.7, 1.0, 1.8]

    print(f"Kosakata Uji : {vocab}")
    print(f"Logits Awal  : {logits.tolist()}\n")

    print(f"{'Kata':<10} | " + " | ".join([f"T = {T:<4} (P)" for T in temperatures]))
    print("-" * 65)

    probs_dict = {}
    for T in temperatures:
        probs_dict[T] = apply_temperature(logits, T)

    for i, word in enumerate(vocab):
        row = f"{word:<10} | "
        for T in temperatures:
            p_val = probs_dict[T][i].item()
            row += f"{p_val:10.4f} | "
        print(row)

    print("-" * 65)

    # Analisis Entropi: Ukuran Keacakan / Ketidakpastian
    print("\n[ANALISIS ENTROPI SHANNON (H = -sum P ln P)]")
    for T in temperatures:
        p = probs_dict[T]
        entropy = -torch.sum(p * torch.log(p)).item()
        print(f"Temperature T = {T:<4} -> Entropi Shannon H = {entropy:.4f}")

    # Simulasi Pengambilan Sampel Nyata (Monte Carlo N = 2000)
    print("\n[SIMULASI FREKUENSI SAMPEL TERPILIH (N = 2.000 Sampel)]")
    torch.manual_seed(42)
    for T in [0.2, 1.0, 1.8]:
        samples = sample_from_probs(probs_dict[T], num_samples=2000)
        counts = torch.bincount(samples, minlength=len(vocab))
        percentages = (counts.float() / 2000.0) * 100.0
        
        counts_str = ", ".join([f"{vocab[i]}: {percentages[i]:.1f}%" for i in range(len(vocab))])
        print(f"T = {T:<4} -> {counts_str}")

    print("\n[RANGKUMAN TEORETIS]")
    print("1. Temperature T < 1.0 (Dingin): Mempertajam puncak distribusi. Menghasilkan jawaban yang sangat konservatif, tertebak, dan faktual.")
    print("2. Temperature T = 1.0 (Netral): Menjaga distribusi probabilitas asli dari lapisan output model.")
    print("3. Temperature T > 1.0 (Panas): Meratakan probabilitas menuju distribusi seragam (uniform). Meningkatkan diversitas, kreativitas, namun meningkatkan risiko inkoherensi atau halusinasi.")
    print("4. CATATAN: Temperature TIDAK mengubah urutan peringkat (ranking) kata; ia hanya mengatur jarak relatif antar-skor probabilitas.")


if __name__ == "__main__":
    main()

