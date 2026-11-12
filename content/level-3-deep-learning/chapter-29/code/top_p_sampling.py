"""
Chapter 29 - Large Language Models & Generative AI
Script 8: Top-p (Nucleus) Sampling & Dynamic Candidate Pool
Mendemonstrasikan strategi Nucleus Sampling (Holtzman et al., 2020):
Menyaring token berdasarkan ambang kumulatif probabilitas p, menghasilkan
ukuran kandidat yang adaptif secara dinamis.
"""

import torch
import torch.nn.functional as F


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Menyaring logits menggunakan Nucleus (Top-p) Filtering.
    Menyisakan subset terkecil dari token yang probabilitas kumulatifnya mencapai p.
    """
    assert 0.0 < p <= 1.0, "Nilai p harus berada dalam rentang (0.0, 1.0]."

    # Urutkan logits dari terbesar ke terkecil
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Hitung probabilitas kumulatif
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Buat mask: buang token dengan kumulatif > p (geser 1 posisi agar token pertama selalu lolos)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # Selalu pertahankan setidaknya 1 token

    # Kembalikan mask ke urutan indeks asli
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float("-inf")
    return filtered_logits


def main():
    print("=" * 70)
    print("CHAPTER 29: TOP-P (NUCLEUS) SAMPLING & DYNAMIC POOL SIZE")
    print("=" * 70)

    vocab = ["mobil", "sepeda", "motor", "pesawat", "kapal", "kuda", "jalan", "roket"]

    # Dua Skenario Nyata:
    # Kasus 1: Model Sangat Yakin (Confident / Peaky) -> "Saya menyetir ..."
    logits_peaky = torch.tensor([6.0, 4.8, 1.0, 0.5, 0.2, -1.0, -2.0, -3.0])

    # Kasus 2: Model Ragu-Ragu / Terbuka (Uncertain / Flat) -> "Saya melihat ..."
    logits_flat = torch.tensor([2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8])

    p_thresholds = [0.5, 0.8, 0.95]

    for kasus_name, current_logits in [("Kasus 1: Model Yakin (Peaky Distribution)", logits_peaky),
                                       ("Kasus 2: Model Ragu-Ragu (Flat Distribution)", logits_flat)]:
        print(f"\n[{kasus_name}]")
        raw_probs = F.softmax(current_logits, dim=-1)
        print("Probabilitas Asli:")
        for w, p_v in zip(vocab, raw_probs.tolist()):
            print(f"  {w:<8}: {p_v:.4f} ({p_v*100:.1f}%)")

        print("\nHasil Penyaringan Nucleus Top-p:")
        for p in p_thresholds:
            filtered = top_p_filtering(current_logits, p=p)
            renorm_probs = F.softmax(filtered, dim=-1)
            active_tokens = [vocab[i] for i in range(len(vocab)) if renorm_probs[i] > 0]
            num_active = len(active_tokens)
            print(f"  Ambang p = {p:4.2f} -> Ukuran Pool: {num_active} kata {active_tokens}")

    print("\n[KEUNGGULAN UTAMA NUCLEUS TOP-P]")
    print("1. Adaptif secara dinamis: Pada kasus yakin (confident), pool menyusut otomatis menjadi 1-2 kata saja.")
    print("2. Pada kasus ragu/kreatif (flat), pool melebar otomatis mencakup banyak alternatif yang masuk akal.")
    print("3. Standar industri modern (*de facto standard*) pada LLM komersial sering menggabungkan Temperature + Top-p.")


if __name__ == "__main__":
    main()

