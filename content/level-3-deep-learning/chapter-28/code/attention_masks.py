"""
Chapter 28 - Attention & Transformer Architecture
Script 6: Attention Masks (Padding Mask & Causal / Look-Ahead Mask)

Mengimplementasikan dua mekanisme masking fundamental dalam arsitektur Transformer:
1. Padding Mask: Menghindari alokasi atensi ke token pengisi (<pad>).
2. Causal / Look-Ahead Mask: Menjaga sifat autoregresif decoder dengan melarang
   token masa kini mengintip token masa depan (j > i).
Menjelaskan perbedaan konvensi boolean mask dan membuktikan probabilitas posisi ter-mask = 0.0000.
"""

import numpy as np
import torch
import torch.nn.functional as F


def create_padding_mask(seq_tokens: np.ndarray, pad_idx: int = 0) -> np.ndarray:
    """
    Menghasilkan boolean mask untuk padding:
    True untuk token valid, False untuk token <pad>.
    seq_tokens: (batch_size, seq_len)
    Mengembalikan: mask berdimensi (batch_size, 1, 1, seq_len) kompatibel broadcasting
    """
    # True jika BUKAN padding
    valid_mask = (seq_tokens != pad_idx)
    # Bentuk (batch_size, 1, 1, seq_len) untuk broadcasting ke (batch_size, num_heads, seq_len_q, seq_len_k)
    return valid_mask[:, np.newaxis, np.newaxis, :]


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Menghasilkan matriks segitiga bawah (Lower Triangular Boolean Mask):
    True untuk posisi masa lalu & kini (j <= i), False untuk masa depan (j > i).
    """
    # np.tril menghasilkan 1 pada diagonal dan segitiga bawah
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return causal_mask


def apply_mask_to_scores(scores: np.ndarray, mask: np.ndarray, mask_value: float = -1e9) -> np.ndarray:
    """Mengganti posisi bernilai False pada mask dengan bilangan negatif sangat besar."""
    return np.where(mask, scores, mask_value)


def main():
    print("=== Chapter 28: Demo Attention Masks (Padding & Causal) ===")

    # 1. Demonstrasi Causal Mask (Look-Ahead Mask pada Decoder)
    seq_len = 4
    causal_mask = create_causal_mask(seq_len)

    print(f"1. Matriks Causal Mask ({seq_len} x {seq_len}):")
    print(causal_mask.astype(int))

    # Simulasi skor atensi sembarang
    rng = np.random.default_rng(42)
    raw_scores = rng.standard_normal((seq_len, seq_len), dtype=np.float32)

    masked_scores = apply_mask_to_scores(raw_scores, causal_mask)
    # Hitung softmax stabil
    exp_s = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    causal_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    print("\nBobot Atensi Causal Setelah Softmax:")
    print(np.round(causal_weights, 4))

    # Verifikasi segitiga atas bernilai tepat 0.0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert causal_weights[i, j] == 0.0, f"Posisi masa depan ({i}, {j}) harus bernilai 0.0!"
    print("-> Sifat autoregresif terbukti: Token i tidak dapat melihat token masa depan j > i.")

    # 2. Demonstrasi Padding Mask
    # Batch 2 sekuens, sekuens pertama panjang 4, sekuens kedua memiliki 2 token padding <pad> (0)
    batch_tokens = np.array([
        [15, 23, 8, 4],   # 4 token valid
        [12, 9, 0, 0],    # 2 token valid, 2 token <pad>
    ], dtype=np.int32)

    pad_mask = create_padding_mask(batch_tokens, pad_idx=0)
    print(f"\n2. Bentuk Padding Mask: {pad_mask.shape}")
    print(f"   Mask Batch 0 (Tanpa Padding) : {pad_mask[0, 0, 0].tolist()}")
    print(f"   Mask Batch 1 (2 Padding)     : {pad_mask[1, 0, 0].tolist()}")

    # Uji atensi pada Batch 1
    scores_b1 = np.ones((1, 4, 4), dtype=np.float32)
    masked_b1 = apply_mask_to_scores(scores_b1, pad_mask[1])
    exp_b1 = np.exp(masked_b1 - np.max(masked_b1, axis=-1, keepdims=True))
    weights_b1 = exp_b1 / np.sum(exp_b1, axis=-1, keepdims=True)

    print("\nBobot Atensi Batch 1 (2 Token Terakhir adalah <pad>):")
    print(np.round(weights_b1[0], 4))

    # Dua kolom terakhir harus bernilai 0.0
    assert np.allclose(weights_b1[0, :, 2:], 0.0), "Token padding harus menerima bobot atensi 0.0!"
    print("-> Token <pad> berhasil diabaikan 100% dari alokasi atensi.")

    # 3. Catatan Konvensi Boolean PyTorch vs Standar
    print("\n[Catatan Kritis Konvensi PyTorch]:")
    print("  Pada PyTorch 'nn.MultiheadAttention(..., key_padding_mask)':")
    print("  - True menandakan posisi yang DIABAIKAN / DIMASK (kebalikan dari konvensi standar NumPy).")
    print("  - Selalu periksa dokumentasi API PyTorch untuk menghindari pembalikan mask!")

    print("\nStatus: attention_masks.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

