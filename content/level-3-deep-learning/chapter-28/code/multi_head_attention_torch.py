"""
Chapter 28 - Attention & Transformer Architecture
Script 8: PyTorch nn.MultiheadAttention Deep Dive

Menganalisis arsitektur resmi PyTorch nn.MultiheadAttention:
- Membedah representasi gabungan bobot in_proj_weight: [W_q; W_k; W_v] berukuran (3*d_model, d_model).
- Menghitung total parameter analitis: 4 * d_model^2 + 4 * d_model.
- Menguji integrasi batch_first=True, key_padding_mask, dan attn_mask (causal look-ahead mask).
"""

import torch
import torch.nn as nn
from typing import Tuple


def main():
    print("=== Chapter 28: Demo PyTorch nn.MultiheadAttention ===")

    d_model = 16
    num_heads = 4
    batch_size = 2
    seq_len = 5

    # 1. Inisialisasi Modul PyTorch
    mha_torch = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
    )

    # 2. Analisis Struktur Parameter Internal
    # in_proj_weight menggabungkan W_Q, W_K, dan W_V menjadi satu matriks besar
    in_proj_w = mha_torch.in_proj_weight
    in_proj_b = mha_torch.in_proj_bias
    out_proj_w = mha_torch.out_proj.weight
    out_proj_b = mha_torch.out_proj.bias

    total_params = sum(p.numel() for p in mha_torch.parameters())
    expected_params = 4 * (d_model ** 2) + 4 * d_model

    print(f"Konfigurasi Model:")
    print(f"  embed_dim = {d_model}, num_heads = {num_heads}, head_dim = {d_model // num_heads}")
    print(f"  Bentuk in_proj_weight  : {in_proj_w.shape} (3*d_model x d_model)")
    print(f"  Bentuk in_proj_bias    : {in_proj_b.shape} (3*d_model)")
    print(f"  Bentuk out_proj.weight : {out_proj_w.shape} (d_model x d_model)")
    print(f"  Total Parameter Aktual : {total_params} parameter")
    print(f"  Total Parameter Teori  : {expected_params} parameter (4*d^2 + 4*d)")

    assert total_params == expected_params, "Jumlah parameter PyTorch MHA harus persis sesuai teori"

    # 3. Pengujian Forward Pass dengan Causal Mask dan Padding Mask
    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_len, d_model)

    # Causal mask: segitiga atas bernilai -inf (atau True jika boolean mask)
    # Pada PyTorch 2.0+, attn_mask bertipe float dengan nilai float('-inf') pada posisi terlarang
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    # Key padding mask: True pada posisi yang DIABAIKAN (<pad>)
    # Batch 0: semua valid; Batch 1: token terakhir adalah padding
    key_padding_mask = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, False, True],
    ], dtype=torch.bool)

    attn_output, attn_weights = mha_torch(
        query=X,
        key=X,
        value=X,
        key_padding_mask=key_padding_mask,
        attn_mask=causal_mask,
        need_weights=True,
        average_attn_weights=False,  # Mengembalikan bobot per head: (batch, num_heads, seq, seq)
    )

    print(f"\nBentuk Output Forward Pass:")
    print(f"  Tensor Output       : {attn_output.shape} (Batch x SeqLen x d_model)")
    print(f"  Tensor Bobot Atensi : {attn_weights.shape} (Batch x Heads x SeqLen x SeqLen)")

    # 4. Validasi Kausalitas dan Padding Mask pada Bobot Atensi
    # Bobot atensi pada segitiga atas (masa depan) harus bernilai 0.0
    weights_np = attn_weights.detach().numpy()
    b0_h0 = weights_np[0, 0]
    print("\nMatriks Atensi Head 0 (Batch 0, Causal Masked):")
    print(weights_np[0, 0].round(3))

    assert weights_np[0, 0, 0, 1] == 0.0, "Token 0 tidak boleh memperhatikan token masa depan 1"
    assert weights_np[1, 0, 0, 4] == 0.0, "Token padding pada posisi 4 harus menerima bobot 0.0"

    print("\nStatus: multi_head_attention_torch.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

