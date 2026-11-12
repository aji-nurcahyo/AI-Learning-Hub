"""
Chapter 28 - Attention & Transformer Architecture
Script 10: Transformer Encoder Block & Layer Normalization

Membangun satu blok utuh Transformer Encoder sesuai rancangan Vaswani et al. (2017):
1. Multi-Head Self-Attention.
2. Residual Connection & Layer Normalization (Add & Norm).
3. Position-wise Feed-Forward Network (FFN) dengan dimensi d_ff = 4 * d_model.
4. Residual Connection & Layer Normalization kedua.
Memvalidasi bentuk tensor masukan dan luaran serta menghitung total parameter terpelajari.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class PositionWiseFeedForward(nn.Module):
    """Lapisan Feed-Forward dua tahap: Linear(d_model -> d_ff) -> ReLU -> Linear(d_ff -> d_model)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class TransformerEncoderBlock(nn.Module):
    """Satu blok encoder standar Post-LN Transformer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        # Sub-layer 1: Multi-Head Self-Attention + Residual Connection + LayerNorm
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: Feed-Forward Network + Residual Connection + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


def main():
    print("=== Chapter 28: Demo Transformer Encoder Block ===")

    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    d_ff = 64  # 4 * d_model

    encoder_block = TransformerEncoderBlock(
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0
    )

    # Analisis Parameter Analitis
    # MHA: 4 * d^2 + 4 * d = 4 * 256 + 64 = 1088
    # LayerNorm 1: 2 * d = 32
    # FFN: (d * d_ff + d_ff) + (d_ff * d + d) = (1024 + 64) + (1024 + 16) = 2128
    # LayerNorm 2: 2 * d = 32
    # Total Teori = 1088 + 32 + 2128 + 32 = 3280 parameter
    actual_params = sum(p.numel() for p in encoder_block.parameters())

    print(f"Konfigurasi Blok Encoder:")
    print(f"  d_model = {d_model}, num_heads = {num_heads}, d_ff = {d_ff}")
    print(f"  Total Parameter Terpelajari: {actual_params} parameter")
    assert actual_params == 3280, f"Jumlah parameter blok encoder harus tepat 3280 (aktual: {actual_params})"

    # Simulasi Forward Pass
    torch.manual_seed(42)
    x_input = torch.randn(batch_size, seq_len, d_model)
    x_encoded = encoder_block(x_input)

    print(f"\nBentuk Tensor Input   : {x_input.shape} (Batch x SeqLen x d_model)")
    print(f"Bentuk Tensor Output  : {x_encoded.shape} (Identik preservasi dimensi)")

    # Periksa rata-rata dan variansi setelah Layer Normalization terakhir
    mean_val = x_encoded.mean(dim=-1).detach().numpy()
    std_val = x_encoded.std(dim=-1).detach().numpy()

    print(f"\nVerifikasi Layer Normalization (Batch 0, Seluruh Posisi Token):")
    print(f"  Rata-rata elemen (Ekspektasi ~0.0) : {mean_val[0].round(4)}")
    print(f"  Standar deviasi  (Ekspektasi ~1.0) : {std_val[0].round(4)}")

    assert x_encoded.shape == x_input.shape, "Dimensi output encoder harus identik dengan input"
    assert np.allclose(mean_val, 0.0, atol=1e-5), "Rata-rata setelah LayerNorm harus ~0.0"

    print("\nStatus: transformer_encoder.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()
