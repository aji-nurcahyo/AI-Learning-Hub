"""
Chapter 28 - Attention & Transformer Architecture
Script 11: Transformer Decoder Block & Autoregressive Gating

Membangun satu blok utuh Transformer Decoder (Vaswani et al. 2017) yang memadukan 3 sub-lapisan:
1. Masked Multi-Head Self-Attention (dengan causal mask masa depan).
2. Multi-Head Cross-Attention (Query dari decoder, Key & Value dari encoder memory).
3. Position-wise Feed-Forward Network (FFN).
Setiap sub-lapisan dilengkapi sambungan sisa (Residual Connection) dan Layer Normalization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class PositionWiseFeedForward(nn.Module):
    """Lapisan Feed-Forward: Linear -> ReLU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class TransformerDecoderBlock(nn.Module):
    """Satu blok decoder standar Post-LN Transformer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1. Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Cross-Attention ke memori Encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # 3. Position-wise FFN
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: Sekuens masukan decoder (batch_size, seq_len_tgt, d_model)
        memory: Luaran dari encoder stack (batch_size, seq_len_src, d_model)
        """
        # Sub-layer 1: Masked Self-Attention
        self_attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Sub-layer 2: Cross-Attention (Q dari x, K & V dari memory)
        cross_attn_out, _ = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Sub-layer 3: FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


def main():
    print("=== Chapter 28: Demo Transformer Decoder Block ===")

    batch_size = 2
    seq_len_tgt = 4  # Sekuens target decoder
    seq_len_src = 6  # Sekuens sumber encoder memory
    d_model = 16
    num_heads = 4
    d_ff = 64

    decoder_block = TransformerDecoderBlock(
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0
    )

    # Analisis Parameter Analitis
    # Self-Attn MHA  : 4 * d^2 + 4 * d = 1088
    # LayerNorm 1    : 2 * d = 32
    # Cross-Attn MHA : 4 * d^2 + 4 * d = 1088
    # LayerNorm 2    : 2 * d = 32
    # FFN            : 2128
    # LayerNorm 3    : 2 * d = 32
    # Total Teori    = 1088 + 32 + 1088 + 32 + 2128 + 32 = 4400 parameter
    actual_params = sum(p.numel() for p in decoder_block.parameters())

    print(f"Konfigurasi Blok Decoder:")
    print(f"  d_model = {d_model}, num_heads = {num_heads}, d_ff = {d_ff}")
    print(f"  Total Parameter Terpelajari: {actual_params} parameter")
    assert actual_params == 4400, f"Jumlah parameter blok decoder harus tepat 4400 (aktual: {actual_params})"

    # Simulasi Forward Pass dengan Causal Mask
    torch.manual_seed(42)
    x_tgt = torch.randn(batch_size, seq_len_tgt, d_model)
    encoder_memory = torch.randn(batch_size, seq_len_src, d_model)

    # Causal mask untuk target
    causal_mask = torch.triu(torch.full((seq_len_tgt, seq_len_tgt), float("-inf")), diagonal=1)

    dec_output = decoder_block(x=x_tgt, memory=encoder_memory, tgt_mask=causal_mask)

    print(f"\nBentuk Tensor Input Target Decoder : {x_tgt.shape} (Batch x SeqTgt x d_model)")
    print(f"Bentuk Tensor Memory dari Encoder  : {encoder_memory.shape} (Batch x SeqSrc x d_model)")
    print(f"Bentuk Tensor Output Decoder Akhir : {dec_output.shape} (Batch x SeqTgt x d_model)")

    assert dec_output.shape == (batch_size, seq_len_tgt, d_model)
    print("\nStatus: transformer_decoder.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

