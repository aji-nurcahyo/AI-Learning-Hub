"""
Chapter 28 - Attention & Transformer Architecture
Script 12: Complete End-to-End Mini Transformer Architecture Demo

Membangun arsitektur terpadu Sequence-to-Sequence Mini Transformer (Encoder-Decoder) lengkap:
- Input & Output Embedding Layers + Sinusoidal Positional Encoding.
- Stack Lapisan Encoder (Self-Attention + Add & Norm + FFN + Add & Norm).
- Stack Lapisan Decoder (Masked Self-Attention + Cross-Attention + FFN + Add & Norm).
- Linear Projection Head ke ruang distribusi probabilitas kosakata target.
- Mengukur latensi komputasi inferensi maju pada CPU standar.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MiniTransformer(nn.Module):
    """Arsitektur Mini Transformer lengkap untuk demonstrasi pedagogis CPU."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Lapisan Embedding & Positional Encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)

        # 2. PyTorch Transformer Core (Encoder & Decoder)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Output Projection Head
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        src: (batch_size, src_len)
        tgt: (batch_size, tgt_len)
        Mengembalikan logits: (batch_size, tgt_len, tgt_vocab_size)
        """
        # Skalakan embedding dengan sqrt(d_model) sesuai paper Vaswani et al.
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.generator(out)
        return logits


def main():
    print("=== Chapter 28: Demo Complete Mini Transformer (Seq2Seq) ===")

    src_vocab_size = 50
    tgt_vocab_size = 60
    d_model = 32
    num_heads = 4
    num_layers = 2
    d_ff = 128
    batch_size = 2
    src_len = 8
    tgt_len = 6

    model = MiniTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Konfigurasi Arsitektur:")
    print(f"  Vocab Sumber: {src_vocab_size}, Vocab Target: {tgt_vocab_size}")
    print(f"  d_model = {d_model}, num_heads = {num_heads}, Layers = {num_layers} (Enc) / {num_layers} (Dec)")
    print(f"  Total Parameter Terpelajari: {total_params:,} parameter")

    # Data Sintetis Masukan
    torch.manual_seed(42)
    src_input = torch.randint(low=1, high=src_vocab_size, size=(batch_size, src_len))
    tgt_input = torch.randint(low=1, high=tgt_vocab_size, size=(batch_size, tgt_len))

    # Causal Look-Ahead Mask untuk Decoder
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)

    # Profiling Latensi Inferensi CPU
    model.eval()
    with torch.no_grad():
        # Warmup
        _ = model(src_input, tgt_input, tgt_mask=tgt_mask)

        start_time = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            logits = model(src_input, tgt_input, tgt_mask=tgt_mask)
        elapsed_ms = ((time.perf_counter() - start_time) / iterations) * 1000.0

    print(f"\nBentuk Input Source : {src_input.shape} (Batch x SrcLen)")
    print(f"Bentuk Input Target : {tgt_input.shape} (Batch x TgtLen)")
    print(f"Bentuk Logits Output: {logits.shape} (Batch x TgtLen x TgtVocabSize)")
    print(f"Latensi Inferensi CPU Rata-Rata: {elapsed_ms:.2f} ms per batch (Sangat Cepat & Responsif!)")

    assert logits.shape == (batch_size, tgt_len, tgt_vocab_size)
    print("\nStatus: transformer_architecture_demo.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

