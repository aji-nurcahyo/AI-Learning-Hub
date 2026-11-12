"""
Chapter 27 - Natural Language Processing Fundamentals
Script 11: GRU (Gated Recurrent Unit) Sequence Model

Membangun arsitektur Gated Recurrent Unit (GRU) modular menggunakan PyTorch:
menjelaskan eliminasi Cell State terpisah dan penyederhanaan menjadi Update Gate (z_t)
serta Reset Gate (r_t), membuktikan efisiensi parameter ~25% lebih hemat dibandingkan LSTM,
serta menguji forward pass model klasifikasi sekuensial pada CPU.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TextClassifierGRU(nn.Module):
    """Model klasifikasi teks sekuensial berbasis GRU."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_ids)
        out, h_n = self.gru(embedded)
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        return logits, last_hidden


def calculate_gru_parameters(input_size: int, hidden_size: int, has_bias: bool = True) -> int:
    """
    Menghitung jumlah parameter lapisan GRU secara analitis:
    Terdapat 3 gerbang (Update, Reset, Candidate):
        Params = 3 * (input_size * hidden_size + hidden_size^2 + (2 * hidden_size if bias))
    """
    bias_count = 2 * hidden_size if has_bias else 0
    return 3 * (input_size * hidden_size + hidden_size * hidden_size + bias_count)


def main():
    print("=== Chapter 27: Demo GRU Architecture & Parameter Efficiency ===")

    vocab_size = 50
    embedding_dim = 16
    hidden_dim = 32
    num_classes = 3
    batch_size = 4
    seq_len = 8

    # 1. Perhitungan Parameter Analitis vs PyTorch
    analytical_gru_params = calculate_gru_parameters(embedding_dim, hidden_dim, has_bias=True)

    model = TextClassifierGRU(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )

    actual_gru_params = sum(p.numel() for p in model.gru.parameters())
    total_model_params = sum(p.numel() for p in model.parameters())

    print(f"Konfigurasi Model GRU:")
    print(f"  Vocab Size={vocab_size}, Embedding Dim={embedding_dim}, Hidden Dim={hidden_dim}")
    print(f"  Analitis Parameter Lapisan GRU  : {analytical_gru_params}")
    print(f"  Aktual Parameter Lapisan GRU    : {actual_gru_params}")
    print(f"  Total Parameter Keseluruhan Model: {total_model_params}")

    assert analytical_gru_params == actual_gru_params, "Jumlah parameter GRU harus persis cocok"

    # Perbandingan rasio efisiensi parameter GRU vs LSTM
    lstm_gru_ratio = actual_gru_params / 6400  # 6400 adalah param LSTM dari script 10
    print(f"\nRasio Parameter GRU terhadap LSTM: {actual_gru_params} / 6400 = {lstm_gru_ratio:.4f} (~75.0%)")
    print("-> Terbukti GRU menghemat ~25% parameter komputasi dibandingkan LSTM pada dimensi seimbang!")

    # 2. Forward Pass Simulation
    dummy_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    logits, last_hidden = model(dummy_input)

    print(f"\nBentuk Tensor Input & Output:")
    print(f"  Input Tokens Tensor : {dummy_input.shape} (Batch x SeqLen)")
    print(f"  Last Hidden State   : {last_hidden.shape} (Batch x HiddenDim)")
    print(f"  Logits Output       : {logits.shape} (Batch x NumClasses)")

    assert logits.shape == (batch_size, num_classes)
    assert last_hidden.shape == (batch_size, hidden_dim)

    print("\nStatus: gru_sequence_model.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

