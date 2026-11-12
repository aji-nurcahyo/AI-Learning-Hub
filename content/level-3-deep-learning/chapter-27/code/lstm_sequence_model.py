"""
Chapter 27 - Natural Language Processing Fundamentals
Script 10: LSTM (Long Short-Term Memory) Sequence Model

Membangun arsitektur jaringan Long Short-Term Memory (LSTM) modular menggunakan PyTorch:
membedah peran matematis Cell State (c_t), Hidden State (h_t), serta tiga gerbang utama
(Forget Gate f_t, Input Gate i_t, dan Output Gate o_t), menghitung jumlah parameter analitis,
serta menguji forward pass model klasifikasi sekuensial (Sequence-to-One).
"""

import torch
import torch.nn as nn
from typing import Tuple


class TextClassifierLSTM(nn.Module):
    """Model klasifikasi sekuens berbasis lapisan Embedding, LSTM, dan Linear Classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: Tensor indeks token berdimensi (batch_size, seq_len)
        Mengembalikan:
          - logits: Nilai prediksi mentah berdimensi (batch_size, num_classes)
          - last_hidden: Hidden state representasi sekuens (batch_size, hidden_dim * directions)
        """
        # 1. Lookup embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # 2. LSTM forward pass:
        # out: (batch_size, seq_len, hidden_dim * directions)
        # h_n: (num_layers * directions, batch_size, hidden_dim)
        # c_n: (num_layers * directions, batch_size, hidden_dim)
        out, (h_n, c_n) = self.lstm(embedded)

        # Ambil hidden state dari langkah waktu terakhir
        if self.num_directions == 1:
            last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        else:
            # Gabungkan arah forward dan backward jika bidirectional
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=-1)

        # 3. Proyeksi klasifikasi linier
        logits = self.classifier(last_hidden)
        return logits, last_hidden


def calculate_lstm_parameters(input_size: int, hidden_size: int, has_bias: bool = True) -> int:
    """
    Menghitung jumlah parameter lapisan LSTM secara analitis:
    Terdapat 4 gerbang (Forget, Input, Candidate, Output):
        Params = 4 * (input_size * hidden_size + hidden_size^2 + (2 * hidden_size if bias))
    """
    bias_count = 2 * hidden_size if has_bias else 0
    return 4 * (input_size * hidden_size + hidden_size * hidden_size + bias_count)


def main():
    print("=== Chapter 27: Demo LSTM Architecture & Parameterization ===")

    vocab_size = 50
    embedding_dim = 16
    hidden_dim = 32
    num_classes = 3
    batch_size = 4
    seq_len = 8

    # 1. Perhitungan Parameter Analitis vs PyTorch
    analytical_lstm_params = calculate_lstm_parameters(embedding_dim, hidden_dim, has_bias=True)
    
    model = TextClassifierLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        bidirectional=False,
    )

    actual_lstm_params = sum(p.numel() for p in model.lstm.parameters())
    total_model_params = sum(p.numel() for p in model.parameters())

    print(f"Konfigurasi Model LSTM:")
    print(f"  Vocab Size={vocab_size}, Embedding Dim={embedding_dim}, Hidden Dim={hidden_dim}")
    print(f"  Analitis Parameter Lapisan LSTM : {analytical_lstm_params}")
    print(f"  Aktual Parameter Lapisan LSTM   : {actual_lstm_params}")
    print(f"  Total Parameter Keseluruhan Model: {total_model_params}")

    assert analytical_lstm_params == actual_lstm_params, "Jumlah parameter LSTM harus persis cocok"

    # 2. Forward Pass Simulation
    dummy_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    logits, last_hidden = model(dummy_input)

    print(f"\nBentuk Tensor Input & Output:")
    print(f"  Input Tokens Tensor : {dummy_input.shape} (Batch x SeqLen)")
    print(f"  Last Hidden State   : {last_hidden.shape} (Batch x HiddenDim)")
    print(f"  Logits Output       : {logits.shape} (Batch x NumClasses)")

    # 3. Softmax Probabilitas Kelas
    probs = torch.softmax(logits, dim=-1)
    print("\nContoh Distribusi Probabilitas Batch Pertama:")
    print(f"  P(Kelas 0, 1, 2) = {probs[0].detach().numpy().round(4)}")

    assert logits.shape == (batch_size, num_classes)
    assert last_hidden.shape == (batch_size, hidden_dim)

    print("\nStatus: lstm_sequence_model.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

