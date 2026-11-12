"""
Chapter 27 - Natural Language Processing Fundamentals
Script 9: Simple Recurrent Neural Network (Simple RNN) Forward Pass

Mengimplementasikan arsitektur Simple RNN (Elman RNN) dari dasar menggunakan NumPy:
    h_t = tanh(x_t W_{xh}^T + h_{t-1} W_{hh}^T + b_ih + b_hh)
serta memvalidasi kesetaraan numerik lintasan maju (forward pass) terhadap PyTorch nn.RNN.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class SimpleRNNFromScratch:
    """Implementasi forward pass Simple RNN murni berbasis NumPy."""

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Matriks bobot dan bias:
        # W_xh: bobot masukan ke hidden (hidden_size x input_size)
        # W_hh: bobot hidden rekuren ke hidden (hidden_size x hidden_size)
        self.W_xh = np.random.randn(hidden_size, input_size).astype(np.float32) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1
        self.b_ih = np.zeros(hidden_size, dtype=np.float32)
        self.b_hh = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x: np.ndarray, h_0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: array dimensi (batch_size, seq_len, input_size)
        h_0: hidden state awal dimensi (batch_size, hidden_size)
        Mengembalikan:
          - all_hidden_states: (batch_size, seq_len, hidden_size)
          - h_n: hidden state langkah waktu terakhir (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        if h_0 is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            h_prev = h_0.copy()

        all_h = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            # h_t = tanh(x_t @ W_xh.T + b_ih + h_{t-1} @ W_hh.T + b_hh)
            gate_in = np.dot(x_t, self.W_xh.T) + self.b_ih
            gate_rec = np.dot(h_prev, self.W_hh.T) + self.b_hh
            h_t = np.tanh(gate_in + gate_rec)
            
            all_h.append(h_t)
            h_prev = h_t

        all_hidden_states = np.stack(all_h, axis=1)  # (batch_size, seq_len, hidden_size)
        return all_hidden_states, h_prev


def main():
    print("=== Chapter 27: Demo Simple RNN Forward Pass ===")

    batch_size = 2
    seq_len = 4
    input_size = 3
    hidden_size = 5

    print(f"Konfigurasi Arsitektur:")
    print(f"  Batch Size={batch_size}, Panjang Sekuens (T)={seq_len}")
    print(f"  Dimensi Input (D)={input_size}, Dimensi Hidden (H)={hidden_size}")

    # Pembangkitan data sekuens acak deterministik
    rng = np.random.default_rng(42)
    x_numpy = rng.standard_normal((batch_size, seq_len, input_size), dtype=np.float32)

    # 1. Inisialisasi RNN Scratch
    rnn_scratch = SimpleRNNFromScratch(input_size, hidden_size)
    h_out_scratch, h_n_scratch = rnn_scratch.forward(x_numpy)

    print(f"\nBentuk Output NumPy Scratch:")
    print(f"  All Hidden States: {h_out_scratch.shape} (Batch x Seq x Hidden)")
    print(f"  Final State h_n  : {h_n_scratch.shape} (Batch x Hidden)")

    # 2. Inisialisasi PyTorch nn.RNN dengan Bobot yang Disinkronisasi
    rnn_torch = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, nonlinearity="tanh")
    
    with torch.no_grad():
        rnn_torch.weight_ih_l0.copy_(torch.from_numpy(rnn_scratch.W_xh))
        rnn_torch.weight_hh_l0.copy_(torch.from_numpy(rnn_scratch.W_hh))
        rnn_torch.bias_ih_l0.copy_(torch.from_numpy(rnn_scratch.b_ih))
        rnn_torch.bias_hh_l0.copy_(torch.from_numpy(rnn_scratch.b_hh))

    x_torch = torch.from_numpy(x_numpy)
    h_out_torch, h_n_torch = rnn_torch(x_torch)

    h_out_torch_np = h_out_torch.detach().numpy()
    h_n_torch_np = h_n_torch.squeeze(0).detach().numpy()

    # 3. Verifikasi Deviasi Numerik
    max_dev_all = np.max(np.abs(h_out_scratch - h_out_torch_np))
    max_dev_final = np.max(np.abs(h_n_scratch - h_n_torch_np))

    print(f"\nDeviasi Numerik Terhadap PyTorch nn.RNN:")
    print(f"  Max Deviasi Seluruh Langkah Waktu : {max_dev_all:.2e}")
    print(f"  Max Deviasi Hidden State Terakhir : {max_dev_final:.2e}")

    assert max_dev_all < 1e-6, "Lintasan maju RNN Scratch harus identik dengan PyTorch"
    print("Status: Kesetaraan numerik Simple RNN terbukti 100% IDENTIK!")

    print("\nStatus: simple_rnn.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

