"""
Chapter 30 - Deep Learning Capstone
Script 3: Deep Neural Network Architecture (PyTorch)
Rancang bangun arsitektur jaringan saraf mendalam (Deep Neural Network) dengan
koneksi residu (residual connection), Layer Normalization, dan Dropout regulasi.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ResidualBlock(nn.Module):
    """Blok Residu MLP: Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> Skip Connection."""
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out


class DeepPredictiveMaintenanceNet(nn.Module):
    """
    Arsitektur Deep Learning Capstone:
    Input (4 fitur) -> Proyeksi Awal -> 2x Residual Blocks -> Classifier Head -> Logits (3 kelas).
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 64, num_classes: int = 3, dropout_rate: float = 0.15):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Proyeksi Masukan
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Tumpukan Blok Residu
        self.res_block1 = ResidualBlock(hidden_dim, dropout_rate)
        self.res_block2 = ResidualBlock(hidden_dim, dropout_rate)

        # Lapisan Klasifikasi Akhir (Menghasilkan Logits Mentah)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        feat = self.res_block1(feat)
        feat = self.res_block2(feat)
        logits = self.classifier_head(feat)
        return logits


def main():
    print("=" * 70)
    print("CHAPTER 30 CAPSTONE: DEEP NEURAL NETWORK ARCHITECTURE")
    print("=" * 70)

    model = DeepPredictiveMaintenanceNet(in_features=4, hidden_dim=64, num_classes=3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Nama Arsitektur       : DeepPredictiveMaintenanceNet")
    print(f"Fitur Masukan         : {model.in_features}")
    print(f"Kelas Keluaran (Logits): {model.num_classes}")
    print(f"Total Parameter Bobot : {total_params:,} parameter")
    print(f"Parameter Terlatih    : {trainable_params:,} parameter")

    # Uji verifikasi dimensi tensor lintasan maju (forward pass)
    batch_size = 8
    dummy_input = torch.randn(batch_size, 4)
    logits_out = model(dummy_input)

    print(f"\n[VERIFIKASI TENSOR LINTASAN MAJU]")
    print(f"Bentuk Tensor Masukan : {dummy_input.shape} (Batch=8, Fitur=4)")
    print(f"Bentuk Tensor Keluaran: {logits_out.shape} (Batch=8, Logits=3)")

    assert logits_out.shape == (batch_size, 3), "Dimensi keluaran tidak sesuai!"
    print("✓ Dimensi tensor terverifikasi cocok secara matematis.")


if __name__ == "__main__":
    main()

