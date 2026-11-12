"""
Chapter 23 - Deep Learning with PyTorch
Contoh 4: Arsitektur Model Menggunakan torch.nn.Module

Skrip ini mendemonstrasikan:
1. Pembuatan kelas neural network turunan torch.nn.Module.
2. Definisi layer di __init__ dan aliran komputasi di forward().
3. Inspeksi parameter terbobot (named_parameters) dan penghitungan parameter trainable.
4. Uji forward pass dengan tensor batch masukan.
5. Prinsip krusial: mengapa model mengeluarkan logits mentah tanpa Softmax
   saat dipasangkan dengan torch.nn.CrossEntropyLoss.
"""

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """
    Multilayer Perceptron modular untuk klasifikasi biner/multikelas.
    Arsitektur: in_features -> hidden_1 (ReLU) -> hidden_2 (ReLU) -> num_classes (Logits).
    """
    def __init__(self, in_features=6, hidden_1=16, hidden_2=8, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_2, num_classes)

    def forward(self, x):
        # Propagasi maju
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        # Output berupa LOGITS MENTAH (bukan probabilitas softmax)
        # Alasan: nn.CrossEntropyLoss di PyTorch mengintegrasikan LogSoftmax
        # dan NLLLoss secara simultan dengan trik subtract-max untuk stabilitas numerik.
        logits = self.out(x)
        return logits


class SimpleBaseline(nn.Module):
    """Model baseline linier lapis tunggal (setara Logistic Regression)."""
    def __init__(self, in_features=6, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


def inspect_model_parameters(model):
    """Menampilkan detail layer, ukuran matriks bobot, dan total parameter."""
    total_params = 0
    print(f"\nInspeksi Parameter Model: {model.__class__.__name__}")
    print("-" * 65)
    print(f"{'Nama Parameter':<25} | {'Shape':<18} | {'Jumlah Elemen':<12}")
    print("-" * 65)
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            total_params += num
            print(f"{name:<25} | {str(list(param.shape)):<18} | {num:<12}")
    print("-" * 65)
    print(f"Total Trainable Parameters: {total_params}")
    return total_params


def main():
    print("=" * 70)
    print("PYTORCH NN.MODULE ARCHITECTURE & PARAMETERS")
    print("=" * 70)

    # 1. Inisialisasi Model
    torch.manual_seed(42)
    mlp_model = TabularMLP(in_features=6, hidden_1=16, hidden_2=8, num_classes=2)
    baseline_model = SimpleBaseline(in_features=6, num_classes=2)

    # 2. Inspeksi Parameter
    total_mlp = inspect_model_parameters(mlp_model)
    total_baseline = inspect_model_parameters(baseline_model)

    # Verifikasi hitungan parameter manual MLP:
    # fc1: 6*16 + 16 = 112
    # fc2: 16*8 + 8  = 136
    # out: 8*2 + 2   = 18
    # Total: 112 + 136 + 18 = 266 parameter
    assert total_mlp == 266, f"Ekspektasi 266 parameter, diperoleh {total_mlp}"
    assert total_baseline == (6 * 2 + 2), f"Ekspektasi 14 parameter baseline"

    # 3. Uji Forward Pass dengan Mock Batch
    batch_size = 4
    mock_input = torch.randn((batch_size, 6))
    
    mlp_model.eval()
    with torch.no_grad():
        mlp_logits = mlp_model(mock_input)
        baseline_logits = baseline_model(mock_input)

    print(f"\n[Forward Pass Verification (Batch Size = {batch_size})]:")
    print(f"  Input shape:          {mock_input.shape}")
    print(f"  MLP Output shape:     {mlp_logits.shape} (Format: (Batch, Num_Classes))")
    print(f"  MLP Raw Logits:\n{mlp_logits}")
    print(f"  Baseline Output shape:{baseline_logits.shape}")

    assert mlp_logits.shape == (batch_size, 2)
    assert baseline_logits.shape == (batch_size, 2)

    # Menunjukkan cara mengonversi logits ke probabilitas jika diperlukan untuk inferensi
    probs = torch.softmax(mlp_logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    print(f"\n  Probabilitas Terhitung (Softmax):\n{probs}")
    print(f"  Prediksi Kelas (Argmax): {preds.tolist()}")
    assert probs.shape == (batch_size, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))

    print("\nKESIMPULAN: Implementasi nn.Module dan aliran tensor terverifikasi 100% konsisten.")


if __name__ == "__main__":
    main()

