"""
Chapter 24 - Convolutional Neural Networks
Contoh 3: Arsitektur Model CNN (FashionCNN) vs Baseline (FlattenMLP)

Skrip ini mendemonstrasikan:
1. Perancangan model Convolutional Neural Network modular menggunakan torch.nn.Module.
2. Komposisi blok Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
3. Transisi dari feature maps 2D ke representasi 1D menggunakan Flatten sebelum fully connected layer.
4. Model pembanding FlattenMLP (arsitektur fully-connected klasik pada piksel 1D).
5. Komparasi parameter count: membuktikan efisiensi parameter sharing CNN vs ledakan parameter MLP.
"""

import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """
    Arsitektur CNN sederhana untuk klasifikasi citra greyscale 28x28 (10 kelas).
    Blok 1: Conv(1->16, 3x3) -> BN -> ReLU -> MaxPool(2x2) [Output: 16 x 14 x 14]
    Blok 2: Conv(16->32, 3x3) -> BN -> ReLU -> MaxPool(2x2) [Output: 32 x 7 x 7]
    Classifier: Flatten -> Linear(1568, 64) -> ReLU -> Linear(64, 10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Blok Konvolusi 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Blok Konvolusi 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Lapisan Klasifikasi (Fully Connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        logits = self.classifier(x)
        return logits


class FlattenMLPBaseline(nn.Module):
    """
    Model pembanding: Multilayer Perceptron yang meratakan seluruh piksel gambar (784 fitur).
    Arsitektur: Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def inspect_architecture(model, input_shape=(1, 1, 28, 28)):
    """Menampilkan detail layer, ukuran output, dan jumlah parameter per layer."""
    print(f"\n=================================================================")
    print(f"DETAIL ARSITEKTUR: {model.__class__.__name__}")
    print(f"=================================================================")
    print(f"{'Nama Lapisan / Komponen':<32} | {'Shape Parameter':<20} | {'Elemen':<10}")
    print("-" * 68)

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            total_params += num
            print(f"{name:<32} | {str(list(param.shape)):<20} | {num:<10}")

    print("-" * 68)
    print(f"TOTAL PARAMETER TRAINABLE: {total_params:,}")

    # Forward pass uji bentuk
    model.eval()
    with torch.no_grad():
        mock_x = torch.zeros(input_shape)
        out_logits = model(mock_x)
    print(f"Uji Forward Pass Input {input_shape} -> Logits Output: {list(out_logits.shape)}")
    assert out_logits.shape == (input_shape[0], 10), "Logits shape tidak sesuai!"
    return total_params


def main():
    print("=" * 70)
    print("CNN ARCHITECTURE VS FLATTEN MLP BASELINE")
    print("=" * 70)

    # 1. Inisialisasi Model
    cnn_model = FashionCNN(num_classes=10)
    mlp_model = FlattenMLPBaseline(num_classes=10)

    # 2. Inspeksi Arsitektur & Parameter
    params_cnn = inspect_architecture(cnn_model)
    params_mlp = inspect_architecture(mlp_model)

    # 3. Analisis Komparasi Bobot & Parameter Sharing
    print("\n[ANALISIS KOMPARATIF PARAMETER SHARING]")
    print("-" * 68)
    print(f"  FashionCNN         : {params_cnn:,} total parameter")
    print(f"  FlattenMLPBaseline : {params_mlp:,} total parameter")
    print(f"  Rasio Efisiensi    : CNN menghemat {(1.0 - params_cnn/params_mlp)*100:.1f}% parameter!")

    # Perhatikan layer pertama:
    # Pada MLP: Linear(784, 128) membutuhkan 784 * 128 + 128 = 100,480 parameter (92% dari total model!)
    # Pada CNN: Conv(1, 16, 3x3) HANYA membutuhkan (1 * 3 * 3 + 1) * 16 = 160 parameter!
    conv1_params = (1 * 3 * 3 + 1) * 16
    fc1_mlp_params = 784 * 128 + 128
    print(f"\n  Lapisan Pertama MLP: {fc1_mlp_params:,} parameter (Sangat boros, tidak peka translasi)")
    print(f"  Lapisan Pertama CNN: {conv1_params} parameter (Weight sharing & local receptive field)")

    assert params_cnn < params_mlp, "CNN seharusnya lebih hemat parameter daripada MLP pada input citra"
    print("\nKESIMPULAN: Arsitektur CNN terbukti menjaga korelasi spasial 2D dengan parameter sharing efisien.")


if __name__ == "__main__":
    main()

