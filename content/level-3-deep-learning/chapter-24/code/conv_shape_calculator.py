"""
Chapter 24 - Convolutional Neural Networks
Contoh 2: Kalkulator Dimensi Output dan Parameter Konvolusi 2D

Skrip ini mendemonstrasikan:
1. Penurunan formula output shape spasial konvolusi:
   Kasus Khusus (Dilation D = 1):
     H_out = floor((H_in + 2*P - K) / S) + 1
   Bentuk Umum (Dilation D >= 1):
     H_out = floor((H_in + 2*P - D*(K - 1) - 1) / S) + 1
2. Perhitungan jumlah parameter terbobot: (K_h * K_w * C_in + 1) * C_out.
3. Eksperimen berbagai varian padding: Valid Padding, Same Padding, Strided Conv, Dilated Conv.
4. Verifikasi komputasi matematis terhadap layer aktual torch.nn.Conv2d dan torch.nn.MaxPool2d.
"""

import math
import torch
import torch.nn as nn


def compute_conv_output_dim(dim_in, kernel_size, padding=0, stride=1, dilation=1):
    """
    Menghitung dimensi luaran spasial 1D (Tinggi atau Lebar).
    Formula umum: floor((dim_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1
    """
    effective_k = dilation * (kernel_size - 1) + 1
    dim_out = math.floor((dim_in + 2 * padding - effective_k) / stride) + 1
    return dim_out


def compute_conv_params(c_in, c_out, kernel_size, bias=True):
    """
    Menghitung total parameter bobot dan bias layer Conv2D.
    Total = (K_h * K_w * C_in + (1 if bias else 0)) * C_out
    """
    k_h = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    k_w = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
    weights_per_filter = k_h * k_w * c_in
    bias_per_filter = 1 if bias else 0
    return (weights_per_filter + bias_per_filter) * c_out


def verify_conv_configuration(name, in_shape, c_out, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    """Menguji kesesuaian formula teoritis terhadap PyTorch nn.Conv2d aktual."""
    N, C_in, H_in, W_in = in_shape

    # 1. Perhitungan Teoritis
    H_expected = compute_conv_output_dim(H_in, kernel_size, padding, stride, dilation)
    W_expected = compute_conv_output_dim(W_in, kernel_size, padding, stride, dilation)
    params_expected = compute_conv_params(C_in, c_out, kernel_size, bias)

    # 2. Instansiasi PyTorch nn.Conv2d
    layer = nn.Conv2d(
        in_channels=C_in,
        out_channels=c_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    )
    mock_input = torch.zeros(in_shape)
    actual_output = layer(mock_input)
    actual_params = sum(p.numel() for p in layer.parameters())

    print(f"\n--- Kasus: {name} ---")
    print(f"  Input Shape:       {in_shape} (NCHW)")
    print(f"  Konfigurasi:       K={kernel_size}, S={stride}, P={padding}, D={dilation}, C_out={c_out}")
    print(f"  Output Shape:      Teoritis ({N}, {c_out}, {H_expected}, {W_expected}) | Aktual {list(actual_output.shape)}")
    print(f"  Jumlah Parameter:  Teoritis {params_expected} | Aktual {actual_params}")

    assert actual_output.shape == (N, c_out, H_expected, W_expected), "Shape mismatch!"
    assert actual_params == params_expected, "Parameter mismatch!"
    print("  Status Verifikasi: VALID (Cocok 100%)")


def main():
    print("=" * 70)
    print("CONVOLUTIONAL SHAPE & PARAMETER CALCULATOR")
    print("=" * 70)

    # Kasus 1: Valid Padding (Tanpa padding, ukuran menyusut)
    # H_out = (28 + 2*0 - 3)/1 + 1 = 26
    verify_conv_configuration(
        name="Valid Padding (No Padding)",
        in_shape=(1, 1, 28, 28),
        c_out=16,
        kernel_size=3,
        stride=1,
        padding=0
    )

    # Kasus 2: Same Padding (Ukuran spasial tetap utuh)
    # H_out = (28 + 2*1 - 3)/1 + 1 = 28
    verify_conv_configuration(
        name="Same Padding (Preserve Resolution)",
        in_shape=(1, 1, 28, 28),
        c_out=32,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # Kasus 3: Strided Convolution (Reduksi resolusi 2x)
    # H_out = floor((28 + 2*1 - 3)/2) + 1 = floor(27/2) + 1 = 13 + 1 = 14
    verify_conv_configuration(
        name="Strided Convolution (Downsampling S=2)",
        in_shape=(1, 32, 28, 28),
        c_out=64,
        kernel_size=3,
        stride=2,
        padding=1
    )

    # Kasus 4: Dilated Convolution (Atrous Conv, Dilation D=2)
    # Effective K = 2*(3-1) + 1 = 5
    # H_out = floor((28 + 2*2 - 5)/1) + 1 = 28
    verify_conv_configuration(
        name="Dilated Convolution (Field of View Luas D=2)",
        in_shape=(1, 64, 28, 28),
        c_out=64,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2
    )

    # Kasus 5: Max Pooling 2x2 Stride 2
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    mock_pool_in = torch.zeros((1, 32, 28, 28))
    pool_out = pool(mock_pool_in)
    print(f"\n--- Kasus: Max Pooling 2x2 Stride 2 ---")
    print(f"  Input Shape:  {mock_pool_in.shape}")
    print(f"  Output Shape: {pool_out.shape} (Resolusi terpangkas tepat separuh: 28 -> 14)")
    print(f"  Parameter:    0 (Pooling tidak memiliki bobot yang dapat dipelajari)")
    assert pool_out.shape == (1, 32, 14, 14)

    print("\nKESIMPULAN: Seluruh formula output shape dan penghitungan parameter konvolusi terverifikasi 100%.")


if __name__ == "__main__":
    main()

