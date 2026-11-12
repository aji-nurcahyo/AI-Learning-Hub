"""
Chapter 24 - Convolutional Neural Networks
Contoh 1: Manual 2D Convolution (Cross-Correlation) vs torch.nn.functional.conv2d

Skrip ini mendemonstrasikan:
1. Perhitungan matematis konvolusi 2D (korelasi silang) dari dasar (scratch loop).
2. Pergeseran sliding window kernel pada matriks masukan.
3. Klarifikasi ilmiah: perbedaan konvolusi matematis formal (kernel dibalik 180 derajat)
   dan konvolusi pada deep learning (cross-correlation tanpa pembalikan kernel).
4. Verifikasi luaran manual terhadap fungsi PyTorch resmi (torch.nn.functional.conv2d).
5. Konvolusi multi-channel masukan dan multi-kernel luaran.
"""

import torch
import torch.nn.functional as F
import numpy as np


def manual_conv2d_single_channel(input_matrix, kernel, bias=0.0, stride=1, padding=0):
    """
    Operasi 2D Cross-Correlation satu channel manual.
    input_matrix: Tensor 2D (H_in, W_in)
    kernel: Tensor 2D (K_h, K_w)
    bias: Skalar float
    """
    H_in, W_in = input_matrix.shape
    K_h, K_w = kernel.shape

    # 1. Terapkan Padding manual jika P > 0
    if padding > 0:
        padded = torch.zeros((H_in + 2 * padding, W_in + 2 * padding), dtype=input_matrix.dtype)
        padded[padding:padding + H_in, padding:padding + W_in] = input_matrix
    else:
        padded = input_matrix

    H_pad, W_pad = padded.shape
    H_out = (H_pad - K_h) // stride + 1
    W_out = (W_pad - K_w) // stride + 1

    output = torch.zeros((H_out, W_out), dtype=input_matrix.dtype)

    # 2. Sliding window kernel melintasi seluruh lokasi spasial
    for i in range(H_out):
        for j in range(W_out):
            r_start = i * stride
            r_end = r_start + K_h
            c_start = j * stride
            c_end = c_start + K_w

            # Ekstrak patch lokal (local receptive field)
            patch = padded[r_start:r_end, c_start:c_end]
            # Perkalian Hadamard dan akumulasi skalar
            val = (patch * kernel).sum() + bias
            output[i, j] = val

    return output


def manual_conv2d_multichannel(input_tensor, kernel_tensor, bias_tensor=None, stride=1, padding=0):
    """
    Konvolusi 2D Multi-Channel & Multi-Kernel:
    input_tensor: (Batch, C_in, H_in, W_in)
    kernel_tensor: (C_out, C_in, K_h, K_w)
    bias_tensor: (C_out,)
    """
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, K_Cin, K_h, K_w = kernel_tensor.shape
    assert C_in == K_Cin, "Channel masukan kernel harus cocok dengan channel input"

    if bias_tensor is None:
        bias_tensor = torch.zeros(C_out)

    H_out = (H_in + 2 * padding - K_h) // stride + 1
    W_out = (W_in + 2 * padding - K_w) // stride + 1

    output = torch.zeros((N, C_out, H_out, W_out), dtype=input_tensor.dtype)

    for b in range(N):
        for out_c in range(C_out):
            channel_sum = torch.zeros((H_out, W_out), dtype=input_tensor.dtype)
            for in_c in range(C_in):
                in_slice = input_tensor[b, in_c]
                k_slice = kernel_tensor[out_c, in_c]
                channel_sum += manual_conv2d_single_channel(in_slice, k_slice, bias=0.0, stride=stride, padding=padding)
            output[b, out_c] = channel_sum + bias_tensor[out_c]

    return output


def main():
    print("=" * 70)
    print("MANUAL CONVOLUTION (CROSS-CORRELATION) VS PYTORCH F.CONV2D")
    print("=" * 70)

    # 1. Eksperimen Saluran Tunggal (Single Channel)
    input_2d = torch.tensor([
        [1.0, 2.0, 3.0, 0.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 1.0],
        [3.0, 0.0, 1.0, 2.0, 0.0],
        [2.0, 1.0, 0.0, 1.0, 2.0],
        [1.0, 0.0, 2.0, 1.0, 0.0]
    ], dtype=torch.float32)  # Shape (5, 5)

    # Kernel Sobel-like pendeteksi tepi vertikal
    kernel_2d = torch.tensor([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype=torch.float32)  # Shape (3, 3)

    bias_val = 0.5

    # Hitung Manual
    manual_res = manual_conv2d_single_channel(input_2d, kernel_2d, bias=bias_val, stride=1, padding=0)

    # Hitung via PyTorch Resmi F.conv2d (memerlukan tensor 4D NCHW)
    pt_input = input_2d.view(1, 1, 5, 5)
    pt_kernel = kernel_2d.view(1, 1, 3, 3)
    pt_bias = torch.tensor([bias_val], dtype=torch.float32)
    pytorch_res = F.conv2d(pt_input, pt_kernel, pt_bias, stride=1, padding=0).squeeze()

    print("\n[1] Hasil Konvolusi Tunggal 5x5 dengan Kernel 3x3 (Valid Padding):")
    print(f"  Input Shape:  {input_2d.shape}")
    print(f"  Kernel Shape: {kernel_2d.shape}")
    print(f"  Output Shape: {manual_res.shape} (Ekspektasi: (5-3)/1 + 1 = 3x3)")
    print(f"\nMatriks Hasil Manual (Loop Python):\n{manual_res}")
    print(f"\nMatriks Hasil PyTorch F.conv2d:\n{pytorch_res}")

    # Verifikasi Numerik
    diff = torch.abs(manual_res - pytorch_res).max().item()
    print(f"\n  Selisih Maksimum Manual vs PyTorch: {diff:.2e}")
    assert torch.allclose(manual_res, pytorch_res, atol=1e-6), "Verifikasi manual vs F.conv2d gagal!"
    print("  Status Verifikasi: 100% IDENTIK (Lolos Uji Presisi Mesin)")

    # 2. Eksperimen Multi-Channel & Multi-Kernel (Format NCHW)
    # Batch=2, C_in=3 (RGB), H=6, W=6
    # C_out=4 (4 filter fitur), Kernel=3x3
    torch.manual_seed(42)
    mc_input = torch.randn((2, 3, 6, 6))
    mc_kernel = torch.randn((4, 3, 3, 3))
    mc_bias = torch.randn(4)

    manual_mc = manual_conv2d_multichannel(mc_input, mc_kernel, mc_bias, stride=1, padding=1)
    pt_mc = F.conv2d(mc_input, mc_kernel, mc_bias, stride=1, padding=1)

    print("\n[2] Hasil Konvolusi Multi-Channel (Format NCHW):")
    print(f"  Input Tensor:  {mc_input.shape} (Batch=2, Channels=3, H=6, W=6)")
    print(f"  Kernel Tensor: {mc_kernel.shape} (C_out=4, C_in=3, K=3x3)")
    print(f"  Output Tensor: {manual_mc.shape} (Ekspektasi: Batch=2, C_out=4, H=6, W=6 dengan Padding=1)")

    diff_mc = torch.abs(manual_mc - pt_mc).max().item()
    print(f"  Selisih Maksimum Multi-Channel: {diff_mc:.2e}")
    assert torch.allclose(manual_mc, pt_mc, atol=1e-5), "Multi-channel verification failed"
    print("  Status Verifikasi Multi-Channel: 100% IDENTIK")

    print("\nKESIMPULAN: Operasi konvolusi 2D PyTorch terbukti merupakan implementasi cross-correlation terstandar.")


if __name__ == "__main__":
    main()

