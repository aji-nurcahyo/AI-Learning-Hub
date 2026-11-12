"""
Chapter 23 - Deep Learning with PyTorch
Contoh 1: Tensor Fundamentals (Shape, Dtype, Device, Operasi, dan NumPy Bridge)

Skrip ini mendemonstrasikan:
1. Pembuatan tensor dari data mentah dan fungsi inisialisasi.
2. Karakteristik tensor: shape/size, ndim, dtype, dan device.
3. Manipulasi bentuk: view, reshape, squeeze, unsqueeze, permute, flatten.
4. Operasi aljabar dan aturan broadcasting.
5. Jembatan zero-copy dua arah antara PyTorch Tensor dan NumPy Array.
"""

import torch
import numpy as np


def main():
    print("=" * 70)
    print("PYTORCH TENSOR FUNDAMENTALS")
    print("=" * 70)

    # 1. Pembuatan Tensor Dasar
    t_from_list = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    t_zeros = torch.zeros((2, 3), dtype=torch.float32)
    t_ones = torch.ones((2, 3), dtype=torch.float32)
    t_randn = torch.randn((2, 3), generator=torch.manual_seed(42))
    t_arange = torch.arange(0, 10, 2, dtype=torch.int64)

    print("\n[1] Pembuatan Tensor:")
    print(f"  Tensor dari list:\n{t_from_list}")
    print(f"  Shape: {t_from_list.shape} | ndim: {t_from_list.ndim} | dtype: {t_from_list.dtype} | device: {t_from_list.device}")
    print(f"  Zeros: shape={t_zeros.shape}, dtype={t_zeros.dtype}")
    print(f"  Arange: {t_arange.tolist()} (dtype={t_arange.dtype})")

    # 2. Reshaping dan Manipulasi Dimensi
    t_base = torch.arange(12, dtype=torch.float32)  # shape (12,)
    t_view = t_base.view(3, 4)                      # shape (3, 4)
    t_unsqueeze = t_view.unsqueeze(0)               # shape (1, 3, 4)
    t_squeeze = t_unsqueeze.squeeze(0)              # shape (3, 4)
    t_permute = t_view.permute(1, 0)                # shape (4, 3)
    t_flat = t_permute.flatten()                    # shape (12,)

    print("\n[2] Manipulasi Dimensi:")
    print(f"  Base shape:       {t_base.shape}")
    print(f"  view(3, 4):       {t_view.shape}")
    print(f"  unsqueeze(0):     {t_unsqueeze.shape}")
    print(f"  squeeze(0):       {t_squeeze.shape}")
    print(f"  permute(1, 0):    {t_permute.shape}")
    print(f"  flatten():        {t_flat.shape}")

    # Validasi keselarasan
    assert t_view.shape == (3, 4), "View shape tidak sesuai"
    assert t_permute.shape == (4, 3), "Permute shape tidak sesuai"

    # 3. Operasi Aljabar & Broadcasting
    # Penjumlahan (3, 1) + (1, 4) -> (3, 4)
    a = torch.tensor([[10.0], [20.0], [30.0]])  # shape (3, 1)
    b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])    # shape (1, 4)
    c_broadcast = a + b                         # shape (3, 4)

    print("\n[3] Operasi Aljabar & Broadcasting:")
    print(f"  a shape: {a.shape} + b shape: {b.shape} -> c shape: {c_broadcast.shape}")
    print(f"  Matriks hasil broadcast:\n{c_broadcast}")

    # Perkalian Hadamard (elemen-ke-elemen) vs Matriks (Matmul @)
    m1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    hadamard = m1 * m2
    matmul = m1 @ m2

    print(f"  Perkalian Hadamard (m1 * m2):\n{hadamard}")
    print(f"  Perkalian Matriks (m1 @ m2):\n{matmul}")
    assert torch.equal(matmul, torch.tensor([[19.0, 22.0], [43.0, 50.0]]))

    # 4. Jembatan Zero-Copy NumPy
    print("\n[4] Jembatan Zero-Copy NumPy:")
    np_arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    torch_tensor = torch.from_numpy(np_arr)
    print(f"  Awal NumPy:  {np_arr}")
    print(f"  Awal Tensor: {torch_tensor.numpy()}")

    # Mutasi in-place membuktikan memori bersama (shared memory)
    torch_tensor[0] = 99.0
    print(f"  Setelah mutasi in-place tensor[0] = 99.0:")
    print(f"    NumPy terupdate otomatis:  {np_arr[0] == 99.0} (Nilai: {np_arr[0]})")
    print(f"    Tensor terupdate otomatis: {torch_tensor[0] == 99.0} (Nilai: {torch_tensor[0].item()})")
    assert np_arr[0] == 99.0, "Memori bersama gagal diverifikasi"

    # 5. Device Management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_device = t_from_list.to(device)
    print("\n[5] Perangkat Komputasi (Device):")
    print(f"  CUDA tersedia: {torch.cuda.is_available()}")
    print(f"  Device aktif:  {device}")
    print(f"  Tensor device: {t_device.device}")

    print("\nKESIMPULAN: Seluruh operasi tensor tervalidasi 100% konsisten.")


if __name__ == "__main__":
    main()

