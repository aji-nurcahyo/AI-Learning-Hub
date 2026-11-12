"""
Chapter 23 - Deep Learning with PyTorch
Contoh 2: Autograd Mechanics (Kalkulus Otomatis, Akumulasi Gradien, dan no_grad)

Skrip ini mendemonstrasikan:
1. Pelacakan graf komputasi menggunakan requires_grad=True.
2. Perhitungan gradien otomatis (.backward()) dibandingkan turunan analitis eksak.
3. Bahaya akumulasi gradien bawaan PyTorch dan perlunya zero_grad().
4. Penghematan memori evaluasi dengan torch.no_grad() dan .detach().
"""

import torch


def main():
    print("=" * 70)
    print("PYTORCH AUTOGRAD MECHANICS")
    print("=" * 70)

    # 1. Autograd vs Turunan Analitis
    # Fungsi: L = sum( (w1*x1 + w2*x2 + b - y)^2 ) untuk 2 sampel data
    X = torch.tensor([[1.5, 2.0], [3.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[5.0], [7.0]], dtype=torch.float32)

    w1 = torch.tensor(0.8, requires_grad=True)
    w2 = torch.tensor(1.2, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)

    # Forward pass
    # z_i = w1 * x_i1 + w2 * x_i2 + b
    z1 = w1 * X[0, 0] + w2 * X[0, 1] + b  # 0.8*1.5 + 1.2*2.0 + 0.5 = 1.2 + 2.4 + 0.5 = 4.1
    z2 = w1 * X[1, 0] + w2 * X[1, 1] + b  # 0.8*3.0 + 1.2*1.0 + 0.5 = 2.4 + 1.2 + 0.5 = 4.1
    diff1 = z1 - y[0, 0]  # 4.1 - 5.0 = -0.9
    diff2 = z2 - y[1, 0]  # 4.1 - 7.0 = -2.9
    L = diff1**2 + diff2**2  # (-0.9)^2 + (-2.9)^2 = 0.81 + 8.41 = 9.22

    # Backward pass
    L.backward()

    # Hitung Analitis Turunan Parsial:
    # dL/dw1 = 2*diff1*x11 + 2*diff2*x21 = 2*(-0.9)*1.5 + 2*(-2.9)*3.0 = -2.7 - 17.4 = -20.1
    # dL/dw2 = 2*diff1*x12 + 2*diff2*x22 = 2*(-0.9)*2.0 + 2*(-2.9)*1.0 = -3.6 - 5.8  = -9.4
    # dL/db  = 2*diff1*1.0 + 2*diff2*1.0 = 2*(-0.9)*1.0 + 2*(-2.9)*1.0 = -1.8 - 5.8  = -7.6
    dL_dw1_analytic = 2.0 * (-0.9) * 1.5 + 2.0 * (-2.9) * 3.0
    dL_dw2_analytic = 2.0 * (-0.9) * 2.0 + 2.0 * (-2.9) * 1.0
    dL_db_analytic = 2.0 * (-0.9) * 1.0 + 2.0 * (-2.9) * 1.0

    print("\n[1] Perbandingan Gradien Autograd vs Turunan Analitis:")
    print(f"  Loss L = {L.item():.4f}")
    print(f"  dL/dw1 -> Autograd: {w1.grad.item():.4f} | Analitis: {dL_dw1_analytic:.4f}")
    print(f"  dL/dw2 -> Autograd: {w2.grad.item():.4f} | Analitis: {dL_dw2_analytic:.4f}")
    print(f"  dL/db  -> Autograd: {b.grad.item():.4f} | Analitis: {dL_db_analytic:.4f}")

    assert abs(w1.grad.item() - dL_dw1_analytic) < 1e-5
    assert abs(w2.grad.item() - dL_dw2_analytic) < 1e-5
    assert abs(b.grad.item() - dL_db_analytic) < 1e-5

    # 2. Investigasi Akumulasi Gradien
    print("\n[2] Demonstrasi Akumulasi Gradien (Gradient Accumulation):")
    print(f"  Gradien w1 saat ini: {w1.grad.item():.4f}")
    
    # Jalankan forward & backward kedua TANPA zero_grad()
    z1_again = w1 * X[0, 0] + w2 * X[0, 1] + b
    z2_again = w1 * X[1, 0] + w2 * X[1, 1] + b
    L_again = (z1_again - y[0, 0])**2 + (z2_again - y[1, 0])**2
    L_again.backward()
    print(f"  Gradien w1 setelah backward kedua (tanpa zero_grad): {w1.grad.item():.4f} (Berlipat ganda!)")
    assert abs(w1.grad.item() - (2 * dL_dw1_analytic)) < 1e-4

    # Lakukan reset gradien secara eksplisit
    w1.grad.zero_()
    w2.grad.zero_()
    b.grad.zero_()
    print(f"  Gradien w1 setelah w1.grad.zero_(): {w1.grad.item():.4f} (Kembali bersih)")
    assert w1.grad.item() == 0.0

    # 3. Nonaktifkan Graf Komputasi dengan torch.no_grad() dan .detach()
    print("\n[3] Penghematan Memori dengan torch.no_grad() & .detach():")
    with torch.no_grad():
        eval_z = w1 * X[0, 0] + w2 * X[0, 1] + b
        print(f"  Di dalam torch.no_grad(): requires_grad={eval_z.requires_grad}")
        assert not eval_z.requires_grad

    # Demonstrasi .detach()
    live_tensor = w1 * 2.5
    detached_tensor = live_tensor.detach()
    print(f"  live_tensor.requires_grad:     {live_tensor.requires_grad}")
    print(f"  detached_tensor.requires_grad: {detached_tensor.requires_grad}")
    assert live_tensor.requires_grad
    assert not detached_tensor.requires_grad

    print("\nKESIMPULAN: Mekanisme autograd, akumulasi, dan no_grad diverifikasi 100% presisi.")


if __name__ == "__main__":
    main()

