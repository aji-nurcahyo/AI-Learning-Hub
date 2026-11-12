"""
lora_from_scratch.py
Fondasi Matematis Low-Rank Adaptation (LoRA) dari Dasar:
- Formulasi dekomposisi matriks: W' = W_0 + (alpha / r) * (B @ A)
- Inisialisasi matematis: A ~ N(0, 1/r), B = 0 sehingga Delta W = 0 saat inisialisasi
- Bukti matematis bahwa perilaku awal model sama persis dengan base model
- Komputasi forward pass dari scratch dan pembuktian rank rendah
"""

import numpy as np

np.random.seed(42)

def lora_linear_forward_scratch(x, W0, b, A, B, alpha, r):
    """
    Komputasi forward linear dengan adaptasi LoRA:
    y = x @ (W0 + (alpha / r) * (B @ A)).T + b
    atau secara lebih efisien tanpa membentuk matriks penuh:
    y = x @ W0.T + (alpha / r) * ((x @ A.T) @ B.T) + b
    
    Dimensi:
    - x  : [Batch, d_in]
    - W0 : [d_out, d_in]
    - A  : [r, d_in]
    - B  : [d_out, r]
    - b  : [d_out]
    """
    # Jalur 1: Base linear projection (bobot beku)
    base_out = x @ W0.T + b # [Batch, d_out]
    
    # Jalur 2: Low-rank adapter path (hemat komputasi)
    # Langkah A: Proyeksi turun (down-projection) ke dimensi rank r
    down = x @ A.T # [Batch, r]
    # Langkah B: Proyeksi naik (up-projection) ke dimensi d_out
    up = down @ B.T # [Batch, d_out]
    
    scaling = alpha / r
    adapter_out = scaling * up
    
    # Penjumlahan hasil
    return base_out + adapter_out

def main():
    print("=== LoRA Mathematical Foundations from Scratch ===")
    d_in = 128
    d_out = 128
    r = 4
    alpha = 8.0
    batch_size = 2
    
    print(f"Dimensi Matriks Base W0 : [{d_out}, {d_in}] (Total Elemen: {d_out * d_in:,})")
    print(f"Rank LoRA (r)           : {r}")
    print(f"Scaling Factor (alpha)  : {alpha} (Skalar Pengali: alpha / r = {alpha / r:.2f})")
    
    # Inisialisasi Bobot Dasar (W0) dan Bias (b)
    W0 = np.random.randn(d_out, d_in) * 0.02
    b = np.zeros(d_out)
    
    # Inisialisasi LoRA: A acak Gauss, B nol mutlak
    A = np.random.randn(r, d_in) * np.sqrt(1.0 / r)
    B = np.zeros((d_out, r))
    
    print(f"Dimensi Matriks A (Down): [{r}, {d_in}] (Total: {r * d_in:,})")
    print(f"Dimensi Matriks B (Up)  : [{d_out}, {r}] (Total: {d_out * r:,})")
    total_lora_params = (r * d_in) + (d_out * r)
    print(f"Total Parameter LoRA    : {total_lora_params:,} parameter")
    savings = (1.0 - (total_lora_params / (d_out * d_in))) * 100.0
    print(f"Efisiensi Penghematan   : {savings:.2f}% pengurangan parameter!")
    
    # Vektor input uji
    x = np.random.randn(batch_size, d_in)
    
    # 1. Uji Saat Inisialisasi: Bukti Delta W = 0
    delta_W_init = (alpha / r) * (B @ A)
    max_delta_init = np.max(np.abs(delta_W_init))
    print(f"\n[Verifikasi 1] Nilai Maksimum Absolut Delta W saat inisialisasi: {max_delta_init:.8f}")
    assert max_delta_init == 0.0, "Delta W awal harus nol mutlak!"
    
    out_base = x @ W0.T + b
    out_lora_init = lora_linear_forward_scratch(x, W0, b, A, B, alpha, r)
    diff_init = np.max(np.abs(out_base - out_lora_init))
    print(f"[Verifikasi 2] Selisih Output LoRA vs Base saat inisialisasi    : {diff_init:.8f}")
    assert diff_init == 0.0, "Output LoRA harus identik dengan Base saat awal!"
    print("-> Terbukti: Model LoRA pada langkah awal t=0 memiliki perilaku identik 100% dengan base model.")
    
    # 2. Uji Setelah Pembaruan Bobot (Simulasi Gradient Step pada B dan A)
    # B mendapat update gradien kecil
    B_updated = B + np.random.randn(d_out, r) * 0.01
    A_updated = A + np.random.randn(r, d_in) * 0.01
    
    # Evaluasi ekuivalensi matematis antara (B @ A) penuh vs forward terpisah
    # Pendekatan 1: Membentuk Delta W penuh lalu mengalikan
    W_merged = W0 + (alpha / r) * (B_updated @ A_updated)
    out_merged = x @ W_merged.T + b
    
    # Pendekatan 2: Tanpa membentuk matriks penuh (LoRA factorization forward)
    out_factored = lora_linear_forward_scratch(x, W0, b, A_updated, B_updated, alpha, r)
    
    diff_forward = np.max(np.abs(out_merged - out_factored))
    print(f"\n[Verifikasi 3] Selisih Perhitungan Faktorisasi vs Matriks Penuh : {diff_forward:.8e}")
    assert diff_forward < 1e-12, "Perhitungan faktorisasi harus identik secara aljabar!"
    
    # Periksa Rank Matriks Delta W
    rank_delta = np.linalg.matrix_rank((B_updated @ A_updated))
    print(f"[Verifikasi 4] Rank Matematis Matriks Delta W (B @ A)          : {rank_delta} (Maksimal r={r})")
    assert rank_delta <= r, f"Rank tidak boleh melebihi r={r}"
    
    print("\nKesimpulan: Implementasi LoRA scratch membuktikan bahwa komputasi low-rank")
    print("memberikan representasi aljabar yang presisi tanpa pernah mewajibkan pembentukan")
    print("matriks penuh di memori saat proses pelatihan.")

if __name__ == "__main__":
    main()

