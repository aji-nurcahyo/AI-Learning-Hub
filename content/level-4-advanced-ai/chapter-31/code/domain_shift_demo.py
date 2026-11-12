"""
domain_shift_demo.py
Simulasi pergeseran domain (Domain Shift & Distribution Shift):
- Covariate Shift: P(X) bergeser sementara hubungan P(Y|X) tetap
- Mengukur degradasi performa model saat berpindah dari Source Domain ke Target Domain
- Evaluasi perbaikan performa setelah adaptasi domain (Domain Adaptation)
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

def generate_domain_data(n_samples, shift_factor=0.0):
    """
    Menghasilkan data sensor telemetri.
    shift_factor merepresentasikan pergeseran kondisi lingkungan (suhu/tekanan ekstrem).
    """
    # Fitur 0: Sinyal getaran dasar
    vibration = torch.randn(n_samples, 1) + shift_factor
    # Fitur 1: Sinyal suhu motor
    temperature = torch.randn(n_samples, 1) + (shift_factor * 1.5)
    # Fitur 2: Derau acak
    noise = torch.randn(n_samples, 1)
    
    X = torch.cat([vibration, temperature, noise], dim=-1)
    # Aturan label fisik: Kerusakan jika kombinasi getaran dan suhu melewati ambang batas
    # P(Y|X) tetap invarian secara fisik
    y = ((vibration + temperature) > 0.5).long().squeeze(-1)
    return X, y

def main():
    print("=== Eksperimen Domain Shift & Covariate Shift ===")
    
    # 1. Source Domain (Kondisi Standar: shift_factor = 0.0)
    X_src_train, y_src_train = generate_domain_data(n_samples=400, shift_factor=0.0)
    X_src_test, y_src_test   = generate_domain_data(n_samples=100, shift_factor=0.0)
    
    # 2. Target Domain (Kondisi Ekstrem: shift_factor = 2.0)
    X_tgt_test, y_tgt_test   = generate_domain_data(n_samples=100, shift_factor=2.0)
    # Sedikit data latih target untuk adaptasi (low resource domain adaptation)
    X_tgt_adapt, y_tgt_adapt = generate_domain_data(n_samples=40, shift_factor=2.0)
    
    # 3. Latih Model pada Source Domain
    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    
    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(model(X_src_train), y_src_train)
        loss.backward()
        optimizer.step()
        
    # Evaluasi In-Domain vs Out-of-Domain (Shifted)
    acc_in_domain = (model(X_src_test).argmax(dim=-1) == y_src_test).float().mean().item() * 100.0
    acc_out_domain_before = (model(X_tgt_test).argmax(dim=-1) == y_tgt_test).float().mean().item() * 100.0
    
    print("\n--- Evaluasi Sebelum Adaptasi Domain ---")
    print(f"Performa In-Domain (Source Domain)      : {acc_in_domain:.2f}%")
    print(f"Performa Out-of-Domain (Target Shifted) : {acc_out_domain_before:.2f}%")
    drop = acc_in_domain - acc_out_domain_before
    print(f"Degradasi Akurasi akibat Domain Shift   : -{drop:.2f}%")
    
    # 4. Melakukan Domain Adaptation (Fine-Tuning pada 40 sampel Target Domain)
    print("\n--- Menjalankan Fine-Tuning Adaptasi Domain (40 sampel target) ---")
    opt_adapt = optim.Adam(model.parameters(), lr=0.005)
    for _ in range(15):
        opt_adapt.zero_grad()
        loss = criterion(model(X_tgt_adapt), y_tgt_adapt)
        loss.backward()
        opt_adapt.step()
        
    acc_out_domain_after = (model(X_tgt_test).argmax(dim=-1) == y_tgt_test).float().mean().item() * 100.0
    print(f"Performa Target Domain Setelah Adaptasi : {acc_out_domain_after:.2f}%")
    gain = acc_out_domain_after - acc_out_domain_before
    print(f"Pemulihan Akurasi (Adaptation Gain)     : +{gain:.2f}%")
    
    print("\nKesimpulan: Model yang unggul di domain sumber dapat mengalami keruntuhan")
    print("performa saat menghadapi pergeseran kovariat. Fine-tuning terarah dengan sedikit")
    print("sampel domain target mampu memulihkan batas keputusan model secara signifikan.")

if __name__ == "__main__":
    main()

