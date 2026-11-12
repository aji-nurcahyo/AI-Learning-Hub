"""
linear_probing.py
Implementasi dan evaluasi Linear Probing:
Hanya melatih kepala klasifikasi baru (New Classification Head)
sementara representasi backbone pra-terlatih tetap dibekukan secara utuh.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

def generate_synthetic_features(n_samples=200, feat_dim=32, n_classes=3):
    """
    Simulasi representasi fitur yang sudah diekstraksi dari backbone pra-terlatih.
    Cluster terpisah secara semantik di ruang berdimensi 32.
    """
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    # Titik pusat cluster kelas
    centers = [
        torch.tensor([ 1.5 if i % 3 == 0 else -0.5 for i in range(feat_dim) ]),
        torch.tensor([ -1.5 if i % 3 == 1 else 0.5 for i in range(feat_dim) ]),
        torch.tensor([ 1.0 if i % 3 == 2 else 0.0 for i in range(feat_dim) ])
    ]
    
    for c_idx, center in enumerate(centers):
        feat = center + torch.randn(samples_per_class, feat_dim) * 0.8
        labels = torch.full((samples_per_class,), c_idx, dtype=torch.long)
        X.append(feat)
        y.append(labels)
        
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    indices = torch.randperm(X.shape[0])
    return X[indices], y[indices]

def main():
    print("=== Eksperimen Linear Probing (Pelatihan Kepala Linier Murni) ===")
    
    # 1. Himpunan Data Fitur Hasil Ekstraksi
    X_train, y_train = generate_synthetic_features(n_samples=240, feat_dim=32, n_classes=3)
    X_test, y_test = generate_synthetic_features(n_samples=90, feat_dim=32, n_classes=3)
    
    print(f"Data Latih : {X_train.shape[0]} vektor fitur (Dimensi: {X_train.shape[1]})")
    print(f"Data Uji   : {X_test.shape[0]} vektor fitur")
    
    # 2. Kepala Linier (Linear Probe)
    linear_probe = nn.Linear(32, 3)
    total_probe_params = sum(p.numel() for p in linear_probe.parameters())
    print(f"Jumlah Parameter Linear Probe yang dilatih: {total_probe_params} parameter")
    
    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_probe.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    
    # 4. Loop Pelatihan Cepat
    print("\nMenjalankan pelatihan Linear Probing (20 Epoch)...")
    for epoch in range(1, 21):
        linear_probe.train()
        optimizer.zero_grad()
        logits = linear_probe(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            acc_train = (logits.argmax(dim=-1) == y_train).float().mean().item() * 100.0
            print(f"Epoch {epoch:2d}/20 | Loss: {loss.item():.4f} | Train Acc: {acc_train:.2f}%")
            
    # 5. Evaluasi pada Test Set
    linear_probe.eval()
    with torch.no_grad():
        test_logits = linear_probe(X_test)
        test_loss = criterion(test_logits, y_test).item()
        test_acc = (test_logits.argmax(dim=-1) == y_test).float().mean().item() * 100.0
        
    print("\n--- Hasil Evaluasi Linear Probing ---")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.2f}%")
    print("Keuntungan Linear Probing: Sangat cepat, komputasi minimal, dan tidak merusak representasi awal.")

if __name__ == "__main__":
    main()

