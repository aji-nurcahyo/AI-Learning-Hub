"""
Chapter 23 - Deep Learning with PyTorch
Contoh 3: Custom Dataset dan DataLoader (Partisi 3-Way & Pencegahan Data Leakage)

Skrip ini mendemonstrasikan:
1. Pembuatan dataset klasifikasi sintetis dengan seed deterministik.
2. Subclassing torch.utils.data.Dataset (__len__, __getitem__).
3. Partisi data 3-way (Train 70%, Validation 15%, Test 15%) tanpa tumpang tindih.
4. Normalisasi fitur bebas kebocoran data (scaler fit hanya pada data latih).
5. Pembungkusan DataLoader dengan konfigurasi batching dan shuffling yang tepat.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticTabularDataset(Dataset):
    """
    Dataset sintetis tabular untuk klasifikasi biner.
    Catatan: Dibuat khusus untuk pengujian alur kerja PyTorch,
    bukan klaim performa data dunia nyata.
    """
    def __init__(self, X, y):
        # Konversi ke float32 untuk fitur dan int64 (long) untuk target klasifikasi
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_synthetic_data(n_samples=1000, n_features=6, seed=42):
    """Membangkitkan data sintetis non-linier terdefinisi dengan seed tetap."""
    generator = torch.manual_seed(seed)
    # Fitur kontinu berdistribusi normal
    X = torch.randn((n_samples, n_features), generator=generator)
    # Aturan target sintetis non-linier: kombinasi kuadratik dan interaksi fitur
    score = (
        1.5 * X[:, 0]
        - 2.0 * X[:, 1]
        + 0.8 * (X[:, 2] ** 2)
        - 1.2 * (X[:, 0] * X[:, 3])
        + 0.5 * torch.randn(n_samples, generator=generator)
    )
    # Binerisasi berdasarkan median skor
    y = (score > score.median()).long()
    return X, y


def split_data_zero_leakage(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Membagi data menjadi Train, Val, Test secara terisolasi.
    Statistik normalisasi (mean, std) dihitung HANYA dari data Train.
    """
    n_total = len(X)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    generator = torch.manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    # Verifikasi tidak ada irisan index (zero overlap)
    set_train = set(idx_train.tolist())
    set_val = set(idx_val.tolist())
    set_test = set(idx_test.tolist())
    assert len(set_train.intersection(set_val)) == 0, "Kebocoran index train-val!"
    assert len(set_train.intersection(set_test)) == 0, "Kebocoran index train-test!"
    assert len(set_val.intersection(set_test)) == 0, "Kebocoran index val-test!"

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # Standardisasi bebas data leakage: fit HANYA pada X_train
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    eps = 1e-8

    X_train_norm = (X_train - mean) / (std + eps)
    X_val_norm = (X_val - mean) / (std + eps)
    X_test_norm = (X_test - mean) / (std + eps)

    return (X_train_norm, y_train), (X_val_norm, y_val), (X_test_norm, y_test)


def main():
    print("=" * 70)
    print("PYTORCH CUSTOM DATASET & DATALOADER")
    print("=" * 70)

    # 1. Pembangkitan Data Sintetis
    X_raw, y_raw = generate_synthetic_data(n_samples=1000, n_features=6, seed=42)
    print(f"\n[1] Pembangkitan Data Sintetis:")
    print(f"  Total sampel:   {len(X_raw)}")
    print(f"  Jumlah fitur:   {X_raw.shape[1]}")
    print(f"  Distribusi y:   Kelas 0 = {(y_raw == 0).sum().item()}, Kelas 1 = {(y_raw == 1).sum().item()}")

    # 2. Pembagian Data 3-Way Bebas Leakage
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_data_zero_leakage(X_raw, y_raw)
    print(f"\n[2] Partisi Data 3-Way Terisolasi:")
    print(f"  Train:      {len(X_tr)} sampel ({len(X_tr)/len(X_raw)*100:.1f}%)")
    print(f"  Validation: {len(X_va)} sampel ({len(X_va)/len(X_raw)*100:.1f}%)")
    print(f"  Test:       {len(X_te)} sampel ({len(X_te)/len(X_raw)*100:.1f}%)")

    # 3. Pembuatan Dataset Objek
    train_dataset = SyntheticTabularDataset(X_tr, y_tr)
    val_dataset = SyntheticTabularDataset(X_va, y_va)
    test_dataset = SyntheticTabularDataset(X_te, y_te)

    # 4. Pembuatan DataLoader
    # Catatan: shuffle=True HANYA untuk training loader!
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n[3] Konfigurasi DataLoader (Batch Size = {batch_size}):")
    print(f"  Batches per Epoch (Train): {len(train_loader)} (shuffle=True)")
    print(f"  Batches per Epoch (Val):   {len(val_loader)} (shuffle=False)")
    print(f"  Batches per Epoch (Test):  {len(test_loader)} (shuffle=False)")

    # 5. Uji Ekstraksi Batch Pertama
    batch_x, batch_y = next(iter(train_loader))
    print(f"\n[4] Inspeksi Tensor Mini-Batch Pertama:")
    print(f"  Fitur Batch Shape:  {batch_x.shape} (dtype={batch_x.dtype})")
    print(f"  Label Batch Shape:  {batch_y.shape} (dtype={batch_y.dtype})")

    assert batch_x.shape == (batch_size, 6)
    assert batch_y.shape == (batch_size,)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.long

    print("\nKESIMPULAN: Abstraksi Dataset dan DataLoader siap digunakan untuk training loop.")


if __name__ == "__main__":
    main()

