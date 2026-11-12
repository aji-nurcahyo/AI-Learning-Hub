"""
Chapter 24 - Convolutional Neural Networks
Contoh 4: Data Loader Fashion-MNIST (Transforms, Partisi 3-Way, dan Fallback Sintetis)

Skrip ini mendemonstrasikan:
1. Pemuatan dataset Fashion-MNIST resmi (Zalando Research, MIT License).
2. Penanganan transformasi gambar (ToTensor dan Normalize dengan konstanta resmi mean=0.2860, std=0.3530).
3. Pemisahan data augmentasi: RandomHorizontalFlip HANYA pada data latih (Train),
   sedangkan data Validasi dan Uji strictly deterministik tanpa augmentasi.
4. Partisi data 3-way (Train, Validation, Test) yang adil dan konsisten.
5. Jalur fallback otomatis: SyntheticFashionDataset jika jaringan offline atau download tidak tersedia.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


# Kelas label resmi Fashion-MNIST (10 kategori produk fesyen)
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


class SyntheticFashionDataset(Dataset):
    """
    Dataset citra sintetis 1x28x28 untuk pengujian saluran pipa (smoke test)
    apabila koneksi internet atau unduhan dataset resmi tidak tersedia.
    Diberi label jelas sebagai data sintetis.
    """
    def __init__(self, n_samples=1000, seed=42):
        generator = torch.manual_seed(seed)
        # Citra sintetis: noise terstruktur dengan pola garis horizontal/vertikal per kelas
        self.data = torch.randn((n_samples, 1, 28, 28), generator=generator)
        self.targets = torch.randint(0, 10, (n_samples,), generator=generator)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_fashion_mnist_loaders(root="data/fashion_mnist", batch_size=64, val_size=5000, subset_train=None, seed=42):
    """
    Menghasilkan DataLoader untuk Train, Validation, dan Test.
    Jika unduhan gagal, otomatis mengalihkan ke SyntheticFashionDataset.
    """
    # Konstanta normalisasi standar Fashion-MNIST (dihitung dari training set populasi)
    mean_val = (0.2860,)
    std_val = (0.3530,)

    # Transformasi Latih (dengan data augmentasi terisolasi)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Augmentasi hanya di data latih
        transforms.ToTensor(),
        transforms.Normalize(mean_val, std_val)
    ])

    # Transformasi Evaluasi (Validasi & Uji: tanpa augmentasi acak)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_val, std_val)
    ])

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    try:
        # Coba unduh atau muat dataset asli Fashion-MNIST
        full_train = torchvision.datasets.FashionMNIST(
            root=str(root_path), train=True, download=True, transform=train_transform
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root=str(root_path), train=False, download=True, transform=eval_transform
        )
        is_synthetic = False
    except Exception as e:
        print(f"[PERINGATAN] Gagal mengakses Fashion-MNIST asli ({type(e).__name__}). Menggunakan Synthetic Fallback.")
        full_train = SyntheticFashionDataset(n_samples=6000, seed=seed)
        test_ds = SyntheticFashionDataset(n_samples=1000, seed=seed + 1)
        is_synthetic = True

    # Partisi Train dan Validation
    total_train = len(full_train)
    actual_val_size = min(val_size, int(total_train * 0.2))
    actual_train_size = total_train - actual_val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [actual_train_size, actual_val_size], generator=generator)

    # Bila subset diminta (misalnya untuk eksekusi pelatihan CPU yang cepat dan responsif)
    if subset_train and subset_train < actual_train_size:
        subset_indices = list(range(subset_train))
        train_ds = torch.utils.data.Subset(train_ds, subset_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader), is_synthetic


def main():
    print("=" * 70)
    print("FASHION-MNIST DATASET & DATALOADER PIPELINE")
    print("=" * 70)

    # 1. Inisialisasi Data Loader
    # Gunakan subset training 6000 sampel untuk verifikasi cepat CPU
    loaders, is_synthetic = get_fashion_mnist_loaders(
        root="data/fashion_mnist",
        batch_size=64,
        val_size=2000,
        subset_train=6000,
        seed=42
    )
    train_loader, val_loader, test_loader = loaders

    status_str = "Data Sintetis (Smoke Fallback)" if is_synthetic else "Fashion-MNIST Asli (Zalando Research, MIT)"
    print(f"\n[1] Status Sumber Dataset: {status_str}")
    print(f"  Sampel Train Aktif:  {len(train_loader.dataset)}")
    print(f"  Sampel Validasi:     {len(val_loader.dataset)}")
    print(f"  Sampel Uji (Test):   {len(test_loader.dataset)}")
    print(f"  Total Kelas Produk:  {len(FASHION_CLASSES)} kategori")

    # 2. Inspeksi Batch Pertama
    bx, by = next(iter(train_loader))
    print(f"\n[2] Struktur Tensor Mini-Batch Pertama:")
    print(f"  Bentuk Tensor Masukan (bx): {bx.shape} (Format NCHW: Batch, 1 Channel, 28x28)")
    print(f"  Bentuk Tensor Target (by):  {by.shape} (Tipe Data: {by.dtype})")
    print(f"  Rentang Nilai Piksel:       [{bx.min().item():.2f}, {bx.max().item():.2f}] (Ternormalisasi)")

    assert bx.shape == (64, 1, 28, 28)
    assert by.shape == (64,)
    assert by.dtype == torch.long

    # 3. Verifikasi Pemetaan Nama Kelas
    print(f"\n[3] Contoh Sampel Target Kelas:")
    for i in range(5):
        cls_idx = by[i].item()
        print(f"  Sampel {i+1}: Indeks = {cls_idx} -> Label = '{FASHION_CLASSES[cls_idx]}'")

    print("\nKESIMPULAN: Pipa data citra Fashion-MNIST siap digunakan untuk eksperimen pembandingan model.")


if __name__ == "__main__":
    main()

