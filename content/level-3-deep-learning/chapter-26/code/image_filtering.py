"""
Chapter 26 - Computer Vision Applications
Skrip 3: Image Filtering, Smoothing, and Denoising Techniques
Mendemonstrasikan operasi penapisan spasial 2D: Box Filter, Gaussian Blur,
Median Filter (penghapus noise salt-and-pepper), dan Bilateral Filter (edge-preserving smoothing),
serta evaluasi kuantitatif kualitas perbaikan citra melalui metrik PSNR (Peak Signal-to-Noise Ratio).
"""

from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import cv2


def apply_spatial_filters(image: np.ndarray, ksize: int = 5) -> Dict[str, np.ndarray]:
    """
    Menerapkan empat teknik penapisan spasial fundamental pada citra masukan:
    1. Box Filter: Rata-rata sederhana seluruh sel jendela.
    2. Gaussian Blur: Pembobotan lonceng Gaussian berdasarkan jarak radial.
    3. Median Filter: Pengurutan median (sangat tangguh terhadap noise impulsif / salt-and-pepper).
    4. Bilateral Filter: Penghalusan non-linier yang mempertahankan ketajaman tepi objek (edge-preserving).
    """
    box_filtered = cv2.blur(image, (ksize, ksize))
    gaussian_filtered = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=1.5)
    median_filtered = cv2.medianBlur(image, ksize)
    bilateral_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    return {
        "box": box_filtered,
        "gaussian": gaussian_filtered,
        "median": median_filtered,
        "bilateral": bilateral_filtered
    }


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Menambahkan derau impulsif garam-dan-merica (salt-and-pepper noise) ke citra."""
    noisy = image.copy()
    h, w = noisy.shape[:2]
    num_noise = int(h * w * amount)

    # Salt (Piksel putih 255)
    for _ in range(num_noise // 2):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        noisy[y, x] = 255

    # Pepper (Piksel hitam 0)
    for _ in range(num_noise // 2):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        noisy[y, x] = 0

    return noisy


def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Menghitung metrik Peak Signal-to-Noise Ratio (PSNR) dalam satuan desibel (dB)."""
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


if __name__ == "__main__":
    print("=== Eksplorasi Penapisan Spasial & Peredaman Derau (Filtering) ===")
    np.random.seed(42)

    # 1. Buat citra dasar berpola kotak bersih
    clean_img = np.full((240, 320, 3), 180, dtype=np.uint8)
    cv2.rectangle(clean_img, (50, 50), (180, 180), (40, 40, 200), -1) # Kotak merah kontras

    # 2. Tambahkan noise Salt-and-Pepper
    noisy_img = add_salt_and_pepper_noise(clean_img, amount=0.08)
    initial_psnr = compute_psnr(clean_img, noisy_img)
    print(f"1. Penambahan Derau Salt-and-Pepper (8% Piksel Rusak):")
    print(f"   PSNR Awal (Citra Bernoise) : {initial_psnr:.2f} dB")

    # 3. Terapkan berbagai filter
    filters = apply_spatial_filters(noisy_img, ksize=5)

    print(f"\n2. Hasil Peredaman Derau & Evaluasi Kualitas (PSNR):")
    for name, filtered in filters.items():
        psnr_val = compute_psnr(clean_img, filtered)
        delta_psnr = psnr_val - initial_psnr
        print(f"   - {name.capitalize():<10} Filter : PSNR = {psnr_val:.2f} dB (Peningkatan: +{delta_psnr:.2f} dB)")

    # Median filter harus memberikan peningkatan PSNR tertinggi pada noise salt-and-pepper
    psnr_median = compute_psnr(clean_img, filters["median"])
    psnr_gaussian = compute_psnr(clean_img, filters["gaussian"])
    
    assert psnr_median > psnr_gaussian, "Median filter harus mengungguli Gaussian filter pada noise salt-and-pepper!"
    print(f"\nKesimpulan Ilmiah Terbukti:")
    print(f"Median Filter terbukti paling superior ({psnr_median:.2f} dB vs Gaussian {psnr_gaussian:.2f} dB) ")
    print(f"karena median mengabaikan nilai ekstrem (0 atau 255) tanpa mengaburkan tepi objek!")

    out_path = Path("experiments/denoised_median_sample.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), filters["median"])
    print(f"Citra hasil denoising median tersimpan di: {out_path}")
    print("\nSeluruh pengujian penapisan spasial berhasil!")

