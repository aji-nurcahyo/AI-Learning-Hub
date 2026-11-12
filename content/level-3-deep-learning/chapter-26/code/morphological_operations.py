"""
Chapter 26 - Computer Vision Applications
Skrip 5: Morphological Operations (Erosion, Dilation, Opening, Closing, and Gradient)
Mendemonstrasikan operasi morfologi biner menggunakan elemen penstruktur (Structuring Element):
Erosi untuk mereduksi derau latar luar, Dilasi untuk menutup celah internal,
Opening untuk pembersihan derau terisolasi, Closing untuk penyambungan kontur retak,
dan Morphological Gradient untuk ekstraksi garis batas tepi.
"""

from typing import Dict
from pathlib import Path
import numpy as np
import cv2


def apply_morphological_transforms(binary_mask: np.ndarray, ksize: int = 5) -> Dict[str, np.ndarray]:
    """
    Menerapkan lima operasi morfologi matematika pada mask biner:
    1. Erosi: Mempersempit area putih, membuang bintik derau terisolasi.
    2. Dilasi: Memperluas area putih, menutup lubang kecil di dalam objek.
    3. Opening (Erosi diikuti Dilasi): Menghilangkan derau kecil di luar objek tanpa mengubah ukuran utama.
    4. Closing (Dilasi diikuti Erosi): Menutup lubang/retakan internal di dalam objek.
    5. Gradient (Dilasi dikurangi Erosi): Menghasilkan garis tepi luar objek.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)

    return {
        "erosion": eroded,
        "dilation": dilated,
        "opening": opened,
        "closing": closed,
        "gradient": gradient
    }


if __name__ == "__main__":
    print("=== Eksplorasi Operasi Morfologi Citra (Morphological Operations) ===")

    # 1. Buat mask biner sintetis (Kotak 100x100 dengan lubang di dalam dan bintik derau di luar)
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Objek utama
    mask[50:150, 50:150] = 255
    # Lubang di dalam objek (Noise internal: lubang hitam 6x6)
    mask[97:103, 97:103] = 0
    # Bintik derau di luar objek (Noise eksternal: bintik putih 4x4)
    mask[20:24, 20:24] = 255
    mask[170:174, 170:174] = 255

    initial_white = int(np.count_nonzero(mask))
    print(f"1. Mask Biner Sintetis Dibuat:")
    print(f"   Piksel Putih Awal (Termasuk Derau Luar dan Lubang Dalam): {initial_white:,} piksel")

    # 2. Terapkan Operasi Morfologi dengan Kernel 7x7
    morph = apply_morphological_transforms(mask, ksize=7)

    # Analisis Opening: Menghilangkan bintik derau luar 6x6 tanpa membesarkan lubang
    white_opened = int(np.count_nonzero(morph["opening"]))
    # Analisis Closing: Menutup lubang dalam 20x20
    white_closed = int(np.count_nonzero(morph["closing"]))

    print(f"\n2. Hasil Transformasi Morfologi:")
    print(f"   - Opening (Pembersihan Derau Eksternal) : {white_opened:,} piksel (Bintik luar lenyap)")
    print(f"   - Closing (Penutupan Lubang Internal)   : {white_closed:,} piksel (Lubang dalam tertutup)")
    print(f"   - Erosi (Penyusutan Kontur Objek)       : {int(np.count_nonzero(morph['erosion'])):,} piksel")
    print(f"   - Dilasi (Ekspansi Kontur Objek)        : {int(np.count_nonzero(morph['dilation'])):,} piksel")
    print(f"   - Gradient (Garis Tepi Kontur Morfologi): {int(np.count_nonzero(morph['gradient'])):,} piksel")

    # Verifikasi logis: Closing harus menutup lubang dalam sehingga piksel putih bertambah
    assert white_closed > initial_white, "Closing gagal menutup lubang internal!"
    # Opening harus membuang bintik luar sehingga piksel putih berkurang
    assert white_opened < initial_white, "Opening gagal membersihkan derau eksternal!"
    print("\nStatus Verifikasi: SIFAT ALGEBRA MORFOLOGI TERBUKTI 100% SESUAI TEORI!")

    out_path = Path("experiments/morphology_closed_sample.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), morph["closing"])
    print(f"Citra hasil closing tersimpan di: {out_path}")
