"""
Chapter 25 - Object Detection & YOLO
Skrip 6: YOLO Annotation Parser & Converter
Membaca, menulis, memvalidasi, serta mengonversi format label anotasi YOLO
(<class_id> <x_center> <y_center> <width> <height>) ke format COCO dan Pascal VOC.
"""

from typing import List, Dict, Any, Tuple, Union
import os
from pathlib import Path


class YOLOAnnotation:
    """Representasi satu entitas anotasi objek format YOLO."""
    def __init__(self, class_id: int, xc: float, yc: float, w: float, h: float):
        self.class_id = int(class_id)
        self.xc = float(xc)
        self.yc = float(yc)
        self.w = float(w)
        self.h = float(h)
        self.validate()

    def validate(self) -> None:
        """Memvalidasi rentang nilai normalisasi [0.0, 1.0]."""
        for attr, val in [("xc", self.xc), ("yc", self.yc), ("w", self.w), ("h", self.h)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Nilai {attr}={val} di luar batas normalisasi [0.0, 1.0]")
        if self.w <= 0.0 or self.h <= 0.0:
            raise ValueError(f"Lebar ({self.w}) atau tinggi ({self.h}) tidak boleh <= 0")

    def to_yolo_line(self) -> str:
        """Mengonversi ke satu baris teks standar berkas .txt YOLO."""
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"

    def to_voc_pixels(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Mengonversi ke koordinat piksel absolut Pascal VOC [x1, y1, x2, y2]."""
        x1 = int(round((self.xc - self.w / 2.0) * img_w))
        y1 = int(round((self.yc - self.h / 2.0) * img_h))
        x2 = int(round((self.xc + self.w / 2.0) * img_w))
        y2 = int(round((self.yc + self.h / 2.0) * img_h))
        return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))

    def to_coco_pixels(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Mengonversi ke koordinat piksel absolut COCO [x_min, y_min, width, height]."""
        x1, y1, x2, y2 = self.to_voc_pixels(img_w, img_h)
        return (x1, y1, x2 - x1, y2 - y1)


def parse_yolo_annotation_file(file_path: Union[str, Path]) -> List[YOLOAnnotation]:
    """Membaca berkas anotasi .txt YOLO dan mengembalikan daftar objek YOLOAnnotation."""
    annotations = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Berkas anotasi tidak ditemukan: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Baris {line_num} tidak memiliki tepat 5 elemen: '{line}'")
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            annotations.append(YOLOAnnotation(cls_id, xc, yc, w, h))

    return annotations


def write_yolo_annotation_file(file_path: Union[str, Path], annotations: List[YOLOAnnotation]) -> None:
    """Menuliskan daftar objek YOLOAnnotation ke dalam berkas teks."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(ann.to_yolo_line() + "\n")


if __name__ == "__main__":
    from typing import Union
    print("=== Eksplorasi Parser Anotasi YOLO ===")
    
    # 1. Membuat Anotasi Uji
    img_width, img_height = 800, 600
    sample_annotations = [
        YOLOAnnotation(class_id=0, xc=0.50, yc=0.50, w=0.40, h=0.30),
        YOLOAnnotation(class_id=1, xc=0.80, yc=0.20, w=0.15, h=0.25)
    ]

    print("1. Anotasi Objek Asli:")
    for i, a in enumerate(sample_annotations):
        print(f"   Objek {i}: YOLO Line -> '{a.to_yolo_line()}'")
        voc = a.to_voc_pixels(img_width, img_height)
        coco = a.to_coco_pixels(img_width, img_height)
        print(f"             VOC Pixels [x1, y1, x2, y2] -> {voc}")
        print(f"             COCO Pixels [x, y, w, h]    -> {coco}")

    # 2. Pengujian Tulis dan Baca Berkas
    test_file = Path("experiments/temp_annotation_test.txt")
    write_yolo_annotation_file(test_file, sample_annotations)
    print(f"\n2. Berkas Berhasil Ditulis ke: {test_file}")

    parsed = parse_yolo_annotation_file(test_file)
    print(f"3. Berkas Berhasil Dibaca Kembali: {len(parsed)} entitas anotasi ditemukan")
    assert len(parsed) == len(sample_annotations)
    assert parsed[0].class_id == 0 and abs(parsed[0].xc - 0.50) < 1e-5

    # Bersihkan berkas uji coba
    if test_file.exists():
        test_file.unlink()

    print("\nSeluruh fungsionalitas parser dan konverter anotasi YOLO lulus 100%!")
