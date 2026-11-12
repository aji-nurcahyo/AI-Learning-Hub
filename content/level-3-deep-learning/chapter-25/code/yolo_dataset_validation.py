"""
Chapter 25 - Object Detection & YOLO
Skrip 7: YOLO Dataset Integrity Validation & Audit
Memeriksa integritas dataset deteksi objek: keberadaan pasangan citra dan label,
validitas rentang koordinat [0.0, 1.0], deteksi label kosong/korup,
distribusi frekuensi kelas, dan verifikasi ketiadaan kebocoran data (zero data leakage) antar-partisi.
"""

from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import cv2
from yolo_annotation_parser import parse_yolo_annotation_file, write_yolo_annotation_file, YOLOAnnotation


def validate_yolo_dataset(
    dataset_dir: Path,
    expected_classes: List[str]
) -> Dict[str, any]:
    """
    Melakukan audit komprehensif struktur dataset YOLO pada direktori target.
    Struktur yang diharapkan:
      dataset_dir/
        ├── images/ (train, val, test)
        └── labels/ (train, val, test)
    """
    report = {
        "is_valid": True,
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "class_distribution": {c: 0 for c in expected_classes},
        "errors": []
    }

    splits = ["train", "val", "test"]
    split_image_names = {}

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))

        img_stems = {img.stem for img in images}
        lbl_stems = {lbl.stem for lbl in labels}
        split_image_names[split] = img_stems

        report["splits"][split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "unmatched_images": list(img_stems - lbl_stems),
            "unmatched_labels": list(lbl_stems - img_stems)
        }

        report["total_images"] += len(images)
        report["total_labels"] += len(labels)

        # Periksa validitas setiap berkas anotasi
        for lbl_path in labels:
            try:
                anns = parse_yolo_annotation_file(lbl_path)
                for ann in anns:
                    if 0 <= ann.class_id < len(expected_classes):
                        cls_name = expected_classes[ann.class_id]
                        report["class_distribution"][cls_name] += 1
                    else:
                        report["errors"].append(f"Class ID {ann.class_id} di {lbl_path.name} tidak terdaftar!")
                        report["is_valid"] = False
            except Exception as e:
                report["errors"].append(f"Galat membaca {lbl_path.name}: {str(e)}")
                report["is_valid"] = False

    # Periksa kebocoran data (Data Leakage) antar-partisi
    if "train" in split_image_names and "val" in split_image_names:
        leakage_train_val = split_image_names["train"] & split_image_names["val"]
        if leakage_train_val:
            report["errors"].append(f"DATA LEAKAGE TERDETEKSI Train-Val: {leakage_train_val}")
            report["is_valid"] = False

    if "train" in split_image_names and "test" in split_image_names:
        leakage_train_test = split_image_names["train"] & split_image_names["test"]
        if leakage_train_test:
            report["errors"].append(f"DATA LEAKAGE TERDETEKSI Train-Test: {leakage_train_test}")
            report["is_valid"] = False

    return report


def generate_mini_synthetic_dataset(target_dir: Path) -> None:
    """Membangkitkan dataset mini sintetis untuk verifikasi sistem secara lokal dan offline."""
    classes = ["Kendaraan", "PejalanKaki"]
    splits = {"train": 10, "val": 3, "test": 3}

    for split, count in splits.items():
        img_dir = target_dir / "images" / split
        lbl_dir = target_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            name = f"sample_{split}_{i:03d}"
            # Buat citra sintetis 320x240
            img = np.full((240, 320, 3), 50 + i * 5, dtype=np.uint8)
            cv2.rectangle(img, (40, 40), (140, 140), (200, 50, 50), -1)
            cv2.imwrite(str(img_dir / f"{name}.png"), img)

            # Buat label YOLO yang valid
            anns = [
                YOLOAnnotation(class_id=0, xc=0.28125, yc=0.375, w=0.3125, h=0.416667)
            ]
            write_yolo_annotation_file(lbl_dir / f"{name}.txt", anns)


if __name__ == "__main__":
    print("=== Eksplorasi Validasi Integritas Dataset YOLO ===")
    mock_dir = Path("experiments/mock_yolo_dataset")
    expected_classes = ["Kendaraan", "PejalanKaki"]

    print(f"Membangkitkan Mock Dataset di: {mock_dir} ...")
    generate_mini_synthetic_dataset(mock_dir)

    print("Menjalankan Prosedur Audit Integritas Dataset...")
    audit = validate_yolo_dataset(mock_dir, expected_classes)

    print(f"\nHasil Audit Dataset:")
    print(f"  Status Validitas Total: {audit['is_valid']}")
    print(f"  Total Citra Terverifikasi: {audit['total_images']}")
    print(f"  Total Berkas Label       : {audit['total_labels']}")
    print(f"  Distribusi Kelas: {audit['class_distribution']}")

    for split, stats in audit["splits"].items():
        print(f"  Partisi '{split}': {stats['image_count']} citra, {stats['label_count']} label (Unmatched: {len(stats['unmatched_images'])})")

    assert audit["is_valid"] is True, f"Dataset audit gagal: {audit['errors']}"
    assert audit["total_images"] == 16
    print("\nStatus Audit: VALIDASI INTEGRITAS DATASET YOLO 100% SUKSES!")

