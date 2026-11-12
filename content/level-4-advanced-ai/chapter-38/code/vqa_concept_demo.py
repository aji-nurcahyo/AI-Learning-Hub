"""Visual Question Answering (VQA) Conceptual Pipeline & Deterministic Baseline.

Demonstrates the conceptual VQA pipeline:
  Image Context + Question Text -> Multimodal Fusion -> Grounded Answer

NOTE: This educational script implements a deterministic visual attribute
extraction baseline (inspecting RGB pixel intensity & foreground bounding geometry)
paired with question-type routing. It is explicitly NOT a neural generative VQA model.
"""

import numpy as np
from paired_dataset_generator import COLORS, generate_paired_splits, PairedMultimodalDataset


def extract_visual_attributes(img_hwc: np.ndarray) -> dict:
    """Deterministically analyzes synthetic image foreground pixels for color and geometry."""
    # Foreground mask where pixel exceeds dark background (0.08)
    fg_mask = np.max(img_hwc, axis=-1) > 0.25
    if not np.any(fg_mask):
        return {"color": "unknown", "shape": "none", "area_px": 0}

    # Mean foreground RGB color -> match nearest known color
    mean_rgb = np.mean(img_hwc[fg_mask], axis=0)
    best_color = min(
        COLORS.keys(),
        key=lambda cname: float(np.linalg.norm(mean_rgb - np.array(COLORS[cname]))),
    )

    # Geometry fill factor inside bounding box:
    # Circle fill ~ pi/4 (~0.785), Square/Rect ~ 1.0, Triangle ~ 0.50
    ys, xs = np.where(fg_mask)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    area = int(fg_mask.sum())
    bbox_area = max(1, h * w)
    fill_ratio = area / bbox_area
    aspect_ratio = w / max(1, h)

    if fill_ratio < 0.63:
        shape = "triangle"
    elif fill_ratio < 0.88:
        shape = "circle"
    elif aspect_ratio > 1.35:
        shape = "rectangle"
    else:
        shape = "square"

    return {
        "color": best_color,
        "shape": shape,
        "area_px": area,
        "fill_ratio": round(fill_ratio, 3),
        "aspect_ratio": round(aspect_ratio, 2),
    }


def answer_vqa_question(img_hwc: np.ndarray, question: str) -> str:
    attrs = extract_visual_attributes(img_hwc)
    q_low = question.lower()
    if "color" in q_low or "warna" in q_low:
        return f"Warna objek dominan pada citra adalah '{attrs['color']}'."
    elif "shape" in q_low or "bentuk" in q_low:
        return f"Bentuk geometris yang terdeteksi adalah '{attrs['shape']}' (fill_ratio={attrs['fill_ratio']})."
    elif "area" in q_low or "luas" in q_low or "besar" in q_low:
        return f"Luas area piksel objek adalah {attrs['area_px']} piksel."
    return f"Objek terdeteksi berupa {attrs['color']} {attrs['shape']}."


def main():
    print("=" * 75)
    print("CHAPTER 38: VISUAL QUESTION ANSWERING (RULE-BASED CONCEPT DEMO)")
    print("=" * 75)
    print("Disclaimer: Rule-based deterministic attribute baseline, NOT neural generative VQA.\n")

    _, test_items = generate_paired_splits(seed=42)
    test_ds = PairedMultimodalDataset(test_items)

    # Probe 4 unseen test samples across different concepts
    for idx in [0, 30, 60, 90]:
        img_hwc = np.transpose(test_ds.images[idx].numpy(), (1, 2, 0))
        true_cap = test_ds.captions[idx]
        q1 = "Apa warna objek pada gambar?"
        q2 = "Apa bentuk geometris objek tersebut?"
        print(f"Test Image #{idx:02d} (Ground Truth: '{true_cap}'):")
        print(f"  Q1: '{q1}' -> {answer_vqa_question(img_hwc, q1)}")
        print(f"  Q2: '{q2}' -> {answer_vqa_question(img_hwc, q2)}\n")

    print("[SUCCESS] VQA conceptual pipeline demonstrated.")


if __name__ == "__main__":
    main()

