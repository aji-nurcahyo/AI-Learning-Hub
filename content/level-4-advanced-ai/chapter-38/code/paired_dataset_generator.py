"""Paired Image-Text Synthetic Dataset Generator for Multimodal AI.

Generates controlled 32x32x3 RGB synthetic images paired with structured text captions:
- 4 Colors: red, blue, green, yellow
- 4 Shapes: circle, square, triangle, rectangle
- Controlled variations in center offset (dx, dy) and radius/size so Train and Test
  splits contain distinct visual instances (preventing memorization of identical images).
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


COLORS: Dict[str, Tuple[float, float, float]] = {
    "red": (0.90, 0.12, 0.12),
    "blue": (0.12, 0.25, 0.92),
    "green": (0.12, 0.80, 0.22),
    "yellow": (0.95, 0.85, 0.10),
}

SHAPES: List[str] = ["circle", "square", "triangle", "rectangle"]
COLOR_NAMES: List[str] = list(COLORS.keys())

# Vocabulary for text encoding
VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "yellow": 4,
    "circle": 5,
    "square": 6,
    "triangle": 7,
    "rectangle": 8,
    "small": 9,
    "large": 10,
}


def render_synthetic_shape(
    color_name: str,
    shape_name: str,
    size_px: int = 8,
    offset_r: int = 0,
    offset_c: int = 0,
    noise_std: float = 0.03,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Renders a 32x32x3 float32 RGB image in [0, 1] with dark background."""
    if rng is None:
        rng = np.random.default_rng(42)

    img = np.full((32, 32, 3), 0.08, dtype=np.float32)  # dark gray background
    rgb = np.array(COLORS[color_name], dtype=np.float32)

    cy, cx = 16 + offset_r, 16 + offset_c
    yy, xx = np.ogrid[:32, :32]

    if shape_name == "circle":
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= size_px ** 2
    elif shape_name == "square":
        mask = (np.abs(xx - cx) <= size_px) & (np.abs(yy - cy) <= size_px)
    elif shape_name == "rectangle":
        mask = (np.abs(xx - cx) <= int(size_px * 1.35)) & (np.abs(yy - cy) <= max(3, int(size_px * 0.65)))
    elif shape_name == "triangle":
        # Isosceles triangle pointing up
        dy = yy - (cy - size_px)
        valid_y = (dy >= 0) & (dy <= 2 * size_px)
        half_w = 0.65 * dy
        mask = valid_y & (np.abs(xx - cx) <= half_w)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

    img[mask] = rgb
    if noise_std > 0:
        img += rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def encode_caption_ids(caption: str) -> List[int]:
    tokens = caption.lower().strip().split()
    return [VOCAB.get(t, 0) for t in tokens]


def generate_paired_splits(seed: int = 42):
    """Creates strictly separated Train (320 pairs) and Test (96 pairs) datasets."""
    rng = np.random.default_rng(seed)
    train_items, test_items = [], []

    # 16 core (color, shape) concepts
    for c_idx, c_name in enumerate(COLOR_NAMES):
        for s_idx, s_name in enumerate(SHAPES):
            concept_id = c_idx * len(SHAPES) + s_idx
            caption = f"{c_name} {s_name}"
            cap_ids = encode_caption_ids(caption)

            # 20 diverse training samples per concept
            for _ in range(20):
                sz = int(rng.integers(5, 10))
                dr = int(rng.integers(-4, 5))
                dc = int(rng.integers(-4, 5))
                img = render_synthetic_shape(c_name, s_name, size_px=sz, offset_r=dr, offset_c=dc, rng=rng)
                train_items.append((img, cap_ids, caption, concept_id, c_name, s_name))

            # 6 unseen test samples with distinct offset/noise realizations
            for _ in range(6):
                sz = int(rng.integers(5, 10))
                dr = int(rng.integers(-5, 6))
                dc = int(rng.integers(-5, 6))
                img = render_synthetic_shape(c_name, s_name, size_px=sz, offset_r=dr, offset_c=dc, noise_std=0.04, rng=rng)
                test_items.append((img, cap_ids, caption, concept_id, c_name, s_name))

    return train_items, test_items


class PairedMultimodalDataset(Dataset):
    """PyTorch Dataset returning (image_CHW, caption_token_ids, concept_label, raw_caption)."""

    def __init__(self, items):
        self.images = torch.tensor(np.transpose(np.array([it[0] for it in items]), (0, 3, 1, 2)), dtype=torch.float32)
        self.tokens = torch.tensor([it[1] for it in items], dtype=torch.long)
        self.captions = [it[2] for it in items]
        self.concept_ids = torch.tensor([it[3] for it in items], dtype=torch.long)
        self.colors = [it[4] for it in items]
        self.shapes = [it[5] for it in items]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int):
        return self.images[idx], self.tokens[idx], self.concept_ids[idx], self.captions[idx]


def main():
    print("=" * 75)
    print("CHAPTER 38: PAIRED SYNTHETIC IMAGE-TEXT DATASET GENERATOR")
    print("=" * 75)

    train_items, test_items = generate_paired_splits(seed=42)
    train_ds = PairedMultimodalDataset(train_items)
    test_ds = PairedMultimodalDataset(test_items)

    print(f"Generated Synthetic Multimodal Geometry Corpus (16 concepts = 4 colors x 4 shapes):")
    print(f"- Train Split Pairs : {len(train_ds)} | Image Tensor Shape: {tuple(train_ds.images.shape)}")
    print(f"- Test  Split Pairs : {len(test_ds)}  | Text Token Shape  : {tuple(test_ds.tokens.shape)}")
    print(f"- Sample Pair [0]   : Caption='{train_ds.captions[0]}', Tokens={train_ds.tokens[0].tolist()}, ConceptID={train_ds.concept_ids[0].item()}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, "paired_dataset_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": "Synthetic Geometric Multimodal Corpus",
            "license": "CC-BY-4.0 (Original Educational Generator)",
            "colors": COLOR_NAMES,
            "shapes": SHAPES,
            "image_resolution": [32, 32, 3],
            "train_pairs": len(train_ds),
            "test_pairs": len(test_ds),
            "vocabulary_size": len(VOCAB),
        }, f, indent=2)

    print(f"\nSaved dataset specification to: {meta_path}")
    print("[SUCCESS] Paired multimodal dataset verified.")


if __name__ == "__main__":
    main()

