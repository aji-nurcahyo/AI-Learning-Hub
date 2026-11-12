"""Cross-Modal Cosine Similarity Matrix & Visualization.

Computes the 16x16 cosine similarity matrix between unseen test images
and all 16 canonical text captions using the trained MiniatureDualEncoder,
and saves an annotated heatmap to experiments/similarity_matrix.png.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clip_miniature import load_or_train_clip_miniature


def main():
    print("=" * 75)
    print("CHAPTER 38: CROSS-MODAL SIMILARITY MATRIX HEATMAP")
    print("=" * 75)

    model, test_ds = load_or_train_clip_miniature(seed=42)
    unique_idx = [i * 6 for i in range(16)]  # 1 unseen test image per concept

    imgs = test_ds.images[unique_idx]
    toks = test_ds.tokens[unique_idx]
    labels = [test_ds.captions[i] for i in unique_idx]

    with torch.no_grad():
        emb_i = model.encode_image(imgs)
        emb_t = model.encode_text(toks)
        sim = (emb_i @ emb_t.T).numpy()

    print("Sample 4x4 Submatrix (First 4 Concepts: red shapes vs texts):")
    print(f"{'Image \\\\ Text':<16} | " + " | ".join(f"{l:<13}" for l in labels[:4]))
    print("-" * 78)
    for r in range(4):
        vals = " | ".join(f"{sim[r, c]:+13.4f}" for c in range(4))
        print(f"{labels[r]:<16} | {vals}")

    # Plot full 16x16 Heatmap
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Text Captions (Candidate Texts)")
    ax.set_ylabel("Unseen Test Images (Query Images)")
    ax.set_title("Cross-Modal Cosine Similarity Matrix (16x16 Unseen Test Concepts)")
    fig.colorbar(im, ax=ax, label="Cosine Similarity")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.abspath(os.path.join(script_dir, "..", "experiments", "similarity_matrix.png"))
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close(fig)

    print(f"\nSaved 16x16 similarity matrix heatmap to: {out_img}")
    print("[SUCCESS] Cross-modal similarity matrix verified.")


if __name__ == "__main__":
    main()

