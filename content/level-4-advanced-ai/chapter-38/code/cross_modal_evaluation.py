"""Bidirectional Cross-Modal Retrieval Evaluation Benchmark.

Evaluates the MiniatureDualEncoder strictly on the unseen test split (96 samples, 16 concepts):
1. Image-to-Text (I->T): For every unseen test image (N=96), ranks all 16 canonical captions.
   Computes Recall@1, Recall@3, Recall@5, and Median Rank.
2. Text-to-Image (T->I): For every canonical caption query (N=16), ranks all 96 unseen test images.
   Computes Recall@1, Recall@3, Recall@5, Precision@6 (exact concept matches), and Median Rank.
"""

import numpy as np
import torch

from paired_dataset_generator import COLOR_NAMES, SHAPES, encode_caption_ids
from clip_miniature import load_or_train_clip_miniature


def evaluate_i2t(img_embs: torch.Tensor, img_concepts: torch.Tensor, txt_embs: torch.Tensor) -> dict:
    # img_embs: (96, D), txt_embs: (16, D)
    sim = (img_embs @ txt_embs.T).numpy()
    ranks = []
    for i in range(len(img_embs)):
        gt_concept = int(img_concepts[i].item())
        order = np.argsort(-sim[i])  # descending
        pos = int(np.where(order == gt_concept)[0][0]) + 1  # 1-indexed rank
        ranks.append(pos)

    ranks_arr = np.array(ranks)
    return {
        "R@1": round(float(np.mean(ranks_arr <= 1) * 100), 2),
        "R@3": round(float(np.mean(ranks_arr <= 3) * 100), 2),
        "R@5": round(float(np.mean(ranks_arr <= 5) * 100), 2),
        "median_rank": float(np.median(ranks_arr)),
    }


def evaluate_t2i(txt_embs: torch.Tensor, img_embs: torch.Tensor, img_concepts: torch.Tensor) -> dict:
    # txt_embs: (16, D), img_embs: (96, D)
    sim = (txt_embs @ img_embs.T).numpy()
    img_c = img_concepts.numpy()

    ranks_first_hit = []
    p_at_6_list = []

    for c_id in range(len(txt_embs)):
        order = np.argsort(-sim[c_id])
        matched_positions = np.where(img_c[order] == c_id)[0] + 1
        ranks_first_hit.append(int(matched_positions[0]))
        p_at_6 = float(np.mean(img_c[order[:6]] == c_id))
        p_at_6_list.append(p_at_6)

    ranks_arr = np.array(ranks_first_hit)
    return {
        "R@1": round(float(np.mean(ranks_arr <= 1) * 100), 2),
        "R@3": round(float(np.mean(ranks_arr <= 3) * 100), 2),
        "R@5": round(float(np.mean(ranks_arr <= 5) * 100), 2),
        "mean_P@6": round(float(np.mean(p_at_6_list) * 100), 2),
        "median_rank": float(np.median(ranks_arr)),
    }


def main():
    print("=" * 75)
    print("CHAPTER 38: BIDIRECTIONAL CROSS-MODAL RETRIEVAL EVALUATION")
    print("=" * 75)

    model, test_ds = load_or_train_clip_miniature(seed=42)
    candidate_captions = [f"{c} {s}" for c in COLOR_NAMES for s in SHAPES]
    cand_tokens = torch.tensor([encode_caption_ids(cap) for cap in candidate_captions], dtype=torch.long)

    with torch.no_grad():
        test_img_embs = model.encode_image(test_ds.images)  # (96, 32)
        canon_txt_embs = model.encode_text(cand_tokens)     # (16, 32)

    i2t = evaluate_i2t(test_img_embs, test_ds.concept_ids, canon_txt_embs)
    t2i = evaluate_t2i(canon_txt_embs, test_img_embs, test_ds.concept_ids)

    print("Evaluation Strictly on Held-Out Unseen Test Samples (N=96 images, 16 concepts):\n")
    print("1. Image-to-Text Retrieval (Query=Image -> 16 Captions):")
    print(f"   Recall@1: {i2t['R@1']:.2f}% | Recall@3: {i2t['R@3']:.2f}% | Recall@5: {i2t['R@5']:.2f}% | Median Rank: {i2t['median_rank']:.1f}")

    print("\n2. Text-to-Image Retrieval (Query=Caption -> 96 Unseen Images):")
    print(f"   Recall@1: {t2i['R@1']:.2f}% | Recall@3: {t2i['R@3']:.2f}% | Recall@5: {t2i['R@5']:.2f}% | Precision@6: {t2i['mean_P@6']:.2f}% | Median Rank: {t2i['median_rank']:.1f}")

    print("\nScope Note: High scores demonstrate proper alignment over controlled synthetic geometries;")
    print("they do NOT imply real-world semantic understanding on open-domain photography.")
    print("[SUCCESS] Cross-modal retrieval benchmark completed.")


if __name__ == "__main__":
    main()

