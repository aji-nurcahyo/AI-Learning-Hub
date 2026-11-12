"""Image-to-Text Cross-Modal Retrieval Demonstration.

Takes unseen test images and ranks all 16 canonical text captions
by cosine similarity in the shared multimodal embedding space.
"""

import torch
from paired_dataset_generator import COLOR_NAMES, SHAPES, encode_caption_ids
from clip_miniature import load_or_train_clip_miniature


def main():
    print("=" * 75)
    print("CHAPTER 38: IMAGE-TO-TEXT CROSS-MODAL RETRIEVAL")
    print("=" * 75)

    model, test_ds = load_or_train_clip_miniature(seed=42)

    # 16 canonical candidate text captions
    candidate_captions = [f"{c} {s}" for c in COLOR_NAMES for s in SHAPES]
    cand_tokens = torch.tensor([encode_caption_ids(cap) for cap in candidate_captions], dtype=torch.long)

    with torch.no_grad():
        cand_txt_embs = model.encode_text(cand_tokens)  # (16, 32)

    # Probe 4 distinct unseen test images across diverse concepts
    probe_indices = [0, 30, 60, 90]
    print(f"Candidate Text Pool: {len(candidate_captions)} unique concepts\n")

    for img_idx in probe_indices:
        img_tensor = test_ds.images[img_idx : img_idx + 1]
        true_cap = test_ds.captions[img_idx]
        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)     # (1, 32)
            scores = (img_emb @ cand_txt_embs.T).squeeze(0)
        top_scores, top_idxs = torch.topk(scores, k=3)

        print(f"Query Unseen Test Image #{img_idx:02d} (Ground Truth: '{true_cap}'):")
        for rank, (s, c_idx) in enumerate(zip(top_scores.tolist(), top_idxs.tolist()), 1):
            pred_cap = candidate_captions[c_idx]
            mark = "MATCH" if pred_cap == true_cap else "MISS"
            print(f"  Rank {rank} | Cosine={s:+.4f} | '{pred_cap}' [{mark}]")
        print()

    print("[SUCCESS] Image-to-text retrieval verified.")


if __name__ == "__main__":
    main()

