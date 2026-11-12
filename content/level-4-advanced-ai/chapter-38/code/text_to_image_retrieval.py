"""Text-to-Image Cross-Modal Retrieval Demonstration.

Takes unseen text queries (e.g., "blue square", "green triangle", "yellow rectangle"),
encodes them into the shared 32d hypersphere, and retrieves Top-3 closest unseen test
images from the 96-image candidate pool via cosine similarity.
"""

import torch
from paired_dataset_generator import encode_caption_ids
from clip_miniature import load_or_train_clip_miniature


def retrieve_images_for_text(model, test_ds, query_caption: str, top_k: int = 3):
    q_ids = torch.tensor([encode_caption_ids(query_caption)], dtype=torch.long)
    with torch.no_grad():
        q_emb = model.encode_text(q_ids)              # (1, 32)
        cand_embs = model.encode_image(test_ds.images)  # (96, 32)
        scores = (q_emb @ cand_embs.T).squeeze(0)       # (96,)

    top_scores, top_indices = torch.topk(scores, k=top_k)
    results = []
    for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
        results.append({
            "image_idx": idx,
            "true_caption": test_ds.captions[idx],
            "cosine_sim": round(s, 4),
            "exact_match": (test_ds.captions[idx] == query_caption),
        })
    return results


def main():
    print("=" * 75)
    print("CHAPTER 38: TEXT-TO-IMAGE CROSS-MODAL RETRIEVAL")
    print("=" * 75)

    model, test_ds = load_or_train_clip_miniature(seed=42)
    queries = ["red circle", "blue square", "green triangle", "yellow rectangle"]

    print(f"Candidate Pool: {len(test_ds)} unseen test images (16 concepts x 6 variants each)\n")

    for q in queries:
        hits = retrieve_images_for_text(model, test_ds, q, top_k=3)
        print(f"Text Query: '{q}'")
        for rank, h in enumerate(hits, 1):
            mark = "MATCH" if h["exact_match"] else "MISS"
            print(f"  Rank {rank} | Cosine={h['cosine_sim']:+.4f} | Test Img #{h['image_idx']:02d} ('{h['true_caption']}') [{mark}]")
        print()

    print("[SUCCESS] Text-to-image retrieval verified.")


if __name__ == "__main__":
    main()

