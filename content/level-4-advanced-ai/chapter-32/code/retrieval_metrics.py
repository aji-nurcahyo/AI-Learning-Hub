"""
retrieval_metrics.py
Metrik Evaluasi Sistem Temu Kembali Informasi (Retrieval Evaluation Metrics):
- Precision@k: Proporsi dokumen yang diambil yang terbukti relevan
- Recall@k: Proporsi seluruh dokumen relevan yang berhasil ditemukan
- Hit Rate@k (Hit@k): Indikator apakah minimal satu dokumen relevan ditemukan dalam top-k
- Mean Reciprocal Rank (MRR): Rata-rata dari kebalikan posisi peringkat dokumen relevan pertama
- Evaluasi komputasi numerik pada kueri uji dengan ground-truth relevansi
"""

import numpy as np

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Precision@k = |Retrieved_k INTERSECT Relevant| / k"""
    top_k = retrieved_ids[:k]
    hits = len(set(top_k).intersection(set(relevant_ids)))
    return hits / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    """Recall@k = |Retrieved_k INTERSECT Relevant| / |Relevant|"""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k).intersection(set(relevant_ids)))
    return hits / len(relevant_ids)

def hit_rate_at_k(retrieved_ids, relevant_ids, k):
    """HitRate@k = 1.0 jika minimal ada 1 dokumen relevan di top-k, selain itu 0.0"""
    top_k = retrieved_ids[:k]
    return 1.0 if len(set(top_k).intersection(set(relevant_ids))) > 0 else 0.0

def reciprocal_rank(retrieved_ids, relevant_ids):
    """RR = 1 / rank_dokumen_relevan_pertama"""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(list_of_retrieved, list_of_relevant):
    """MRR = (1 / |Q|) * SUM(RR_q)"""
    rr_scores = [reciprocal_rank(ret, rel) for ret, rel in zip(list_of_retrieved, list_of_relevant)]
    return float(np.mean(rr_scores))

def main():
    print("=== Perhitungan Metrik Evaluasi Retrieval (Precision, Recall, MRR) ===")
    
    # Contoh Kasus 3 Kueri Evaluasi
    benchmark_data = [
        {
            "query": "Kueri 1: Konsep Transformasi Fourier",
            "relevant": ["DOC-15"],
            "retrieved": ["DOC-15", "DOC-13", "DOC-01", "DOC-05", "DOC-08"] # DOC-15 di rank 1
        },
        {
            "query": "Kueri 2: Fine-Tuning Parameter Efisien",
            "relevant": ["DOC-08", "DOC-09"],
            "retrieved": ["DOC-01", "DOC-08", "DOC-09", "DOC-02", "DOC-10"] # Relevan di rank 2 dan 3
        },
        {
            "query": "Kueri 3: Deteksi Kerusakan Sensor Getaran",
            "relevant": ["DOC-19", "DOC-14"],
            "retrieved": ["DOC-04", "DOC-03", "DOC-07", "DOC-19", "DOC-11"] # Relevan di rank 4
        }
    ]
    
    k_values = [1, 3, 5]
    all_retrieved = [b["retrieved"] for b in benchmark_data]
    all_relevant = [b["relevant"] for b in benchmark_data]
    
    print(f"Jumlah Kueri Evaluasi : {len(benchmark_data)}")
    print(f"Kedalaman Peringkat k : {k_values}\n")
    
    # 1. Evaluasi Per Kueri
    for q_idx, item in enumerate(benchmark_data, 1):
        ret = item["retrieved"]
        rel = item["relevant"]
        rr = reciprocal_rank(ret, rel)
        print(f"[{item['query']}]")
        print(f"Target Relevan: {rel}")
        print(f"Hasil Ambil   : {ret[:5]}")
        print(f"Reciprocal Rank (RR): {rr:.4f} (Dokumen relevan pertama di posisi rank {int(1/rr) if rr>0 else 'None'})")
        for k in k_values:
            p_k = precision_at_k(ret, rel, k)
            r_k = recall_at_k(ret, rel, k)
            h_k = hit_rate_at_k(ret, rel, k)
            print(f"   @k={k} -> P@{k}: {p_k:.2f} | R@{k}: {r_k:.2f} | Hit@{k}: {h_k:.1f}")
        print()
        
    # 2. Agregasi Rata-rata Keseluruhan Kueri
    print("=" * 65)
    print("AGREGASI METRIK RATA-RATA SISTEM (MACRO-AVERAGE)")
    print("=" * 65)
    mrr = mean_reciprocal_rank(all_retrieved, all_relevant)
    print(f"Mean Reciprocal Rank (MRR) : {mrr:.4f}")
    
    for k in k_values:
        mean_p = np.mean([precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)])
        mean_r = np.mean([recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)])
        mean_h = np.mean([hit_rate_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)])
        print(f"Rata-rata Precision@{k:<2}       : {mean_p:.4f} ({mean_p*100:.1f}%)")
        print(f"Rata-rata Recall@{k:<2}          : {mean_r:.4f} ({mean_r*100:.1f}%)")
        print(f"Rata-rata Hit Rate@{k:<2}        : {mean_h:.4f} ({mean_h*100:.1f}%)")
        print("-" * 65)

if __name__ == "__main__":
    main()

