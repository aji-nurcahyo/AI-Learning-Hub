"""
Chapter 29 - Large Language Models & Generative AI
Script 12: Hallucination Evaluation & Groundedness Verification
Mendemonstrasikan kerangka kerja audit halusinasi:
Dekomposisi klaim kalimat, verifikasi terhadap dokumen sumber acuan,
serta perhitungan skor Groundedness dan Tingkat Halusinasi (Hallucination Rate).
"""

from typing import List, Dict


def evaluate_hallucination(claims: List[str], ground_truth_context: str) -> Dict[str, float]:
    """
    Mengevaluasi setiap klaim yang dihasilkan terhadap teks konteks acuan.
    Mengklasifikasikan klaim menjadi GROUNDED (didukung) atau HALLUCINATED (tidak didukung/bertentangan).
    """
    gt_lower = ground_truth_context.lower()
    
    results = []
    supported_count = 0

    for claim in claims:
        # Pengecekan overlap entitas kunci
        claim_lower = claim.lower()
        
        # Heuristik pencocokan fakta sederhana (misal kata kunci angka/nama)
        words = [w for w in claim_lower.split() if len(w) > 3]
        matched_words = [w for w in words if w in gt_lower]
        
        # Klaim dianggap didukung jika mayoritas kata kunci substansial terdapat pada konteks
        is_supported = len(matched_words) >= max(1, int(len(words) * 0.7))
        if is_supported:
            supported_count += 1
            verdict = "TERVERIFIKASI (Grounded)"
        else:
            verdict = "HALUSINASI (Unsupported Claim)"

        results.append({
            "claim": claim,
            "supported": is_supported,
            "verdict": verdict,
            "match_ratio": len(matched_words) / max(1, len(words))
        })

    total_claims = len(claims)
    groundedness_score = (supported_count / total_claims) if total_claims > 0 else 0.0
    hallucination_rate = 1.0 - groundedness_score

    return {
        "total_claims": total_claims,
        "supported_claims": supported_count,
        "groundedness_score": groundedness_score,
        "hallucination_rate": hallucination_rate,
        "details": results
    }


def main():
    print("=" * 70)
    print("CHAPTER 29: HALLUCINATION EVALUATION & FACTUALITY AUDIT")
    print("=" * 70)

    # Dokumen Acuan (Kebenaran Dasar / Ground Truth)
    context_gt = (
        "Institut Teknologi Sumatera (ITERA) memiliki kampus seluas 287 hektar di Lampung Selatan. "
        "Pada tahun 2024, ITERA memiliki lebih dari 20.000 mahasiswa aktif yang tersebar di puluhan program studi teknik dan sains."
    )

    print("Konteks Rujukan Resmi (Ground Truth):")
    print(f"\"{context_gt}\"\n")

    # Respon Model A: Respon Faktual Disiplin
    response_A_claims = [
        "ITERA memiliki kampus seluas 287 hektar.",
        "Kampus ITERA berlokasi di Lampung Selatan.",
        "Jumlah mahasiswa aktif pada tahun 2024 mencapai lebih dari 20.000 orang."
    ]

    # Respon Model B: Respon Terkontaminasi Halusinasi
    response_B_claims = [
        "ITERA memiliki kampus seluas 287 hektar.",
        "ITERA didirikan oleh Presiden Soekarno pada tahun 1965.",  # Halusinasi sejarah
        "Kampus ITERA memiliki fakultas kedokteran hewan internasional.", # Halusinasi program studi
        "Lokasi kampus berada di Lampung Selatan."
    ]

    print("[EVALUASI RESPON A: MODEL TERKALIBRASI FAKTUAL]")
    eval_A = evaluate_hallucination(response_A_claims, context_gt)
    for idx, d in enumerate(eval_A["details"], 1):
        status_icon = "✓" if d["supported"] else "✗"
        print(f"  {status_icon} Klaim {idx}: '{d['claim']}' -> {d['verdict']}")
    print(f"  --> Skor Groundedness : {eval_A['groundedness_score']:.2%}")
    print(f"  --> Hallucination Rate: {eval_A['hallucination_rate']:.2%}")

    print("\n[EVALUASI RESPON B: MODEL BERHALUSINASI EKSTRINSIK]")
    eval_B = evaluate_hallucination(response_B_claims, context_gt)
    for idx, d in enumerate(eval_B["details"], 1):
        status_icon = "✓" if d["supported"] else "✗"
        print(f"  {status_icon} Klaim {idx}: '{d['claim']}' -> {d['verdict']}")
    print(f"  --> Skor Groundedness : {eval_B['groundedness_score']:.2%}")
    print(f"  --> Hallucination Rate: {eval_B['hallucination_rate']:.2%}")

    print("\n[TAKSONOMI HALUSINASI PADA LLM]")
    print("1. Intrinsic Hallucination: Luaran model secara eksplisit bertentangan dengan konteks dokumen sumber yang disediakan.")
    print("2. Extrinsic Hallucination: Luaran model menambahkan fakta baru yang terdengar sangat meyakinkan namun sama sekali tidak dapat diverifikasi dari sumber.")
    print("3. Bahaya Terbesar: LLM menyajikan halusinasi dengan nada keyakinan tinggi (*high confidence*), sehingga mudah mengecoh pengguna awam.")


if __name__ == "__main__":
    main()

