"""Privacy Protection, Data Minimization, Re-Identification Risk & Laplace DP Demo.

Uses strictly synthetic dummy identifiers (zero real personal data).
"""

import numpy as np


def demonstrate_quasi_id_reidentification():
    # Synthetic records where direct PII ("name") was removed, leaving quasi-identifiers:
    # (postal_prefix, birth_year, role_code)
    records = [
        {"name": "SYNTH-01", "zip3": "351", "birth_year": 2002, "dept": "IF", "grade": 88.5},
        {"name": "SYNTH-02", "zip3": "351", "birth_year": 2002, "dept": "IF", "grade": 91.0},
        {"name": "SYNTH-03", "zip3": "353", "birth_year": 2001, "dept": "EL", "grade": 79.0},  # Unique combination!
        {"name": "SYNTH-04", "zip3": "351", "birth_year": 2002, "dept": "IF", "grade": 84.0},
    ]

    # Data minimization: drop name & generalize birth_year -> 5-year bucket (k-anonymity mitigation)
    minimized = []
    qi_counts = {}
    for r in records:
        qi = (r["zip3"], r["birth_year"], r["dept"])
        qi_counts[qi] = qi_counts.get(qi, 0) + 1
        minimized.append({"zip3": r["zip3"], "age_bucket": "2000-2004", "grade": r["grade"]})

    singleton_risks = sum(1 for c in qi_counts.values() if c == 1)
    assert singleton_risks == 1
    return records, minimized, singleton_risks


def laplace_dp_mean(values: np.ndarray, epsilon: float, sensitivity: float, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    true_mean = float(np.mean(values))
    scale = sensitivity / epsilon
    noise = float(rng.laplace(0.0, scale))
    return true_mean + noise


def main():
    print("=" * 78)
    print("CHAPTER 39: PRIVACY, DATA MINIMIZATION & RE-IDENTIFICATION RISK")
    print("=" * 78)

    raw, minimized, singletons = demonstrate_quasi_id_reidentification()
    print(f"1. Re-Identification Audit on Naive De-Identified Records (n={len(raw)}):")
    print(f"   - Dropped direct identifier ('name'), kept (zip3, birth_year, dept).")
    print(f"   - Unique singleton quasi-identifiers found: {singletons} (Re-identification risk = {100*singletons/len(raw):.1f}%!)")
    print(f"   - Generalized minimized record sample    : {minimized[2]}")
    print("   Lesson: Simple name deletion / anonymization is NOT an absolute privacy guarantee.\n")

    grades = np.array([r["grade"] for r in raw], dtype=float)
    true_avg = float(np.mean(grades))
    sensitivity = (100.0 - 0.0) / len(grades)  # bounded range [0, 100] / n
    print("2. Laplace Differential Privacy Concept (Bounded Mean Query):")
    print(f"   - True Non-Private Mean : {true_avg:.2f}")
    for eps in [2.0, 0.5, 0.1]:
        noisy_avg = laplace_dp_mean(grades, epsilon=eps, sensitivity=sensitivity, seed=42)
        print(f"   - DP Mean (epsilon={eps:3.1f}) : {noisy_avg:.2f} (Smaller epsilon -> stronger privacy, more noise)")

    print("\n[SUCCESS] Privacy controls & limits demonstration passed.")


if __name__ == "__main__":
    main()

