"""Teacher vs Scratch Student vs Distilled Student Comparison.

Benchmarks accuracy, parameter counts, serialized .pt disk size (KB),
and CPU latency across the Teacher model and compact Student variants.
"""

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from knowledge_distillation import train_student_distilled, train_student_scratch
from model_profiling import get_serialized_size_kb, profile_cpu_latency


def main():
    print("=" * 75)
    print("CHAPTER 37: TEACHER VS STUDENT BENCHMARK COMPARISON")
    print("=" * 75)

    teacher, (X_train, y_train, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    scratch = train_student_scratch(X_train, y_train, epochs=25, seed=42)
    distilled = train_student_distilled(teacher, X_train, y_train, epochs=25, temperature=3.0, alpha=0.6, seed=42)

    variants = [
        ("Teacher (FP32 Large)", teacher),
        ("Student (Scratch)", scratch),
        ("Student (Distilled)", distilled),
    ]

    print(f"{'Model Variant':<22} | {'Params':<10} | {'Size (KB)':<10} | {'Test Acc':<10} | {'Mean Lat (ms)'}")
    print("-" * 75)

    for name, m in variants:
        p = sum(param.numel() for param in m.parameters())
        kb = get_serialized_size_kb(m)
        acc = evaluate_accuracy(m, X_test, y_test) * 100
        lat = profile_cpu_latency(m, X_test[:1], warmup=50, n_runs=400)
        print(f"{name:<22} | {p:<10,} | {kb:<10.2f} | {acc:<9.2f}% | {lat['mean_ms']:.4f}")

    print("\n[SUCCESS] Teacher-student comparative benchmark completed.")


if __name__ == "__main__":
    main()

