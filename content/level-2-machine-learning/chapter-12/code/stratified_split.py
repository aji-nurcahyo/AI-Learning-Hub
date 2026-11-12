"""
Script 02: Stratified Splitting vs Random Splitting
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan mengapa Stratified Splitting krusial pada dataset
yang mengalami ketidakseimbangan kelas (class imbalance).
Perbandingan dilakukan antara Random Split dan Stratified Split
pada target 'failure' (~10.8% positif).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def compare_single_split(df: pd.DataFrame, test_ratio: float = 0.20):
    print("--- 1. SINGLE TRAIN/TEST SPLIT: RANDOM VS STRATIFIED ---")
    y = df["failure"]
    global_prop = y.mean() * 100
    print(f"Proporsi Global Target Positif (Failure=1): {y.sum()}/{len(y)} ({global_prop:.2f}%)\n")
    
    seeds = [7, 42, 101, 2024]
    results = []
    
    for s in seeds:
        # Random split (stratify=None)
        _, _, y_train_rand, y_test_rand = train_test_split(
            df, y, test_size=test_ratio, random_state=s, stratify=None
        )
        # Stratified split (stratify=y)
        _, _, y_train_strat, y_test_strat = train_test_split(
            df, y, test_size=test_ratio, random_state=s, stratify=y
        )
        
        results.append({
            "Seed": s,
            "Rand Train (%)": f"{y_train_rand.mean()*100:.2f}%",
            "Rand Test (%)": f"{y_test_rand.mean()*100:.2f}%",
            "Strat Train (%)": f"{y_train_strat.mean()*100:.2f}%",
            "Strat Test (%)": f"{y_test_strat.mean()*100:.2f}%",
            "Rand Test Pos": y_test_rand.sum(),
            "Strat Test Pos": y_test_strat.sum(),
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print()


def compare_kfold(df: pd.DataFrame, n_splits: int = 5):
    print(f"--- 2. {n_splits}-FOLD CROSS-VALIDATION: STANDARD KFOLD VS STRATIFIED KFOLD ---")
    y = df["failure"].values
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    kf_props = []
    skf_props = []
    
    print(f"{'Fold':<6} | {'K-Fold Pos':<12} | {'K-Fold Prop (%)':<16} | {'Strat-KFold Pos':<16} | {'Strat-KFold Prop (%)':<20}")
    print("-" * 75)
    
    for i, ((_, test_kf), (_, test_skf)) in enumerate(zip(kf.split(df, y), skf.split(df, y))):
        pos_kf = y[test_kf].sum()
        prop_kf = (pos_kf / len(test_kf)) * 100
        kf_props.append(prop_kf)
        
        pos_skf = y[test_skf].sum()
        prop_skf = (pos_skf / len(test_skf)) * 100
        skf_props.append(prop_skf)
        
        print(f"Fold {i+1:<2} | {pos_kf:<12} | {prop_kf:<16.2f} | {pos_skf:<16} | {prop_skf:<20.2f}")
        
    print("-" * 75)
    print(f"Standar Deviasi Proporsi Kelas Positif K-Fold     : {np.std(kf_props):.4f}%")
    print(f"Standar Deviasi Proporsi Kelas Positif Strat-KFold : {np.std(skf_props):.4f}%")
    print("\nKesimpulan: StratifiedKFold menjaga varians proporsi kelas mendekati nol antar fold.")


def main():
    print("=" * 75)
    print("DEMONSTRASI STRATIFIED SPLIT VS RANDOM SPLIT")
    print("=" * 75)
    df = load_telemetry_data()
    compare_single_split(df)
    compare_kfold(df)
    print("=" * 75)


if __name__ == "__main__":
    main()

