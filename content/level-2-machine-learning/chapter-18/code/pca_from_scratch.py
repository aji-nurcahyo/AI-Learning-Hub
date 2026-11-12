"""Principal Component Analysis (PCA) dari Dasar Menggunakan NumPy Murni
Diverifikasi presisi terhadap implementasi Scikit-Learn.
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAFromScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        N = X.shape[0]
        # 1. Menghitung rata-rata empiris dan memusatkan data (centering)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Menghitung matriks kovarians sampel (N - 1 pembagi Bessel)
        cov_matrix = np.dot(X_centered.T, X_centered) / (N - 1)

        # 3. Dekomposisi nilai eigen dan vektor eigen (eigh untuk matriks simetris)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Mengurutkan nilai eigen dari terbesar ke terkecil
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Menyimpan komponen utama terpilih (baris sebagai komponen seperti Scikit-Learn)
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.components_ = eigenvectors[:, :self.n_components].T

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def run_pca_scratch():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "signal_strength_dbm",
        "packet_loss_pct",
        "latency_ms",
        "temperature_c",
        "voltage_v",
        "energy_consumption_mah"
    ]
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PCA From Scratch
    pca_scratch = PCAFromScratch(n_components=2)
    Z_scratch = pca_scratch.fit_transform(X_scaled)

    # 2. PCA Scikit-Learn
    pca_sklearn = PCA(n_components=2, random_state=42)
    Z_sklearn = pca_sklearn.fit_transform(X_scaled)

    # Menyelaraskan ambiguitas tanda (+/-) antar vektor eigen
    for col in range(2):
        if np.dot(pca_scratch.components_[col], pca_sklearn.components_[col]) < 0:
            pca_scratch.components_[col] *= -1
            Z_scratch[:, col] *= -1

    diff_components = np.max(np.abs(pca_scratch.components_ - pca_sklearn.components_))
    diff_projection = np.max(np.abs(Z_scratch - Z_sklearn))
    diff_variance_ratio = np.max(np.abs(pca_scratch.explained_variance_ratio_ - pca_sklearn.explained_variance_ratio_))

    print("======================================================================")
    print("VERIFIKASI PCA FROM SCRATCH VS SCIKIT-LEARN")
    print("======================================================================")
    print(f"Explained Variance Ratio Scratch: {pca_scratch.explained_variance_ratio_}")
    print(f"Explained Variance Ratio Sklearn: {pca_sklearn.explained_variance_ratio_}")
    print(f"Total Varians Terjelaskan (k=2) : {np.sum(pca_scratch.explained_variance_ratio_)*100:.2f}%")
    print(f"Maksimum Selisih Komponen       : {diff_components:.8e}")
    print(f"Maksimum Selisih Proyeksi (Z)   : {diff_projection:.8e}")
    print(f"Maksimum Selisih Rasio Varians  : {diff_variance_ratio:.8e}")
    print("======================================================================")

    assert diff_components < 1e-6, "Komponen utama berbeda secara signifikan!"
    assert diff_projection < 1e-6, "Proyeksi data berbeda secara signifikan!"
    assert diff_variance_ratio < 1e-6, "Rasio varians berbeda secara signifikan!"
    print("STATUS: VERIFIKASI MATEMATIS IDENTIK LULUS (Perbedaan < 1e-6)")

if __name__ == "__main__":
    run_pca_scratch()

