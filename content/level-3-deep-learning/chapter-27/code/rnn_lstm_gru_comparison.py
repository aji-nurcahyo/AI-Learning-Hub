"""
Chapter 27 - Natural Language Processing Fundamentals
Script 12: Sequence Modeling Benchmarking (Simple RNN vs LSTM vs GRU)

Mengeksekusi eksperimen perbandingan terstandar antara tiga keluarga model sekuensial:
Simple RNN, LSTM, dan GRU pada tugas klasifikasi sekuens teks berbobot CPU:
- Mengukur jumlah parameter model, kecepatan inferensi/pelatihan, konvergensi loss.
- Mengamati dinamika norma gradien (Gradient Norm) untuk mendeteksi vanishing gradients.
- Menyimpan log hasil eksperimen ke 'experiments/results_ch27.json'.
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SequenceClassifier(nn.Module):
    """Model terpadu yang dapat diinstansiasi sebagai RNN, LSTM, atau GRU."""

    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.model_type = model_type.upper()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if self.model_type == "RNN":
            self.seq_layer = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        elif self.model_type == "LSTM":
            self.seq_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.model_type == "GRU":
            self.seq_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Tipe model '{model_type}' tidak didukung!")

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        if self.model_type == "LSTM":
            out, (h_n, c_n) = self.seq_layer(emb)
        else:
            out, h_n = self.seq_layer(emb)
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        return logits


def generate_synthetic_sequence_data(num_samples: int = 120, seq_len: int = 10, vocab_size: int = 40):
    """Menghasilkan dataset sekuens teks sintetis deterministik untuk benchmarking CPU."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Pola: jika kata kunci tertentu (token > 30) muncul di awal/tengah, label = 1, jika tidak label = 0
    X = torch.randint(low=1, high=vocab_size, size=(num_samples, seq_len), dtype=torch.long)
    y = (torch.sum(X[:, :seq_len // 2] > 30, dim=1) > 0).long()

    # Partisi Train (80 sampel) dan Val (40 sampel)
    train_data = TensorDataset(X[:80], y[:80])
    val_data = TensorDataset(X[80:], y[80:])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    return train_loader, val_loader


def train_and_evaluate(
    model_type: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 8,
):
    torch.manual_seed(42)
    model = SequenceClassifier(model_type, vocab_size, embedding_dim, hidden_dim, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    param_count = sum(p.numel() for p in model.parameters())
    history = {"train_loss": [], "grad_norm": []}

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Hitung norma gradien global untuk memantau aliran gradien
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0).item()
            total_grad_norm += grad_norm

            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        avg_grad = total_grad_norm / batches
        history["train_loss"].append(round(avg_loss, 4))
        history["grad_norm"].append(round(avg_grad, 4))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Evaluasi Akurasi pada Validation Set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            preds = model(batch_x).argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    val_acc = (correct / total) * 100.0

    return {
        "model_type": model_type,
        "parameter_count": param_count,
        "training_time_ms": round(elapsed_ms, 2),
        "val_accuracy": round(val_acc, 2),
        "final_loss": history["train_loss"][-1],
        "avg_grad_norm": round(float(np.mean(history["grad_norm"])), 4),
        "history": history,
    }


def main():
    print("=== Chapter 27: Benchmarking Sequence Models (RNN vs LSTM vs GRU) ===")

    vocab_size = 40
    embedding_dim = 16
    hidden_dim = 24
    epochs = 8

    train_loader, val_loader = generate_synthetic_sequence_data(
        num_samples=120, seq_len=12, vocab_size=vocab_size
    )

    models_to_test = ["RNN", "LSTM", "GRU"]
    results = {}

    print(f"\nMenjalankan Pelatihan CPU Komparatif ({epochs} Epochs per model):")
    for m_type in models_to_test:
        print(f"  Melatih model {m_type:5s}...", end=" ", flush=True)
        res = train_and_evaluate(
            m_type, vocab_size, embedding_dim, hidden_dim, train_loader, val_loader, epochs=epochs
        )
        results[m_type] = res
        print(f"Selesai dalam {res['training_time_ms']:6.2f} ms | Akurasi Val: {res['val_accuracy']:5.2f}% | Params: {res['parameter_count']}")

    print("\n--- Ringkasan Hasil Eksperimen Komparasi ---")
    print(f"{'Model':<8} | {'Params':<8} | {'Waktu (ms)':<10} | {'Loss Akhir':<10} | {'Akurasi Val':<12} | {'Norm Gradien'}")
    print("-" * 70)
    for m_type in models_to_test:
        r = results[m_type]
        print(f"{r['model_type']:<8} | {r['parameter_count']:<8} | {r['training_time_ms']:<10.2f} | {r['final_loss']:<10.4f} | {r['val_accuracy']:<11.2f}% | {r['avg_grad_norm']}")

    # Simpan hasil ke direktori experiments
    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results_ch27.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nLog eksperimen tersimpan di: {out_file}")

    # Verifikasi assert
    assert results["RNN"]["parameter_count"] < results["GRU"]["parameter_count"] < results["LSTM"]["parameter_count"], (
        "Hirarki jumlah parameter harus: RNN < GRU < LSTM"
    )
    assert all(r["val_accuracy"] >= 50.0 for r in results.values()), "Model harus mencapai akurasi layak"

    print("Status: rnn_lstm_gru_comparison.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

