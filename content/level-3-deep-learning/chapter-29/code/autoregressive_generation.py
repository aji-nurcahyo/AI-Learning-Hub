"""
Chapter 29 - Large Language Models & Generative AI
Script 4: Autoregressive Generation Loop
Mendemonstrasikan siklus pembangkitan autoregresif langkah-demi-langkah:
Konteks -> Prediksi Token Baru -> Penambahan ke Konteks -> Evaluasi Ulang.
"""

from typing import List, Dict
import torch
import torch.nn.functional as F


class MockAutoregressiveLLM:
    """
    Mock LLM untuk mendemonstrasikan mekanisme pembangkitan autoregresif.
    Memiliki aturan transisi n-gram probabilitas kontekstual.
    """
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}
        
        # Aturan transisi sintaksis sederhana
        self.transitions = {
            "pembelajaran": ["mesin", "mendalam", "berkelanjutan"],
            "mesin": ["adalah", "cabang", "berkembang"],
            "adalah": ["bidang", "fondasi", "bagian"],
            "bidang": ["kecerdasan", "komputasi", "teknologi"],
            "kecerdasan": ["buatan", "manusia", "digital"],
            "buatan": ["modern", "<eos>", "yang"],
            "modern": ["saat", "ini", "<eos>"],
            "ini": ["<eos>", "sangat", "pesat"]
        }

    def predict_next_token_probs(self, context_ids: List[int]) -> torch.Tensor:
        """Menghasilkan probabilitas token berikutnya berdasarkan token terakhir."""
        last_token = self.id2token[context_ids[-1]]
        probs = torch.zeros(len(self.vocab))
        
        candidates = self.transitions.get(last_token, ["<eos>"])
        for word in candidates:
            if word in self.token2id:
                probs[self.token2id[word]] = 1.0 / len(candidates)
        
        if probs.sum() == 0:
            probs[self.token2id["<eos>"]] = 1.0
        return probs


def autoregressive_generate(
    model: MockAutoregressiveLLM,
    prompt: str,
    max_new_tokens: int = 10,
    stop_token: str = "<eos>"
) -> List[str]:
    """Siklus autoregresif utama."""
    generated_tokens = prompt.strip().lower().split()
    context_ids = [model.token2id.get(w, model.token2id["<unk>"]) for w in generated_tokens]
    stop_id = model.token2id[stop_token]

    print(f"\n[MEMULAI SIKLUS AUTOREGRESIF]")
    print(f"Prompt Masukan: '{prompt}' (Panjang: {len(context_ids)} token)")

    for step in range(1, max_new_tokens + 1):
        probs = model.predict_next_token_probs(context_ids)
        # Pilih token probabilitas tertinggi (Greedy)
        next_token_id = torch.argmax(probs).item()
        next_token = model.id2token[next_token_id]
        prob_val = probs[next_token_id].item()

        print(f"Langkah {step:02d}: Context window '{' '.join([model.id2token[i] for i in context_ids])}' "
              f"-> Prediksi: '{next_token}' (P = {prob_val:.2f})")

        # Cek kondisi henti
        if next_token_id == stop_id:
            print(f"--> Menemukan token henti [{stop_token}]. Pembangkitan selesai.")
            break

        # Umpankan kembali token baru ke dalam konteks
        context_ids.append(next_token_id)

    final_text = " ".join([model.id2token[i] for i in context_ids])
    return final_text


def main():
    print("=" * 70)
    print("CHAPTER 29: AUTOREGRESSIVE GENERATION MECHANICS")
    print("=" * 70)

    vocab = [
        "<pad>", "<unk>", "<bos>", "<eos>",
        "pembelajaran", "mesin", "mendalam", "berkelanjutan",
        "adalah", "cabang", "berkembang", "bidang", "fondasi",
        "bagian", "kecerdasan", "komputasi", "teknologi",
        "buatan", "manusia", "digital", "modern", "saat", "ini", "sangat", "pesat"
    ]

    model = MockAutoregressiveLLM(vocab)
    prompt = "pembelajaran mesin"
    
    result = autoregressive_generate(model, prompt=prompt, max_new_tokens=8)
    
    print("-" * 70)
    print(f"Hasil Akhir Teks Pembangkitan:\n\"{result}\"")
    print("-" * 70)

    print("\n[PENJELASAN ARSITEKTUR]")
    print("1. Pembangkitan autoregresif bersifat rekursif: y_t = LLM(y_1, ..., y_{t-1}).")
    print("2. Setiap kata baru yang dihasilkan memperbesar konteks masukan untuk langkah berikutnya.")
    print("3. Pembangkitan berhenti saat model memprediksi token khusus <eos> atau mencapai batas panjang.")


if __name__ == "__main__":
    main()

