"""
Chapter 29 - Large Language Models & Generative AI
Script 10: In-Context Learning & Prompt Engineering Patterns
Mendemonstrasikan pola Zero-Shot, One-Shot, dan Few-Shot prompting
untuk tugas klasifikasi sentimen, ekstraksi entitas, dan format JSON terstruktur.
"""

import json
from typing import List, Dict


def build_zero_shot_prompt(text: str) -> str:
    """Membangun prompt zero-shot tanpa contoh demonstrasi."""
    return f"""Tugas: Klasifikasikan sentimen ulasan berikut menjadi POSITIF atau NEGATIF.
Ulasan: "{text}"
Sentimen:"""


def build_few_shot_prompt(text: str, examples: List[Dict[str, str]]) -> str:
    """Membangun prompt few-shot dengan demonstrasi terstruktur."""
    prompt = "Tugas: Klasifikasikan sentimen ulasan berikut menjadi POSITIF atau NEGATIF.\n\nContoh:\n"
    for ex in examples:
        prompt += f'Ulasan: "{ex["text"]}" -> Sentimen: {ex["label"]}\n'
    prompt += f'\nUlasan: "{text}" -> Sentimen:'
    return prompt


def build_entity_extraction_prompt(text: str) -> str:
    """Prompt ekstraksi entitas informasi spesifik."""
    return f"""Tugas: Ekstraksi nama orang, institusi, dan kota dari teks berikut.
Format keluaran wajib JSON valid dengan kunci: "orang", "institusi", "kota".

Teks: "{text}"
Keluaran JSON:"""


def build_structured_json_prompt(ticket: str) -> str:
    """Prompt terstruktur untuk penanganan tiket customer service."""
    return f"""Tugas: Analisis tiket komplain pelanggan berikut dan ekstrak ringkasan, urgensi, dan departemen.
Wajib luaran JSON murni.

Tiket: "{ticket}"
JSON Output:
{{
  "urgensi": "RENDAH|SEDANG|TINGGI",
  "departemen": "BILLING|TEKNIS|LOGISTIK",
  "inti_masalah": "string ringkas"
}}"""


def main():
    print("=" * 70)
    print("CHAPTER 29: IN-CONTEXT LEARNING (ZERO-SHOT, FEW-SHOT & STRUCTURED PROMPTS)")
    print("=" * 70)

    # 1. Zero-Shot vs Few-Shot Prompting
    review_target = "Layanan pengiriman sangat lambat, paket rusak saat tiba."
    
    few_shot_examples = [
        {"text": "Aplikasi sangat responsif dan mudah digunakan.", "label": "POSITIF"},
        {"text": "Makanan dingin dan pelayan tidak sopan.", "label": "NEGATIF"},
        {"text": "Kualitas barang melebihi ekspektasi harga murah.", "label": "POSITIF"}
    ]

    zero_shot_str = build_zero_shot_prompt(review_target)
    few_shot_str = build_few_shot_prompt(review_target, few_shot_examples)

    print("\n--- 1. POLA ZERO-SHOT PROMPT ---")
    print(zero_shot_str)

    print("\n--- 2. POLA FEW-SHOT PROMPT ---")
    print(few_shot_str)

    # 2. Ekstraksi Entitas Terstruktur
    entity_text = "Prof. Dr. Aji mengajar kecerdasan buatan di Institut Teknologi Sumatera yang berlokasi di Lampung Selatan."
    entity_prompt = build_entity_extraction_prompt(entity_text)

    print("\n--- 3. POLA PROMPT EKSTRAKSI ENTITAS ---")
    print(entity_prompt)

    # Simulasi Luaran Model yang Diharapkan
    simulated_json_output = {
        "orang": "Prof. Dr. Aji",
        "institusi": "Institut Teknologi Sumatera",
        "kota": "Lampung Selatan"
    }
    print("Simulasi Luaran Model:")
    print(json.dumps(simulated_json_output, indent=2, ensure_ascii=False))

    # 3. Tiket Customer Support Terstruktur
    ticket_text = "Sistem pembayaran saya gagal terpotong dua kali di server backend tetapi saldo tidak bertambah segera bantu!"
    ticket_prompt = build_structured_json_prompt(ticket_text)

    print("\n--- 4. POLA PROMPT SCHEMA JSON TERSTRUKTUR ---")
    print(ticket_prompt)

    simulated_ticket_output = {
        "urgensi": "TINGGI",
        "departemen": "BILLING",
        "inti_masalah": "Gagal transaksi pemotongan saldo ganda"
    }
    print("Simulasi Luaran Model:")
    print(json.dumps(simulated_ticket_output, indent=2, ensure_ascii=False))

    print("\n[PRINSIP FUNDAMENTAL IN-CONTEXT LEARNING]")
    print("1. In-Context Learning (ICL) mengubah perilaku model HANYA melalui prompt masukan saat inferensi.")
    print("2. Bobot model dasar (weights) TIDAK mengalami perubahan atau pembaruan gradien apa pun.")
    print("3. Kualitas dan format demonstrasi Few-Shot sangat menentukan kekokohan format luaran.")
    print("4. Batasan ICL: Terikat pada panjang jendela konteks (context window) dan rentan terhadap sensitivitas urutan contoh.")


if __name__ == "__main__":
    main()

