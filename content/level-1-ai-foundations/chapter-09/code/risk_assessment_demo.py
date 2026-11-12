"""AI Ethics Risk Assessment Matrix Demonstration.

Mengimplementasikan kerangka kerja penilaian risiko etis dan dampak sosial AI
berdasarkan standar NIST AI RMF 1.0, ISO/IEC 42001, dan EU AI Act.
Menghitung Skor Risiko (Likelihood x Impact), menentukan kategori risiko,
dan mengekspor daftar risiko etika (AI Ethics Risk Register) ke format CSV.
"""

from pathlib import Path
import pandas as pd


def calculate_risk_level(score: int) -> str:
    """Mengklasifikasikan tingkat keparahan risiko berdasarkan skor total."""
    if score >= 15:
        return "TINGGI (Kritis)"
    elif score >= 8:
        return "SEDANG (Perlu Mitigasi)"
    else:
        return "RENDAH (Dapat Diterima)"


def evaluate_ai_risks() -> pd.DataFrame:
    """Mengevaluasi lima skenario implementasi sistem AI di dunia nyata."""
    risks = [
        {
            "risk_id": "RSK-01",
            "system_domain": "Sistem Seleksi Beasiswa / Rekrutmen Otomatis",
            "threat_event": "Diskriminasi algoritmik dan disparate impact pada kelompok minoritas",
            "likelihood": 4,  # Skala 1-5 (Sangat Sering)
            "impact": 5,      # Skala 1-5 (Bencana bagi hak kesetaraan & reputasi hukum)
            "existing_controls": "Hanya akurasi global tanpa pemisahan metrik keadilan kelompok",
            "proposed_mitigation": "Audit Equal Opportunity, threshold tuning kelompok, pengawasan Human-in-the-Loop",
            "governance_standard": "EU AI Act High-Risk / UU PDP No. 27/2022"
        },
        {
            "risk_id": "RSK-02",
            "system_domain": "Sistem Diagnostik Medis Radiologi Berbasis AI",
            "threat_event": "False Negative kanker langka akibat data training kurang representatif",
            "likelihood": 3,
            "impact": 5,
            "existing_controls": "Evaluasi ROC-AUC rata-rata 94%",
            "proposed_mitigation": "Data augmentation kasus langka, second-opinion dokter spesialis wajib",
            "governance_standard": "WHO Guidance on AI for Health"
        },
        {
            "risk_id": "RSK-03",
            "system_domain": "Chatbot Layanan Informasi Mahasiswa (LLM Generatif)",
            "threat_event": "Halusinasi informasi beasiswa dan kebocoran data sensitif via prompt injection",
            "likelihood": 4,
            "impact": 3,
            "existing_controls": "System prompt guardrail dasar",
            "proposed_mitigation": "Retrieval-Augmented Generation (RAG) berbasis dokumen resmi, filter input/output",
            "governance_standard": "NIST AI RMF Generative AI Profile"
        },
        {
            "risk_id": "RSK-04",
            "system_domain": "Pengawasan Kampus Cerdas (Facial Recognition)",
            "threat_event": "Pengawasan massal invasif privasi & akurasi rendah pada gender/ras tertentu",
            "likelihood": 4,
            "impact": 4,
            "existing_controls": "CCTV kampus standar",
            "proposed_mitigation": "Penilaian Dampak Privasi (DPIA), persetujuan eksplisit (consent), opt-out manual",
            "governance_standard": "Pasal 20-30 UU PDP No. 27/2022"
        },
        {
            "risk_id": "RSK-05",
            "system_domain": "Pemeliharaan Prediktif Turbin Generator Industri",
            "threat_event": "Penghapusan pencilan menyebabkan model gagal mendeteksi sinyal breakdown",
            "likelihood": 2,
            "impact": 4,
            "existing_controls": "Deteksi anomali batas Tukey IQR",
            "proposed_mitigation": "Winsorizing capping, konsultasi rutin teknisi rekayasa domain",
            "governance_standard": "ISO/IEC 42001 AI Management"
        }
    ]
    
    df = pd.DataFrame(risks)
    df["risk_score"] = df["likelihood"] * df["impact"]
    df["risk_level"] = df["risk_score"].apply(calculate_risk_level)
    
    return df


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "ai_ethics_risk_register.csv"
    
    df_risks = evaluate_ai_risks()
    df_risks.to_csv(csv_path, index=False)
    
    print("=== MATRIKS PENILAIAN RISIKO ETIKA AI (AI ETHICS RISK REGISTER) ===")
    print(f"Total Risiko Dievaluasi: {len(df_risks)}")
    print(f"Laporan tersimpan ke   : {csv_path.name}\n")
    
    for _, r in df_risks.iterrows():
        print(f"[{r['risk_id']}] {r['system_domain']}")
        print(f"  - Ancaman Etis      : {r['threat_event']}")
        print(f"  - Peluang (L)       : {r['likelihood']}/5 | Dampak (I): {r['impact']}/5")
        print(f"  - Skor Risiko (LxI) : {r['risk_score']} -> {r['risk_level']}")
        print(f"  - Rencana Mitigasi  : {r['proposed_mitigation']}")
        print(f"  - Acuan Regulasi    : {r['governance_standard']}\n")
        
    # Distribusi Tingkat Risiko
    dist = df_risks["risk_level"].value_counts()
    print("=== DISTRIBUSI KEPARAHAN RISIKO ===")
    for lvl, cnt in dist.items():
        print(f"- {lvl}: {cnt} skenario ({cnt/len(df_risks)*100:.0f}%)")


if __name__ == "__main__":
    main()

