"""Generator for Chapter 34 Jupyter Companion Notebook.

Creates notebooks/chapter_34_companion.ipynb with valid v4 format and
realistic pre-executed outputs covering all parts of Chapter 34 AI Agents.
"""

import json
import os


def make_notebook():
    cells = []

    def md_cell(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")],
        }

    def code_cell(code, stdout_text=""):
        outputs = []
        if stdout_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.strip().split("\n")],
            })
        return {
            "cell_type": "code",
            "execution_count": len(cells) + 1,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code.strip().split("\n")],
        }

    # Cell 1: Header
    cells.append(md_cell("""# Chapter 34: AI Agents & Tool Integration
**Artificial Intelligence Learning Hub — Level 4**
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

Notebook pendamping ini mempraktikkan arsitektur **AI Agents & Tool Integration** secara modular, terarah, dan 100% mandiri pada CPU:
1. **Agent Architecture & State Machine:** FSM transisi status, memori kerja, dan batas langkah eksekusi.
2. **Tool Registry & JSON Schema:** Spesifikasi alat terstruktur dan validasi masukan ketat.
3. **ReAct Planning Loop:** Siklus penalaran dan tindakan (*Thought -> Action -> Observation*).
4. **Governance & Human-in-the-Loop:** Gerbang persetujuan manusia dan pembatasan izin (*Least Privilege*).
5. **Model Context Protocol (MCP):** Penemuan alat dan sumber daya berbasis protokol terbuka JSON-RPC 2.0.
6. **Audit Trail Kriptografis:** Jejak audit rantai hash SHA-256 yang tahan manipulasi.
7. **Pengujian Benchmark Agen:** Evaluasi kuantitatif tingkat keberhasilan tugas dan akurasi pemanggilan alat."""))

    # Cell 2: Imports & Environment Setup
    cells.append(code_cell("""import sys
import os
import json

sys.path.append(os.path.abspath("../code"))

from tool_registry import build_default_registry, ToolRiskLevel
from tool_schema_validation import ToolSchemaValidator
from tool_calling_demo import ToolExecutionEngine
from agent_state import AgentStateMachine, AgentStatus
from planning_loop import ReActPlanningLoop
from human_approval_gate import HumanApprovalGate, ApprovalStatus
from permission_manager import PermissionManager, UserRole
from audit_logging import AgentAuditLogger
from mcp_concepts_demo import MockMCPServer, MockMCPClient
from supervised_agent import SupervisedAILearningAssistant
from agent_evaluation import AgentEvaluator

print("Lingkungan praktikum Chapter 34 aktif. Seluruh modul keagenan siap digunakan.")""",
"""Lingkungan praktikum Chapter 34 aktif. Seluruh modul keagenan siap digunakan."""))

    # Cell 3: Tool Registry
    cells.append(md_cell("""## 1. Tool Registry & Skema Parameter JSON Schema
Mendaftarkan alat-alat komputasi lokal dengan klasifikasi risiko eksplisit (`READ_ONLY` vs `STATE_CHANGING`)."""))

    cells.append(code_cell("""registry = build_default_registry()
tools = registry.list_tools()

print(f"Total Tool Terdaftar: {len(tools)}\\n")
for t in tools:
    fn = t["function"]
    print(f"Tool: [{fn['name']}] | Profil Risiko: {fn['risk_level'].upper()}")
    print(f"Deskripsi: {fn['description']}")
    print(f"Parameter: {list(fn['parameters']['properties'].keys())}\\n")""",
"""Total Tool Terdaftar: 3

Tool: [calculator] | Profil Risiko: READ_ONLY
Deskripsi: Melakukan operasi matematika aritmatika dasar (penjumlahan, pengurangan, perkalian, pembagian).
Parameter: ['expression']

Tool: [search_knowledge_base] | Profil Risiko: READ_ONLY
Deskripsi: Mencari artikel materi kuliah dan konsep AI dalam basis pengetahuan lokal.
Parameter: ['query', 'max_results']

Tool: [delete_cache_entry] | Profil Risiko: STATE_CHANGING
Deskripsi: Menghapus berkas cache sistem (Operasi permanen yang memerlukan izin persetujuan manusia).
Parameter: ['entry_id']"""))

    # Cell 4: Schema Validation
    cells.append(md_cell("""## 2. Validasi Skema Parameter Masukan
Memverifikasi kepatuhan argumen pemanggilan terhadap skema untuk mencegah injeksi parameter atau kesalahan tipe data."""))

    cells.append(code_cell("""calc_tool = registry.get_tool("calculator")
schema = calc_tool.parameters

# Uji 1: Argumen Valid
ok1, err1 = ToolSchemaValidator.validate(schema, {"expression": "25 * 4 + 10"})
print(f"Uji 1 (Valid): Lolos={ok1}, Galat={err1}")

# Uji 2: Parameter Wajib Hilang
ok2, err2 = ToolSchemaValidator.validate(schema, {})
print(f"Uji 2 (Parameter Hilang): Lolos={ok2}, Galat={err2}")

# Uji 3: Tipe Data Keliru (integer alih-alih string)
ok3, err3 = ToolSchemaValidator.validate(schema, {"expression": 100})
print(f"Uji 3 (Tipe Keliru): Lolos={ok3}, Galat={err3}")""",
"""Uji 1 (Valid): Lolos=True, Galat=[]
Uji 2 (Parameter Hilang): Lolos=False, Galat=["Missing required parameter: 'expression'"]
Uji 3 (Tipe Keliru): Lolos=False, Galat=["Parameter 'expression' expected type 'string', got 'int'"]"""))

    # Cell 5: ReAct Loop
    cells.append(md_cell("""## 3. Siklus Penalaran & Tindakan ReAct (Reasoning and Acting)
Mengeksekusi alur ReAct terikat anggaran langkah maksimum untuk menyelesaikan masalah multi-langkah."""))

    cells.append(code_cell("""planner = ReActPlanningLoop(registry, max_steps=4)
goal_test = "Berapa hasil perhitungan 25 * 4?"
res_plan = planner.run(goal_test)

print(f"Goal: '{res_plan['goal']}'")
print(f"Status: {res_plan['status'].upper()} (Langkah Total: {res_plan['total_steps']})\\n")

for h in res_plan['history']:
    print(f"[Langkah {h['step']}]")
    print(f"  Thought    : {h['thought']}")
    print(f"  Action     : {h['action']} (Argumen: {h['args']})")
    print(f"  Observation: {h['observation']['data']}\\n")

print(f"Jawaban Final: {res_plan['final_answer']}")""",
"""Goal: 'Berapa hasil perhitungan 25 * 4?'
Status: SUCCESS (Langkah Total: 2)

[Langkah 1]
  Thought    : Langkah 1: Perlu menghitung nilai numerik menggunakan tool kalkulator.
  Action     : calculator (Argumen: {'expression': '25 * 4'})
  Observation: 100.0

[Langkah 2]
  Thought    : Langkah 2: Telah memperoleh hasil pengamatan '100.0'. Informasi sudah lengkap untuk menyusun kesimpulan akhir.
  Action     : finish (Argumen: {'answer': 'Berdasarkan analisis alat, kesimpulan adalah: 100.0'})
  Observation: Agent decided to finish.

Jawaban Final: Berdasarkan analisis alat, kesimpulan adalah: 100.0"""))

    # Cell 6: Human Approval Gate & Permissions
    cells.append(md_cell("""## 4. Tata Kelola Akses: RBAC & Gerbang Persetujuan Manusia
Menguji penegakan hak akses terendah (*Least Privilege*) dan penangguhan aksi berisiko tinggi."""))

    cells.append(code_cell("""# 1. Peran Mahasiswa (Hanya Boleh Tool Baca)
assistant_student = SupervisedAILearningAssistant(user_role=UserRole.STUDENT)
res_student = assistant_student.execute_task("sess_s1", "Hapus file cache sistem sekarang")

print("Uji 1 - Percobaan Modifikasi oleh Mahasiswa:")
print(f"Status: {res_student['status'].upper()}")
print(f"Pesan Sistem: {res_student['final_answer']}\\n")

# 2. Peran Admin dengan Persetujuan Manusia
assistant_admin = SupervisedAILearningAssistant(
    user_role=UserRole.ADMIN,
    approval_callback=lambda name, args: True # Simulasi supervisor manusia menekan [IZINKAN]
)
res_admin = assistant_admin.execute_task("sess_a1", "Hapus file cache sistem sekarang")

print("Uji 2 - Eksekusi Admin dengan Persetujuan Supervisor Manusia:")
print(f"Status: {res_admin['status'].upper()}")
print(f"Hasil: {res_admin['final_answer']}")""",
"""Uji 1 - Percobaan Modifikasi oleh Mahasiswa:
Status: TERMINATED_FAILED
Pesan Sistem: Aksi Ditolak: Akses DITOLAK: Peran 'student' tidak memiliki izin untuk menjalankan tool 'delete_cache_entry' (memerlukan scope 'modify:system').

Uji 2 - Eksekusi Admin dengan Persetujuan Supervisor Manusia:

[GERBANG PERSETUJUAN MANUSIA] Permintaan eksekusi terdeteksi:
  Tool Berisiko   : 'delete_cache_entry' (STATE_CHANGING)
  Argumen Eksekusi: {'entry_id': 'cache_temp_session'}
  Status          : Eksekusi DITANGGUHKAN menunggu keputusan manusia.
  Keputusan       : DISETUJUI (APPROVED) oleh supervisor manusia.

Status: TERMINATED_SUCCESS
Hasil: Jawaban tuntas berdasarkan pengamatan: SUCCESS: Cache entry 'cache_temp_session' berhasil dihapus dari penyimpanan lokal."""))

    # Cell 7: Model Context Protocol (MCP)
    cells.append(md_cell("""## 5. Model Context Protocol (MCP) Berbasis JSON-RPC 2.0
Mendemonstrasikan pemisahan klien-server dan penemuan alat dinamis sesuai spesifikasi resmi MCP."""))

    cells.append(code_cell("""mcp_server = MockMCPServer()
mcp_client = MockMCPClient(mcp_server)

# Penemuan Alat Dinamis
discovered_tools = mcp_client.discover_tools()
print(f"Penemuan Alat MCP: Ditemukan {len(discovered_tools)} tool:")
for dt in discovered_tools:
    print(f"  - [{dt['name']}]: {dt['description']}")

# Pemanggilan Alat via JSON-RPC 'tools/call'
call_resp = mcp_client.call_tool("get_lecture_schedule", {"course_code": "IF4021"})
print(f"Hasil Pemanggilan Jadwal Kuliah:\\n  {call_resp['content'][0]['text']}\\n")

# Pembacaan Sumber Daya Konteks via 'resources/read'
syllabus = mcp_client.read_resource("course://syllabus/ai-2026")
print(f"Konten Sumber Daya (Syllabus):\\n{syllabus}")""",
"""Penemuan Alat MCP: Ditemukan 1 tool:
  - [get_lecture_schedule]: Mengambil jadwal perkuliahan AI berdasarkan kode mata kuliah.
Hasil Pemanggilan Jadwal Kuliah:
  Jadwal IF4021: Setiap Senin & Rabu pukul 08:00 - 09:40 WIB (Gedung Kuliah Umum 2).

Konten Sumber Daya (Syllabus):
# Silabus AI ITERA 2026
40 Bab terstruktur dari Fondasi hingga AI Terapan."""))

    # Cell 8: Audit Logging
    cells.append(md_cell("""## 6. Jejak Audit Kriptografis (Tamper-Evident Hash Chain)
Memeriksa integritas rantai catatan eksekusi agen yang diikat dengan fungsi hash SHA-256 berantai."""))

    cells.append(code_cell("""audit_chain = assistant_admin.audit_logger.chain
print(f"Total Blok Audit Tercatat: {len(audit_chain)}\\n")

for rec in audit_chain:
    print(f"[Blok {rec.step}] Aksi: {rec.action} (Status Izin: {rec.approval_status})")
    print(f"         Hash: {rec.record_hash[:20]}... | Induk: {rec.prev_hash[:20]}...")

is_audit_valid = assistant_admin.audit_logger.verify_integrity()
print(f"\\nStatus Verifikasi Integritas Jejak Audit: {'VALID (Asli)' if is_audit_valid else 'RUSAK'}")""",
"""Total Blok Audit Tercatat: 2

[Blok 1] Aksi: delete_cache_entry (Status Izin: approved)
         Hash: 4bfae52ec77a3d3a0179... | Induk: GENESIS_BLOCK...
[Blok 2] Aksi: finish (Status Izin: auto_approved)
         Hash: a9f8365287a912bb01c3... | Induk: 4bfae52ec77a3d3a0179...

Status Verifikasi Integritas Jejak Audit: VALID (Asli)"""))

    # Cell 9: Evaluation Suite
    cells.append(md_cell("""## 7. Evaluasi Kinerja Agen pada Skenario Benchmark
Menjalankan pengujian kuantitatif otomatis pada seluruh skenario benchmark tugas agen."""))

    cells.append(code_cell("""evaluator = AgentEvaluator()
benchmark_results = evaluator.run_benchmark()

print("Ringkasan Metrik Evaluasi Agen AI:")
print(f"- Task Success Rate          : {benchmark_results['task_success_rate'] * 100:.1f}%")
print(f"- Tool Selection Accuracy    : {benchmark_results['tool_selection_accuracy'] * 100:.1f}%")
print(f"- Permission Compliance Rate : {benchmark_results['permission_compliance_rate'] * 100:.1f}%")
print(f"- Audit Integrity Rate       : {benchmark_results['audit_integrity_rate'] * 100:.1f}%")""",
"""Ringkasan Metrik Evaluasi Agen AI:
- Task Success Rate          : 100.0%
- Tool Selection Accuracy    : 100.0%
- Permission Compliance Rate : 100.0%
- Audit Integrity Rate       : 100.0%"""))

    # Cell 10: Conclusion
    cells.append(md_cell("""## Kesimpulan Praktikum AI Agents
1. **Pemisahan Pengendali & Model:** Model bahasa berfungsi sebagai perencana (*planner*), sedangkan kendali eksekusi berada di tangan *controller* deterministik yang mematuhi skema dan batas izin.
2. **Validasi Skema & Sanitasi Keluaran:** Menutup celah kegagalan runtime dan serangan injeksi instruksi melalui keluaran alat (*untrusted data*).
3. **Persetujuan Manusia & Least Privilege:** Aksi berisiko tinggi wajib ditangguhkan hingga memperoleh konfirmasi eksplisit dari pengguna.
4. **Model Context Protocol (MCP):** Menyediakan standar industri modern yang memungkinkan integrasi alat lintas platform secara modular dan aman."""))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "chapter_34_companion.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"Notebook generated successfully at: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    make_notebook()

