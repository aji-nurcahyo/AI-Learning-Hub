# Chapter 34: AI Agents & Tool Integration — Code Catalog

Koleksi script Python mandiri, deterministik, dan dapat dijalankan secara offline 100% pada CPU untuk mempelajari arsitektur AI Agent, integrasi alat (*tool calling*), dan tata kelola keselamatan.

## Daftar Script

1. `agent_architecture.py`: Komponen dasar arsitektur AI Agent (Controller, Policy, Tool Registry, State, Execution Engine, Audit Logger).
2. `tool_registry.py`: Registri alat terpusat dengan definisi skema parameter JSON Schema dan klasifikasi risiko (`READ_ONLY` vs `STATE_CHANGING`).
3. `tool_schema_validation.py`: Validator skema parameter masukan untuk memastikan kesesuaian tipe data dan parameter wajib (*required fields*).
4. `tool_calling_demo.py`: Siklus hidup eksekusi pemanggilan alat (*tool calling lifecycle*) dengan penanganan eksepsi yang terisolasi.
5. `agent_state.py`: Finite State Machine (FSM) yang mengatur transisi status eksekusi agen dan pembatasan kuota langkah (*step budget*).
6. `planning_loop.py`: Siklus penalaran dan tindakan ReAct (*Thought -> Action -> Observation -> Thought*) dengan pencegahan perulangan tak hingga (*infinite loop*).
7. `human_approval_gate.py`: Gerbang persetujuan manusia (*human-in-the-loop*) yang menangguhkan aksi berisiko tinggi hingga disetujui secara eksplisit.
8. `permission_manager.py`: Manajemen izin berbasis peran (RBAC) dan penegakan prinsip hak akses terendah (*least privilege*).
9. `tool_error_handling.py`: Penanganan kesalahan waktu eksekusi (pembagian nol, parameter hilang, tool tidak terdaftar) dan pemberian petunjuk koreksi diri.
10. `audit_logging.py`: Pencatatan jejak audit terstruktur yang tahan manipulasi dengan verifikasi rantai hash kriptografis SHA-256.
11. `mcp_concepts_demo.py`: Konseptualisasi Model Context Protocol (MCP) berbasis pesan JSON-RPC 2.0 (`tools/list`, `tools/call`, `resources/read`).
12. `supervised_agent.py`: Proyek acuan: Asisten AI Pembelajaran Terkendali (*Supervised AI Learning Assistant*) dengan alat baca lokal dan kalkulator.
13. `agent_evaluation.py`: Suite pengujian benchmark evaluasi agen mengukur *task success rate*, akurasi seleksi alat, dan kepatuhan izin.
14. `safe_agent_workflow.py`: Alur kerja agen defensif yang mensterilkan muatan *indirect prompt injection* pada keluaran alat dan membatasi eksekusi dalam sandbox aman.

## Cara Menjalankan

Seluruh script dapat dijalankan secara mandiri melalui virtual environment:

```bash
# Menjalankan reference project agen terarah
python content/level-4-advanced-ai/chapter-34/code/supervised_agent.py

# Menjalankan benchmark evaluasi agen
python content/level-4-advanced-ai/chapter-34/code/agent_evaluation.py

# Menjalankan demonstrasi konsep Model Context Protocol (MCP)
python content/level-4-advanced-ai/chapter-34/code/mcp_concepts_demo.py
```

