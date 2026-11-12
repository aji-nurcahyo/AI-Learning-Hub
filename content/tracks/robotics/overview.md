# Track E — Robotics & Intelligent Systems: Overview, Roadmap & Studi Kasus

**Fokus Jalur:** Agen cerdas (*Intelligent Agents*), pengambilan keputusan otonom, algoritma pencarian heuristik, persepsi visual, perluasan *State Estimation*, fusi multi-sensor, serta *Reinforcement Learning*.

---

## 1. Pengantar Jalur Spesialisasi
Jalur **Robotics & Intelligent Systems** memadukan AI simbolik (pencarian A*, logika, representasi keadaan) dengan pembelajaran persepsi visual dan *Reinforcement Learning*. Seluruh praktikum berlangsung aman menggunakan simulator **GridWorld** serta data sintetis di CPU lokal tanpa mewajibkan platform robotik fisik. Konsep **State Estimation** (estimasi keadaan dari pembacaan sensor bising) diperkenalkan sebagai perluasan domain terapan atas fondasi probabilitas (`Chapter 06`) dan model agen (`Chapter 03`).

## 2. Roadmap Pembelajaran Terkurasi
1. **Tahap Fondasi:** **Chapter 03** (Intelligent Agents & PEAS), **Chapter 04** (Uninformed & Informed Search A*), **Chapter 05** (Knowledge & Reasoning), **Chapter 06** (Probabilitas Bayes & Aljabar Linear).
2. **Tahap Inti Persepsi & Kontrol:** **Chapter 11** (Intro ML), **Chapter 21 & 23–24** (Neural Networks & CNN PyTorch), **Chapter 26** (CV Applications), **Chapter 35** (Reinforcement Learning).
3. **Tahap Lanjutan Integrasi & Keselamatan:** **Chapter 34** (Tool-Using Agents & FSM), **Chapter 37** (Edge AI), **Chapter 38** (Multimodal Fusion), **Chapter 39** (AI Safety & Human Oversight).

## 3. Studi Kasus Domain Robotika Cerdas

### Studi Kasus 1: Navigasi Gudang Otomatis Bebas Tabrakan (*GridWorld Warehouse*)
- **Tantangan:** Peta rintangan statis bisa berubah jika kardus jatuh menutup lorong.
- **Pipeline:** Peta graf A* (`Chapter 04` & `robotics_navigation_lab.py`) untuk perencanaan jalur terpendek (`8 langkah`) dipadukan dengan kebijakan *Q-Learning* (`Chapter 35`) serta verifikasi keselamatan aksi sebelum gerak (*Chapter 39*).

### Studi Kasus 2: Persepsi Rintangan Visual + Sensor Jarak Ultrasonik
- **Tantangan:** Kamera terganggu bayangan gelap, sementara sensor ultrasonik tetap mendeteksi jarak pantul.
- **Solusi:** Arsitektur *Late/Hybrid Fusion* (`Chapter 38`) agar keputusan berhenti aman tetap bekerja saat satu modalitas mengalami degradasi.

