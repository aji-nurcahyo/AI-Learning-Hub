# Track E Capstone Specification: Autonomous GridWorld Navigation with Intelligent Perception

**Jalur:** Track E — Robotics & Intelligent Systems  
**Lingkungan Wajib:** Simulator CPU Lokal (GridWorld + Citra/Sensor Sintetis — Robot fisik **tidak** diwajibkan).

---

## 1. Deliverables Wajib
1. **Environment Specification:** Definisi peta grid, rintangan, zona terlarang, dan aturan transisi deterministik/stokastik.
2. **State & Action Definitions:** Representasi ruang keadaan (termasuk observasi persepsi/keadaan bising) dan himpunan aksi legal.
3. **Search Baseline:** Implementasi jalur acuan deterministik menggunakan **A* Search** (`Chapter 04`).
4. **Reinforcement Learning Policy:** Pelatihan **Q-Learning** atau **REINFORCE** di CPU dengan kurva return & evaluasi serakah ($\epsilon = 0$).
5. **Navigation Evaluation:** Perbandingan laju keberhasilan, panjang langkah rata-rata, dan tingkat benturan melampaui 50 episode uji.
6. **Safety Constraints:** Penegakan penolakan aksi ilegal/menabrak batas dan fallback saat sinyal persepsi ambigu.
7. **Error Analysis:** Analisis kasus kegagalan kebijakan pada rintangan tersembunyi atau *reward hacking*.
8. **Documentation:** Panduan reproduksi (`seed=42`) dan batasan tegas simulasi 2D diskrit dibanding kinematika robot fisik nyata.

## 2. Rubrik Penilaian Track E (Total 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Keunggulan |
| :--- | :---: | :--- |
| **Environment, State Space & A* Baseline** | **20 poin** | Simulator bersih tanpa bug, heuristik A* valid, dan state/action terdokumentasi jelas. |
| **RL Policy & Perception Integration** | **30 poin** | Q-Learning/Policy Gradient konvergen dan integrasi persepsi/multi-sensor berjalan benar. |
| **Navigation Evaluation & Error Analysis** | **25 poin** | Evaluasi lintas episode stokastik lengkap beserta pembongkaran kasus kegagalan. |
| **Safety Constraints & Limits Disclosure** | **15 poin** | Batas keselamatan aksi ditegakkan + pengakuan jujur batas simulasi diskrit vs robot nyata. |
| **Documentation & Reproducibility** | **10 poin** | Notebook/skrip modular dapat direproduksi ulang persis dari awal. |

