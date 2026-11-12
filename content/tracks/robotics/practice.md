# Track E — Robotics & Intelligent Systems: Applied Laboratories

Delapan laboratorium integratif yang menjembatani pencarian jalur klasik, stimulasi GridWorld Q-Learning, persepsi visual, dan fusi multi-sensor di CPU lokal.

---

## 1. Lab Domain Navigasi Otonom A* vs Q-Learning (Run Langsung)
```bash
python content/tracks/robotics/labs/robotics_navigation_lab.py
```
- **Output Terverifikasi:** Membandingkan perencanaan jalur optimal **A* (`8 steps`)** dan rollout kebijakan terlatih **Q-Learning (`8 steps`)** dalam GridWorld 5x5 berpenghalang secara deterministik di CPU.

---

## 2. Daftar Delapan Laboratorium Terintegrasi

1. **Lab 1 — Search-Based Path Planning (Chapter 04):** `bfs_search.py`.
2. **Lab 2 — A* Navigation (Chapter 04 & Track E):** `astar_search.py` & `robotics_navigation_lab.py`.
3. **Lab 3 — State Representation (Chapter 03):** `04_grid_state_space.py`.
4. **Lab 4 — GridWorld Simulation (Chapter 35):** `gridworld_environment.py`.
5. **Lab 5 — Q-Learning (Chapter 35 & Track E):** `q_learning.py` & `robotics_navigation_lab.py`.
6. **Lab 6 — Policy Evaluation (Chapter 35):** `policy_evaluation.py`.
7. **Lab 7 — Computer Vision Perception (Chapter 26):** `computer_vision_pipeline.py`.
8. **Lab 8 — Multi-Sensor Decision Making (Chapter 38):** `cross_attention_fusion.py`.

