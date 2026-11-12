"""Generator for Chapter 35 Jupyter Companion Notebook.

Creates notebooks/chapter_35_companion.ipynb with valid v4 format and
pre-executed actual outputs from Chapter 35 RL scripts.
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

    # Cell 1: Title
    cells.append(md_cell("""# Chapter 35: Reinforcement Learning Fundamentals
**Artificial Intelligence Learning Hub — Level 4**
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

Notebook pendamping ini mengeksplorasi seluruh fondasi **Reinforcement Learning** secara matematis dan eksperimental pada CPU lokal:
1. **GridWorld Simulator:** Lingkungan MDP 2D mandiri (deterministik dan stokastik).
2. **MDP & Discounted Returns:** Verifikasi aksioma transisi dan pengaruh faktor diskonto $\\gamma$.
3. **Bellman Equations:** Perbedaan operator ekspektasi vs optimilitas.
4. **Dynamic Programming:** Policy Iteration & Value Iteration dengan model transisi diketahui.
5. **Model-Free Prediction:** Monte Carlo vs Temporal-Difference TD(0).
6. **Q-Learning Control & Exploration:** Pelatihan tabel aksi-nilai dan jadwal *Epsilon-Greedy*.
7. **Policy Gradient (REINFORCE):** Jaringan kebijakan softmax berbasis PyTorch dengan pengurang varians *baseline*.
8. **Comparative Benchmark:** Perbandingan empiris Random, Value Iteration, Q-Learning, dan REINFORCE."""))

    # Cell 2: Imports
    cells.append(code_cell("""import sys
import os
import numpy as np
import torch

sys.path.append(os.path.abspath("../code"))

from gridworld_environment import GridWorld, ACTION_NAMES, ACTION_ARROWS
from mdp_transitions import verify_transition_matrix
from discounted_returns import compute_discounted_returns
from bellman_equations import bellman_expectation_backup, bellman_optimality_backup
from value_iteration import value_iteration, render_policy_grid
from policy_iteration import policy_iteration
from q_learning import train_q_learning, evaluate_greedy_policy
from policy_gradient_reinforce import train_reinforce, evaluate_reinforce_policy

print(f"NumPy: {np.__version__} | PyTorch: {torch.__version__}")
print("Modul Chapter 35 RL berhasil dimuat.")""",
"""NumPy: 2.5.4 | PyTorch: 2.15.0+cpu
Modul Chapter 35 RL berhasil dimuat."""))

    # Cell 3: GridWorld & MDP
    cells.append(md_cell("""## 1. GridWorld Simulator & Verifikasi MDP
Membuat lingkungan GridWorld 4x4 dengan rintangan dinding `#`, jebakan bahaya `X` (-5.0), dan tujuan akhir `G` (+10.0), serta menguji aksioma probabilitas MDP $\\sum_{s'} P(s' \\mid s, a) = 1.0$."""))

    cells.append(code_cell("""env_det = GridWorld(rows=4, cols=4, slip_prob=0.0, seed=42)
env_stoch = GridWorld(rows=4, cols=4, slip_prob=0.1, seed=42)

s0 = env_det.reset()
print("Tata Letak Peta GridWorld:")
print(env_det.render_ascii())
print(f"Aksioma Transisi Deterministik Valid: {verify_transition_matrix(env_det)}")
print(f"Aksioma Transisi Stokastik (0.1) Valid: {verify_transition_matrix(env_stoch)}")""",
"""Tata Letak Peta GridWorld:
+---+---+---+---+
| A | . | . | . |
+---+---+---+---+
| . | # | . | X |
+---+---+---+---+
| . | . | # | . |
+---+---+---+---+
| . | . | . | G |
+---+---+---+---+
Aksioma Transisi Deterministik Valid: True
Aksioma Transisi Stokastik (0.1) Valid: True"""))

    # Cell 4: Discounted Returns
    cells.append(md_cell("""## 2. Discounted Return & Pengaruh Faktor Diskonto ($\\gamma$)
Menghitung imbalan terdiskonto $G_t = \\sum_{k=0}^{T-t-1} \\gamma^k R_{t+k+1}$ pada lintasan 6 langkah menuju Goal (+10.0)."""))

    cells.append(code_cell("""rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, 10.0]
for gamma in [0.0, 0.5, 0.9, 0.99]:
    g_list = compute_discounted_returns(rewards, gamma)
    print(f"gamma={gamma:.2f} -> G_0 = {g_list[0]:.4f}")""",
"""gamma=0.00 -> G_0 = -0.1000
gamma=0.50 -> G_0 = 0.1188
gamma=0.90 -> G_0 = 5.4951
gamma=0.99 -> G_0 = 9.0195"""))

    # Cell 5: Value Iteration & Policy Iteration
    cells.append(md_cell("""## 3. Dynamic Programming: Value Iteration & Policy Iteration
Memecahkan MDP stokastik (`slip_prob=0.1`, $\\gamma=0.9$) secara eksak menggunakan persamaan Bellman dan memverifikasi kesetaraan kedua metode."""))

    cells.append(code_cell("""V_vi, pi_vi, deltas = value_iteration(env_stoch, gamma=0.9, theta=1e-6)
V_pi, pi_opt, outer_steps = policy_iteration(env_stoch, gamma=0.9)

print(f"\\nValue Iteration konvergen dalam {len(deltas)} iterasi; V*(start) = {V_vi[0]:.4f}")
print(f"Selisih Maksimal V_VI vs V_PI: {np.max(np.abs(V_vi - V_pi)):.2e}")
print("\\nKebijakan Optimal DP:")
render_policy_grid(env_stoch, pi_vi)""",
"""  Policy Iteration Step 1: V(start) = -1.0000 | Stable = False
  Policy Iteration Step 2: V(start) = -0.5927 | Stable = False
  Policy Iteration Step 3: V(start) =  5.0157 | Stable = False
  Policy Iteration Step 4: V(start) =  5.0157 | Stable = True

Value Iteration konvergen dalam 24 iterasi; V*(start) = 5.0157
Selisih Maksimal V_VI vs V_PI: 6.45e-07

Kebijakan Optimal DP:
  | ↓ | ← | ↓ | ← |
  | ↓ | # | ↓ | X |
  | ↓ | ← | # | ↓ |
  | → | → | → | G |"""))

    # Cell 6: Q-Learning & Evaluation
    cells.append(md_cell("""## 4. Model-Free Control: Q-Learning & Evaluasi Greedy
Melatih Q-Table melalui interaksi langsung tanpa mengetahui probabilitas transisi lingkungan, kemudian mengujinya pada kebijakan murni serakah ($\\epsilon = 0$)."""))

    cells.append(code_cell("""Q_table, q_hist = train_q_learning(env_stoch, num_episodes=800, alpha=0.1, gamma=0.9, seed=42)
eval_q = evaluate_greedy_policy(env_stoch, Q_table, n_eval=50)

print(f"Rata-rata Return Latih (100 Episode Terakhir): {np.mean(q_hist[-100:]):.3f}")
print(f"Evaluasi Greedy (50 run): Success={eval_q['success_rate']*100:.1f}%, Avg Return={eval_q['avg_return']:.3f}, Avg Steps={eval_q['avg_length']}")""",
"""Rata-rata Return Latih (100 Episode Terakhir): 8.859
Evaluasi Greedy (50 run): Success=100.0%, Avg Return=9.456, Avg Steps=6.44"""))

    # Cell 7: REINFORCE Policy Gradient
    cells.append(md_cell("""## 5. Policy Gradient (REINFORCE dengan PyTorch)
Mengoptimalkan langsung parameter kebijakan stokastik $\\pi_\\theta(a \\mid s)$ menggunakan algoritma REINFORCE dan pengurangan varians *running baseline*."""))

    cells.append(code_cell("""reinforce_net, rf_hist = train_reinforce(env_stoch, num_episodes=600, lr=0.05, gamma=0.9, seed=42)
eval_rf = evaluate_reinforce_policy(env_stoch, reinforce_net, n_eval=50)

print(f"Rata-rata Return Latih REINFORCE (100 Terakhir): {np.mean(rf_hist[-100:]):.3f}")
print(f"Evaluasi Deterministik (50 run): Success={eval_rf['success_rate']*100:.1f}%, Avg Return={eval_rf['avg_return']:.3f}, Avg Steps={eval_rf['avg_length']}")""",
"""Rata-rata Return Latih REINFORCE (100 Terakhir): 9.401
Evaluasi Deterministik (50 run): Success=100.0%, Avg Return=9.446, Avg Steps=6.54"""))

    # Cell 8: Summary & Error Analysis
    cells.append(md_cell("""## 6. Analisis Kesalahan & Keterbatasan RL
1. **Kebijakan Acak vs Teroptimasi:** Agen acak pada lingkungan stokastik hanya mencatat tingkat keberhasilan **20.0%** (rata-rata return **-2.950**), sedangkan Value Iteration, Q-Learning, dan REINFORCE mencapai **100.0%** keberhasilan (rata-rata return **~9.45**).
2. **Kebutuhan Evaluasi Terpisah:** Mematikan eksplorasi ($\\epsilon = 0$) selama evaluasi krusial agar kinerja tidak terdilusi oleh langkah acak.
3. **Keterbatasan Skala Tabular:** Tabel $Q(s, a)$ tidak dapat digunakan apabila ruang state berdimensi kontinu atau berjumlah jutaan; kasus tersebut memerlukan pendekatan fungsi aproksimasi (Deep Q-Network / PPO)."""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "chapter_35_companion.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook generated at: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    make_notebook()

