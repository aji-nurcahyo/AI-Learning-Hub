# Chapter 35: Reinforcement Learning Fundamentals — Code Catalog

Koleksi 16 script Python mandiri, deterministik, dan berjalan 100% pada CPU lokal menggunakan simulator `GridWorld` mandiri (NumPy, Matplotlib, PyTorch) tanpa dependensi eksternal berat.

## Daftar Script

1. `rl_foundations.py`: Siklus interaksi *Agent-Environment*, pembentukan trajektori episode, dan evaluasi kebijakan acak dasar.
2. `gridworld_environment.py`: Simulator 2D `GridWorld` terkonfigurasi dengan dinamika transisi deterministik maupun stokastik (`slip_prob`), rintangan, jebakan bahaya, dan visualisasi ASCII.
3. `mdp_transitions.py`: Formalisasi tuple MDP $(S, A, P, R, \gamma)$ dan verifikasi aksioma probabilitas $\sum_{s'} P(s' \mid s, a) = 1.0$.
4. `discounted_returns.py`: Penghitungan *discounted return* $G_t$ rekursif dan analisis pengaruh faktor diskonto $\gamma \in \{0.0, 0.5, 0.9, 0.99\}$.
5. `value_functions.py`: Penghitungan *State Value Function* $V^\pi(s)$ dan *Action Value Function* $Q^\pi(s, a)$ serta pembuktian identitas $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$.
6. `bellman_equations.py`: Verifikasi numerik perbedaan *Bellman Expectation Equation* dan *Bellman Optimality Equation* beserta contoh perhitungan manual.
7. `policy_evaluation.py`: Algoritma *Iterative Policy Evaluation* berbasis pemrograman dinamis hingga konvergensi toleransi $\theta$.
8. `value_iteration.py`: Algoritma *Value Iteration* mencari nilai optimal $V^*(s)$ dan mengekstraksi kebijakan optimal $\pi^*(s)$.
9. `policy_iteration.py`: Algoritma *Policy Iteration* (evaluasi + perbaikan kebijakan) beserta verifikasi kesetaraan nilai terhadap *Value Iteration*.
10. `monte_carlo_returns.py`: Estimasi bebas model (*model-free*) *First-Visit Monte Carlo* memprediksi $V^\pi(s)$ dari episode sampel lengkap.
11. `td_learning.py`: Pembelajaran *Temporal-Difference TD(0)* secara online menggunakan *bootstrapping* dan perbandingan RMSE terhadap Monte Carlo.
12. `q_learning.py`: Algoritma kontrol *Off-Policy Q-Learning* lengkap dengan contoh hitungan manual step-by-step, pelatihan GridWorld, dan evaluasi kebijakan serakah ($\epsilon = 0$).
13. `epsilon_greedy.py`: Perbandingan jadwal eksplorasi (*Pure Greedy*, *Constant*, *Linear Decay*, *Exponential Decay*) terhadap cakupan state dan konvergensi.
14. `q_table_visualization.py`: Pencetakan tabel matriks Q-Table lengkap, panah kebijakan optimal ASCII, dan penyimpanan heatmap Matplotlib (`q_table_heatmap.png`).
15. `policy_gradient_reinforce.py`: Implementasi algoritme *REINFORCE (Monte Carlo Policy Gradient)* berbasis PyTorch dengan pemilihan aksi *softmax* dan *baseline* pengurang varians.
16. `rl_policy_comparison.py`: Benchmark komparatif 4 paradigma kebijakan (*Random*, *Value Iteration*, *Q-Learning*, *REINFORCE*) pada seed evaluasi identik stokastik.

## Cara Menjalankan

```bash
# Jalankan contoh dasar Q-Learning
python content/level-4-advanced-ai/chapter-35/code/q_learning.py

# Jalankan benchmark perbandingan kebijakan RL
python content/level-4-advanced-ai/chapter-35/code/rl_policy_comparison.py
```

