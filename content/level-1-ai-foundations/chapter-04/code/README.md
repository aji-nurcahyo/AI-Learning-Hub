# Katalog Kode Praktik Chapter 04: Search Algorithms

Repositori ini memuat implementasi runnable dari 7 algoritma pencarian kecerdasan buatan klasik dan terinformasi:
1. `bfs_search.py`: Breadth-First Search
2. `dfs_search.py`: Depth-First Search
3. `depth_limited_search.py`: Depth-Limited Search (DLS)
4. `iterative_deepening_search.py`: Iterative Deepening Search (IDDFS)
5. `uniform_cost_search.py`: Uniform-Cost Search (UCS / Dijkstra)
6. `greedy_best_first_search.py`: Greedy Best-First Search
7. `astar_search.py`: A* Search
8. `compare_search_algorithms.py`: Evaluasi dan benchmarking komparatif lengkap
9. `search_framework.py`: Modul fondasi (Node, SearchResult, SearchProblem, GridNavigation, RomaniaMap)

---

## Petunjuk Menjalankan

Masuk ke direktori kode praktik Chapter 04:

```bash
cd content/level-1-ai-foundations/chapter-04/code

# Menjalankan algoritma individual
python bfs_search.py
python dfs_search.py
python depth_limited_search.py
python iterative_deepening_search.py
python uniform_cost_search.py
python greedy_best_first_search.py
python astar_search.py

# Menjalankan perbandingan benchmark otomatis
python compare_search_algorithms.py
```

---

## Hasil Benchmark Nyata (Terverifikasi Python 3.12.10)

```text
===============================================================================================
BENCHMARK: PETA GRAF BERBOBOT ROMANIA (ARAD -> BUCHAREST)
===============================================================================================
Algoritma                 | Status   | Langkah | Biaya/Jarak  | Expanded | Generated | Frontier | Waktu (ms)
-----------------------------------------------------------------------------------------------
Breadth-First Search      | SUCCESS  | 3       | 450.0        | 5        | 13        | 5        | 0.058     
Depth-First Search        | SUCCESS  | 3       | 450.0        | 4        | 10        | 5        | 0.023     
Depth-Limited (L=4)       | SUCCESS  | 3       | 450.0        | 3        | 4         | 4        | 0.067     
Iterative Deepening       | SUCCESS  | 3       | 450.0        | 8        | 18        | 4        | 0.051     
Uniform-Cost Search       | SUCCESS  | 4       | 418.0        | 13       | 31        | 4        | 0.047     
Greedy Best-First         | SUCCESS  | 3       | 450.0        | 4        | 10        | 5        | 0.022     
A* Search                 | SUCCESS  | 4       | 418.0        | 6        | 16        | 6        | 0.025     
===============================================================================================

===============================================================================================
BENCHMARK: NAVIGASI GRID 2D 4X4 DENGAN RINTANGAN
===============================================================================================
Algoritma                 | Status   | Langkah | Biaya/Jarak  | Expanded | Generated | Frontier | Waktu (ms)
-----------------------------------------------------------------------------------------------
Breadth-First Search      | SUCCESS  | 6       | 6.0          | 10       | 22        | 2        | 0.045     
Depth-First Search        | SUCCESS  | 6       | 6.0          | 12       | 25        | 3        | 0.034     
Depth-Limited (L=6)       | SUCCESS  | 6       | 6.0          | 6        | 8         | 7        | 0.021     
Iterative Deepening       | SUCCESS  | 6       | 6.0          | 31       | 44        | 7        | 0.130     
Uniform-Cost Search       | SUCCESS  | 6       | 6.0          | 13       | 27        | 3        | 0.037     
Greedy Best-First         | SUCCESS  | 6       | 6.0          | 7        | 14        | 3        | 0.033     
A* Search (Manhattan)     | SUCCESS  | 6       | 6.0          | 12       | 25        | 3        | 0.032     
===============================================================================================
```

