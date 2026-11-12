"""
Chapter 22 - Backpropagation & Optimization
Script 2: Graf Komputasi & Diferensiasi Otomatis Terbalik (Computational Graph)
Mengimplementasikan graf komputasi terarah (Directed Acyclic Graph) dengan eksekusi
propagasi maju (Forward Computation) dan propagasi mundur (Reverse-Mode Automatic Differentiation).
"""

import numpy as np

class Var:
    """Node skalar di dalam graf komputasi yang mencatat nilai, gradien, dan operasi penciptanya."""
    def __init__(self, value, children=(), op=""):
        self.value = float(value)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.value + other.value, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.value * other.value, (self, other), "*")
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Var(max(0.0, self.value), (self,), "ReLU")
        def _backward():
            self.grad += (1.0 if self.value > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Mengeksekusi reverse-mode differentiation melalui topological sort graf."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Gradien root terhadap dirinya sendiri adalah 1.0 (dL/dL = 1)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

def run_graph_demonstration():
    print("=" * 80)
    print("CHAPTER 22: GRAF KOMPUTASI & DIFERENSIASI REVERSE-MODE (AUTOGRAD DASAR)")
    print("=" * 80)
    
    # Fungsi: L = ReLU(w1 * x1 + w2 * x2 + b)
    # Parameter konkret:
    x1 = Var(1.5, op="x1")
    w1 = Var(2.0, op="w1")
    x2 = Var(-1.0, op="x2")
    w2 = Var(3.0, op="w2")
    b  = Var(0.5, op="b")
    
    print("Mengevaluasi Komputasi Maju:")
    print("  p1 = w1 * x1 = 2.0 * 1.5 = 3.0")
    print("  p2 = w2 * x2 = 3.0 * (-1.0) = -3.0")
    print("  s  = p1 + p2 = 0.0")
    print("  z  = s + b   = 0.0 + 0.5 = 0.5")
    print("  L  = ReLU(z) = ReLU(0.5) = 0.5")
    
    p1 = w1 * x1
    p2 = w2 * x2
    s = p1 + p2
    z = s + b
    L = z.relu()
    
    print(f"\nHasil Forward Pass L.value = {L.value}")
    
    # Jalankan Reverse-Mode Backpropagation
    L.backward()
    
    print("\nHasil Backward Pass (Gradien dL/d...):")
    print(f"  dL/dL  : {L.grad:.4f} (Benih awal / seed)")
    print(f"  dL/dz  : {z.grad:.4f} (Gradien lokal ReLU pada z=0.5 > 0)")
    print(f"  dL/db  : {b.grad:.4f} (Gradien terhadap bias)")
    print(f"  dL/dw1 : {w1.grad:.4f} (Ekspektasi: dL/dz * dz/dw1 = 1.0 * x1 = 1.5)")
    print(f"  dL/dw2 : {w2.grad:.4f} (Ekspektasi: dL/dz * dz/dw2 = 1.0 * x2 = -1.0)")
    print(f"  dL/dx1 : {x1.grad:.4f} (Ekspektasi: dL/dz * dz/dx1 = 1.0 * w1 = 2.0)")
    print(f"  dL/dx2 : {x2.grad:.4f} (Ekspektasi: dL/dz * dz/dx2 = 1.0 * w2 = 3.0)")
    
    # Verifikasi assert
    assert np.isclose(w1.grad, 1.5), "Gradien w1 salah!"
    assert np.isclose(w2.grad, -1.0), "Gradien w2 salah!"
    assert np.isclose(b.grad, 1.0), "Gradien b salah!"
    
    print("=" * 80)
    print("STATUS VERIFIKASI: Seluruh gradien graf komputasi cocok persis dengan turunan analitis!")

if __name__ == "__main__":
    run_graph_demonstration()

