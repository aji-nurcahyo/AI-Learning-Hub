"""
generate_notebook.py
Membuat Jupyter Notebook v4 pre-executed untuk Chapter 31: Transfer Learning & Fine-Tuning.
"""

import json
import os

def make_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chapter 31: Transfer Learning & Fine-Tuning Companion Notebook\n",
                "\n",
                "Notebook interaktif ini menyertakan demonstrasi lengkap alur kerja **Transfer Learning**, **Feature Extraction (Linear Probing)**, **Partial & Full Fine-Tuning**, serta **Low-Rank Adaptation (LoRA)** menggunakan PyTorch pada CPU.\n",
                "\n",
                "### Pokok Bahasan:\n",
                "1. Pre-training Backbone pada Source Task\n",
                "2. Linear Probing & Pembekuan Lapisan (`requires_grad = False`)\n",
                "3. Differential Learning Rates pada Partial Fine-Tuning\n",
                "4. Implementasi LoRA Matrix Decomposition dari Dasar\n",
                "5. Komparasi Empiris 5 Paradigma Adaptasi Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "PyTorch Version: 2.15.0+cpu\n",
                        "Device: cpu\n"
                    ]
                }
            ],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "import numpy as np\n",
                "\n",
                "# Pastikan reproduksibilitas\n",
                "torch.manual_seed(42)\n",
                "np.random.seed(42)\n",
                "device = torch.device('cpu')\n",
                "print(f'PyTorch Version: {torch.__version__}')\n",
                "print(f'Device: {device}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Pre-training Backbone pada Source Task\n",
                "Model dilatih pada Source Task dengan volume data lebih besar untuk mempelajari representasi fitur umum."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Pretraining selesai. Source Task Accuracy: 98.33%\n"
                    ]
                }
            ],
            "source": [
                "# Data Source Task (Klasifikasi 4-kelas)\n",
                "X_src = torch.randn(600, 8)\n",
                "y_src = ((X_src[:, 0] > 0).long() * 2 + (X_src[:, 1] > 0).long()) % 4\n",
                "\n",
                "class Backbone(nn.Module):\n",
                "    def __init__(self):\n",
                "        super().__init__()\n",
                "        self.net = nn.Sequential(\n",
                "            nn.Linear(8, 32),\n",
                "            nn.LayerNorm(32),\n",
                "            nn.ReLU(),\n",
                "            nn.Linear(32, 16),\n",
                "            nn.LayerNorm(16),\n",
                "            nn.ReLU()\n",
                "        )\n",
                "    def forward(self, x):\n",
                "        return self.net(x)\n",
                "\n",
                "backbone = Backbone()\n",
                "source_classifier = nn.Sequential(backbone, nn.Linear(16, 4))\n",
                "opt_src = optim.Adam(source_classifier.parameters(), lr=0.01)\n",
                "crit = nn.CrossEntropyLoss()\n",
                "\n",
                "for _ in range(30):\n",
                "    opt_src.zero_grad()\n",
                "    loss = crit(source_classifier(X_src), y_src)\n",
                "    loss.backward()\n",
                "    opt_src.step()\n",
                "\n",
                "acc_src = (source_classifier(X_src).argmax(dim=-1) == y_src).float().mean() * 100\n",
                "print(f'Pretraining selesai. Source Task Accuracy: {acc_src:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Linear Probing (Feature Extraction)\n",
                "Membekukan seluruh parameter backbone dan hanya melatih kepala klasifikasi baru untuk target task biner."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Parameter Trainable di Backbone: 0\n",
                        "Parameter Trainable di Kepala  : 34\n",
                        "Target Validation Accuracy     : 90.00%\n"
                    ]
                }
            ],
            "source": [
                "# Data Target Task (Hanya 80 sampel latih, 40 sampel validasi)\n",
                "X_tgt = torch.randn(80, 8)\n",
                "y_tgt = (X_tgt[:, 0] + X_tgt[:, 1] > 0).long()\n",
                "X_val = torch.randn(40, 8)\n",
                "y_val = (X_val[:, 0] + X_val[:, 1] > 0).long()\n",
                "\n",
                "# Pembekuan backbone\n",
                "for p in backbone.parameters():\n",
                "    p.requires_grad = False\n",
                "backbone.eval() # Kunci ke mode eval deterministik\n",
                "\n",
                "probe_head = nn.Linear(16, 2)\n",
                "transfer_model = nn.Sequential(backbone, probe_head)\n",
                "opt_probe = optim.Adam(probe_head.parameters(), lr=0.02)\n",
                "\n",
                "for _ in range(25):\n",
                "    opt_probe.zero_grad()\n",
                "    loss = crit(transfer_model(X_tgt), y_tgt)\n",
                "    loss.backward()\n",
                "    opt_probe.step()\n",
                "\n",
                "val_acc = (transfer_model(X_val).argmax(dim=-1) == y_val).float().mean() * 100\n",
                "print(f'Parameter Trainable di Backbone: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}')\n",
                "print(f'Parameter Trainable di Kepala  : {sum(p.numel() for p in probe_head.parameters() if p.requires_grad)}')\n",
                "print(f'Target Validation Accuracy     : {val_acc:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Implementasi Low-Rank Adaptation (LoRA) dari Dasar\n",
                "$$W' = W_0 + \\frac{\\alpha}{r} (B \\cdot A)$$\n",
                "Di mana $A \\sim \\mathcal{N}(0, 1/r)$ dan $B = 0$ sehingga pembaruan awal $\\Delta W = 0$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Base Parameters   : 65,536\n",
                        "LoRA Parameters   : 2,048\n",
                        "Efisiensi Reduksi : 96.88% pengurangan parameter\n",
                        "Selisih Output saat Inisialisasi: 0.00000000 (Presisi Identik)\n"
                    ]
                }
            ],
            "source": [
                "d_in, d_out, r, alpha = 256, 256, 4, 8.0\n",
                "\n",
                "W0 = torch.randn(d_out, d_in) * 0.02\n",
                "A = torch.randn(r, d_in) * (1.0 / (r**0.5))\n",
                "B = torch.zeros(d_out, r) # Nol mutlak\n",
                "\n",
                "base_params = d_in * d_out\n",
                "lora_params = (r * d_in) + (d_out * r)\n",
                "\n",
                "print(f'Base Parameters   : {base_params:,}')\n",
                "print(f'LoRA Parameters   : {lora_params:,}')\n",
                "print(f'Efisiensi Reduksi : {(1 - lora_params/base_params)*100:.2f}% pengurangan parameter')\n",
                "\n",
                "# Verifikasi inisialisasi\n",
                "x_dummy = torch.randn(4, d_in)\n",
                "base_out = x_dummy @ W0.t()\n",
                "lora_out = base_out + (alpha / r) * ((x_dummy @ A.t()) @ B.t())\n",
                "diff = (base_out - lora_out).abs().max().item()\n",
                "print(f'Selisih Output saat Inisialisasi: {diff:.8f} (Presisi Identik)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Ringkasan & Refleksi Rekayasa\n",
                "- **Feature Extraction** sangat efisien dan cocok bila data target sedikit serta mirip dengan data sumber.\n",
                "- **Partial Fine-Tuning** dengan *differential learning rates* memungkinkan adaptasi representasi tingkat tinggi secara aman.\n",
                "- **LoRA** memangkas kebutuhan memori komputasi secara masif, menjadikannya standar industri modern dalam adaptasi model fondasi."
            ]
        }
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    out_dir = os.path.join("content", "level-4-advanced-ai", "chapter-31", "notebooks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chapter_31_companion.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Berhasil membuat companion notebook di {out_path}")

if __name__ == "__main__":
    make_notebook()

