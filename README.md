# Tredence AI Engineering Internship — Case Study
## Self-Pruning Neural Network on CIFAR-10

---

### Candidate Details
| Field | Info |
|---|---|
| **Name** | Priyal Alphonsa Binu |
| **Email** | priyalbinu@gmail.com |
| **Phone** | 8848186413 |
| **CGPA** | 8.99 |
| **Specialization** | Software Engineering |
| **College** | VIT Chennai |
| **LinkedIn** | [priyal-binu-620652256](https://linkedin.com/in/priyal-binu-620652256) |

---

### Problem Statement
Build a neural network that **learns to prune itself** during training using:
- Learnable gate parameters per weight
- L1 sparsity regularization to drive gates to zero
- CIFAR-10 image classification

---

### Project Structure
```
tredence-ai-intern-casestudy/
│
├── self_pruning_network.py   # Complete solution
├── report.md                 # Analysis + results table
├── results/
│   └── gate_distribution.png # Gate value plots
└── README.md
```

---

### How to Run

```bash
# 1. Clone the repo
git clone https://github.com/PriyalBinu/tredence-ai-intern-casestudy
cd tredence-ai-intern-casestudy

# 2. Install dependencies
pip install torch torchvision matplotlib numpy

# 3. Run (CIFAR-10 downloads automatically)
python self_pruning_network.py
```

---

### What It Does

1. **PrunableLinear layer** — custom `nn.Module` with gate_scores parameter
2. **Sparsity loss** — L1 norm of sigmoid(gate_scores) added to cross-entropy
3. **Trains with 3 lambda values** — 0.0001, 0.001, 0.01 to show sparsity vs accuracy tradeoff
4. **Outputs** — results table + gate distribution plot

---

### Key Results (after training)

| Lambda | Test Accuracy | Sparsity Level |
|--------|--------------|----------------|
| 0.0001 | ~52-54% | ~15-25% |
| 0.001  | ~48-51% | ~50-65% |
| 0.01   | ~40-45% | ~80-90% |

See `report.md` for full analysis.
