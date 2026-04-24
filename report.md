# Self-Pruning Neural Network — Case Study Report

**Author:** Priyal Alphonsa Binu  
**Email:** priyalbinu@gmail.com  
**CGPA:** 8.99 | **Specialization:** Software Engineering, VIT Chennai

---

## 1. Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

The total loss is defined as:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

Where `SparsityLoss = Σ |gate_i|` (L1 norm of all gate values).

Since gates are outputs of the Sigmoid function, they are always in the range **(0, 1)** — always non-negative. So the L1 norm simplifies to the **sum of all gate values**.

### Why L1 and not L2?

| Regularization | Effect on small values |
|---|---|
| **L2** (sum of squares) | Shrinks values toward 0, but rarely reaches exactly 0 |
| **L1** (sum of absolutes) | Creates a **constant gradient** regardless of value size — pushes small values all the way to exactly 0 |

The L1 penalty applies a **constant downward pressure** on every gate. The optimizer is always "incentivized" to reduce each gate to zero unless the classification loss specifically needs that weight to remain active.

### The Sparsity Mechanism in Practice

1. At the start, all `gate_scores` are initialized to 1.0, so `sigmoid(1.0) ≈ 0.73` — most gates are open.
2. During training, the sparsity loss penalizes high gate values.
3. Unimportant weights: the classification gradient is weak → sparsity gradient dominates → gate is driven to 0.
4. Important weights: the classification gradient is strong → the network keeps the gate open despite the penalty.
5. The result: a **bimodal distribution** — gates cluster near 0 (pruned) or near 1 (active).

This is the key insight — L1 encourages **exact zeros**, while L2 only encourages **small values**.

---

## 2. Results Table

| Lambda (λ) | Test Accuracy | Sparsity Level (%) | Notes |
|---|---|---|---|
| 0.0001 | ~52–54% | ~15–25% | Light pruning, higher accuracy |
| 0.001 | ~48–51% | ~50–65% | Balanced trade-off |
| 0.01 | ~40–45% | ~80–90% | Aggressive pruning, lower accuracy |

> **Note:** Exact values depend on hardware and random seed. Run `self_pruning_network.py` to reproduce results. Results will be added here after training.

### Key Observations

- **Low λ (0.0001):** The classification loss dominates. Very few gates are driven to zero. The model retains most weights → higher accuracy but minimal pruning.
- **Medium λ (0.001):** A balanced trade-off. Significant sparsity is achieved while maintaining reasonable accuracy. This is typically the **best practical setting**.
- **High λ (0.01):** The sparsity loss dominates. Most gates are driven to zero aggressively → very sparse network but accuracy drops as important weights also get pruned.

---

## 3. Gate Value Distribution

The plot `results/gate_distribution.png` shows the distribution of all gate values after training.

### What a Successful Result Looks Like

```
Count
  |
  |█                           █
  |█                        █████
  |██                      ███████
  |████                  ██████████
  |██████              ████████████
  +─────────────────────────────────→ Gate Value
  0   0.1   0.2   0.3   0.4  ...  1.0

  ↑ Large spike at 0           ↑ Cluster of active gates
  (pruned weights)              (important weights)
```

A successful self-pruning run shows:
- A **large spike near gate = 0** → many weights pruned
- A **second cluster away from 0** (near 0.5–1.0) → important weights retained

---

## 4. Architecture

```
Input (CIFAR-10: 3×32×32 = 3072)
        ↓
PrunableLinear(3072 → 512) + ReLU + Dropout(0.3)
        ↓
PrunableLinear(512 → 256) + ReLU + Dropout(0.3)
        ↓
PrunableLinear(256 → 10)
        ↓
Output (10 classes)
```

Each `PrunableLinear` layer has:
- `weight`: shape `(out, in)` — standard learnable weights
- `gate_scores`: shape `(out, in)` — learnable gate parameters
- Forward: `output = Linear(weight * sigmoid(gate_scores), x)`

---

## 5. How to Run

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run training (downloads CIFAR-10 automatically)
python self_pruning_network.py
```

**Output:**
- Training logs per epoch
- Final results table (Lambda / Accuracy / Sparsity)
- `results/gate_distribution.png` — gate value distribution plot

---

## 6. Design Decisions

- **Sigmoid for gates:** Keeps gates in (0,1), making them interpretable as "how active is this weight"
- **Gate init at 1.0:** sigmoid(1.0) ≈ 0.73 — starts mostly open, gives the network a fair start before pruning begins
- **Adam optimizer:** Updates both weights and gate_scores jointly; adaptive learning rates help gate_scores converge to 0 or 1
- **Cosine LR scheduler:** Smooth learning rate decay improves final accuracy
- **Dropout + sparsity:** Dropout prevents co-adaptation; sparsity loss removes redundant weights — complementary regularization
- **Threshold = 0.01:** Gates below 1% are considered "pruned" — conservative threshold to avoid false positives
