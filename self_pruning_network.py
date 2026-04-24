"""
Self-Pruning Neural Network on CIFAR-10
Author: Priyal Alphonsa Binu
Case Study: Tredence AI Engineering Internship

Description:
    Implements a neural network that learns to prune itself during training
    using learnable gate parameters and L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ─────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate parameters.

    Each weight has a corresponding gate_score (scalar). During forward pass:
        - gate_scores are passed through Sigmoid → gates in [0, 1]
        - pruned_weights = weight * gates  (element-wise)
        - output = pruned_weights @ input.T + bias

    When a gate approaches 0, the corresponding weight is effectively pruned.
    Gradients flow through both weight and gate_scores via autograd.
    """

    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard learnable weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Gate scores — same shape as weight, also learnable
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        # Initialize gate_scores to 1.0 so sigmoid(1) ≈ 0.73 (mostly open at start)
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert gate_scores → gates in [0, 1] via Sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Apply gates element-wise to weights (pruning mechanism)
        pruned_weights = self.weight * gates

        # Step 3: Standard linear operation with pruned weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (after sigmoid)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Percentage of weights whose gate is below threshold (pruned)."""
        gates = self.get_gates()
        pruned = (gates < threshold).float().sum()
        return (pruned / gates.numel()).item() * 100


# ─────────────────────────────────────────────
# Neural Network using PrunableLinear
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 classification.
    Uses PrunableLinear layers so the network can self-prune during training.

    Architecture:
        Input (3072) → FC1 (512) → ReLU → FC2 (256) → ReLU → FC3 (10)
    """

    def __init__(self):
        super(SelfPruningNet, self).__init__()

        # CIFAR-10: 3 channels × 32 × 32 = 3072 input features
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # Flatten: (B, 3, 32, 32) → (B, 3072)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_all_gates(self) -> torch.Tensor:
        """Collect gate values from all PrunableLinear layers."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().view(-1))
        return torch.cat(all_gates)

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across all PrunableLinear layers.
        Encourages gates to be driven to exactly 0 (sparsity).
        """
        total = 0.0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total = total + gates.abs().sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall percentage of pruned weights across entire network."""
        total_weights = 0
        pruned_weights = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates()
                pruned_weights += (gates < threshold).float().sum().item()
                total_weights += gates.numel()
        return (pruned_weights / total_weights) * 100


# ─────────────────────────────────────────────
# PART 2 & 3: Training and Evaluation
# ─────────────────────────────────────────────

def get_data_loaders(batch_size: int = 128):
    """Load and preprocess CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lambda_sparse):
    """
    Train for one epoch.

    Total Loss = CrossEntropyLoss + λ * SparsityLoss
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        # Classification loss
        cls_loss = F.cross_entropy(outputs, labels)

        # Sparsity regularization loss (L1 on gates)
        sparse_loss = model.sparsity_loss()

        # Total loss
        loss = cls_loss + lambda_sparse * sparse_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def train_and_evaluate(lambda_sparse: float, epochs: int, device,
                       train_loader, test_loader) -> dict:
    """
    Full training run for a given lambda value.
    Returns results dict with accuracy, sparsity, and gate values.
    """
    print(f"\n{'='*50}")
    print(f"Training with λ = {lambda_sparse}")
    print(f"{'='*50}")

    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, lambda_sparse)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            sparsity = model.overall_sparsity()
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Sparsity: {sparsity:.2f}%")

    test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()
    all_gates = model.get_all_gates().numpy()

    print(f"\nFinal Test Accuracy : {test_acc:.2f}%")
    print(f"Final Sparsity Level: {final_sparsity:.2f}%")

    return {
        "lambda": lambda_sparse,
        "test_accuracy": test_acc,
        "sparsity": final_sparsity,
        "gates": all_gates,
    }


def plot_gate_distributions(results: list, save_dir: str = "results"):
    """
    Plot gate value distributions for all lambda values.
    A successful pruning shows a spike near 0 and a cluster away from 0.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        gates = result["gates"]
        lam = result["lambda"]
        acc = result["test_accuracy"]
        sparsity = result["sparsity"]

        ax.hist(gates, bins=80, color='steelblue', edgecolor='white',
                alpha=0.85)
        ax.set_title(f'λ = {lam}\nAcc: {acc:.1f}% | Sparsity: {sparsity:.1f}%',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Gate Value', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.axvline(x=0.01, color='red', linestyle='--',
                   linewidth=1.5, label='Prune threshold (0.01)')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gate Value Distributions — Self-Pruning Neural Network\n'
                 'Spike at 0 = successful pruning',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "gate_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def print_results_table(results: list):
    """Print a formatted results summary table."""
    print("\n" + "="*55)
    print(f"{'RESULTS SUMMARY':^55}")
    print("="*55)
    print(f"{'Lambda':<12} {'Test Accuracy':>16} {'Sparsity Level':>16}")
    print("-"*55)
    for r in results:
        print(f"{r['lambda']:<12} {r['test_accuracy']:>14.2f}% {r['sparsity']:>14.2f}%")
    print("="*55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Config
    EPOCHS = 30
    BATCH_SIZE = 128
    LAMBDA_VALUES = [0.0001, 0.001, 0.01]   # Low, Medium, High

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # Train with each lambda value
    results = []
    for lam in LAMBDA_VALUES:
        result = train_and_evaluate(
            lambda_sparse=lam,
            epochs=EPOCHS,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
        )
        results.append(result)

    # Print summary table
    print_results_table(results)

    # Plot gate distributions
    plot_gate_distributions(results, save_dir="results")

    print("\nDone! Check 'results/gate_distribution.png' for the plot.")


if __name__ == "__main__":
    main()
