"""MNIST addition with a neuro-symbolic bridge.

This script turns the notebook experiment into a small, reproducible project:
- a perception network reads each digit image,
- a symbolic sum layer combines digit probabilities,
- a dense baseline predicts the sum directly from pixels,
- evaluation covers standard accuracy, OOD digit ranges, and noise robustness.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


SUM_CLASSES = 19
DIGIT_CLASSES = 10
DEFAULT_SEED = 7
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("results")


@dataclass
class ExperimentConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    batch_size: int = 64
    seed: int = DEFAULT_SEED
    num_workers: int = 0
    perception_epochs: int = 1
    standard_train_pairs: int = 10_000
    standard_test_pairs: int = 2_000
    standard_epochs: int = 5
    ood_train_pairs: int = 10_000
    ood_test_pairs: int = 2_000
    ood_epochs: int = 2
    noise_levels: Tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
    ood_train_digit_max: int = 7
    ood_test_digit_min: int = 8
    quick_mode: bool = False


class MNISTAdditionDataset(Dataset):
    """Pair two MNIST samples and return their digit sum."""

    def __init__(self, base_dataset: Dataset, num_pairs: int, seed: int = DEFAULT_SEED) -> None:
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        self.dataset_size = len(base_dataset)
        rng = np.random.default_rng(seed)
        self.pairs = [
            (int(rng.integers(0, self.dataset_size)), int(rng.integers(0, self.dataset_size)))
            for _ in range(num_pairs)
        ]

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, index: int):
        idx1, idx2 = self.pairs[index]
        img1, label1 = self.base_dataset[idx1]
        img2, label2 = self.base_dataset[idx2]
        target_sum = int(label1) + int(label2)
        return img1, img2, target_sum


class PerceptionNet(nn.Module):
    """Tiny CNN that predicts a single digit from a normalized MNIST image."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, DIGIT_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class NeuroSymbolicAddition(nn.Module):
    """Combine digit probabilities with a fixed symbolic addition rule."""

    def __init__(self, perception_model: PerceptionNet) -> None:
        super().__init__()
        self.perception = perception_model

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        batch_size = img1.size(0)
        digit_logits1 = self.perception(img1)
        digit_logits2 = self.perception(img2)
        digit_prob1 = torch.softmax(digit_logits1, dim=1)
        digit_prob2 = torch.softmax(digit_logits2, dim=1)
        joint_prob = torch.bmm(digit_prob1.unsqueeze(2), digit_prob2.unsqueeze(1))
        sum_prob = torch.zeros(batch_size, SUM_CLASSES, device=img1.device, dtype=joint_prob.dtype)

        for digit1 in range(DIGIT_CLASSES):
            for digit2 in range(DIGIT_CLASSES):
                sum_prob[:, digit1 + digit2] += joint_prob[:, digit1, digit2]

        return torch.log(sum_prob.clamp_min(1e-8))


class PureNeuralBaseline(nn.Module):
    """Dense baseline that predicts the sum directly from two flattened images."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, SUM_CLASSES),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        x1 = img1.view(img1.size(0), -1)
        x2 = img2.view(img2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def load_mnist(data_dir: Path):
    train_dataset = torchvision.datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=get_transform()
    )
    test_dataset = torchvision.datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=get_transform()
    )
    return train_dataset, test_dataset


def make_addition_loaders(
    base_train: Dataset,
    base_test: Dataset,
    train_pairs: int,
    test_pairs: int,
    batch_size: int,
    seed: int,
    num_workers: int,
):
    train_dataset = MNISTAdditionDataset(base_train, num_pairs=train_pairs, seed=seed)
    test_dataset = MNISTAdditionDataset(base_test, num_pairs=test_pairs, seed=seed + 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataset, test_dataset, train_loader, test_loader


def filter_by_digit_range(dataset: Dataset, minimum_digit: int, maximum_digit: int) -> Subset:
    selected_indices = [index for index, (_, label) in enumerate(dataset) if minimum_digit <= int(label) <= maximum_digit]
    return Subset(dataset, selected_indices)


def train_single_digit_model(
    model: PerceptionNet,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float = 1e-3,
) -> PerceptionNet:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        total_items = 0
        correct_items = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_items += (predictions == labels).sum().item()
            total_items += labels.size(0)

        mean_loss = running_loss / max(len(loader), 1)
        accuracy = 100.0 * correct_items / max(total_items, 1)
        print(f"[Perception] Epoch {epoch + 1}/{epochs} - Loss: {mean_loss:.4f} - Accuracy: {accuracy:.2f}%")

    return model


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def train_sum_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    label: str,
):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_items = 0
        correct_items = 0

        for img1, img2, target_sums in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            target_sums = target_sums.to(device)

            optimizer.zero_grad()
            output_log_probs = model(img1, img2)
            loss = criterion(output_log_probs, target_sums)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(output_log_probs, dim=1)
            correct_items += (predictions == target_sums).sum().item()
            total_items += target_sums.size(0)

        mean_loss = running_loss / max(len(loader), 1)
        accuracy = 100.0 * correct_items / max(total_items, 1)
        print(f"[{label}] Epoch {epoch + 1}/{epochs} - Loss: {mean_loss:.4f} - Accuracy: {accuracy:.2f}%")

    return model


@torch.no_grad()
def evaluate_sum_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct_items = 0
    total_items = 0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for img1, img2, target_sums in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        target_sums = target_sums.to(device)
        output_log_probs = model(img1, img2)
        predictions = torch.argmax(output_log_probs, dim=1)
        correct_items += (predictions == target_sums).sum().item()
        total_items += target_sums.size(0)
        all_targets.extend(target_sums.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    accuracy = 100.0 * correct_items / max(total_items, 1)
    return accuracy, all_targets, all_predictions


@torch.no_grad()
def evaluate_noise_robustness(
    nesy_model: nn.Module,
    baseline_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_levels: Sequence[float],
):
    nesy_results: List[float] = []
    baseline_results: List[float] = []

    nesy_model.eval()
    baseline_model.eval()

    for noise_level in noise_levels:
        nesy_correct = 0
        baseline_correct = 0
        total_items = 0

        for img1, img2, target_sums in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            target_sums = target_sums.to(device)
            total_items += target_sums.size(0)

            corrupted_img1 = apply_gaussian_noise(img1, noise_level)
            corrupted_img2 = apply_gaussian_noise(img2, noise_level)

            nesy_predictions = torch.argmax(nesy_model(corrupted_img1, corrupted_img2), dim=1)
            baseline_predictions = torch.argmax(baseline_model(corrupted_img1, corrupted_img2), dim=1)

            nesy_correct += (nesy_predictions == target_sums).sum().item()
            baseline_correct += (baseline_predictions == target_sums).sum().item()

        nesy_results.append(100.0 * nesy_correct / max(total_items, 1))
        baseline_results.append(100.0 * baseline_correct / max(total_items, 1))

    return nesy_results, baseline_results


def apply_gaussian_noise(images: torch.Tensor, noise_level: float) -> torch.Tensor:
    noisy_images = images + noise_level * torch.randn_like(images)
    return torch.clamp(noisy_images, -1.0, 1.0)


def save_line_chart(x_values, y_series, labels, title, xlabel, ylabel, output_path: Path):
    plt.figure(figsize=(8, 5))
    for series, label, marker, color, linestyle in y_series:
        plt.plot(x_values, series, marker=marker, linewidth=2.2, color=color, linestyle=linestyle, label=label)

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_bar_chart(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, output_path: Path):
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(labels, values, color=["#1f4e79", "#b22222"], width=0.55)
    plt.ylim(0, 100)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def maybe_shorten_for_quick_mode(config: ExperimentConfig) -> ExperimentConfig:
    if not config.quick_mode:
        return config

    return ExperimentConfig(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        perception_epochs=1,
        standard_train_pairs=1_500,
        standard_test_pairs=400,
        standard_epochs=2,
        ood_train_pairs=1_500,
        ood_test_pairs=400,
        ood_epochs=1,
        noise_levels=config.noise_levels,
        ood_train_digit_max=config.ood_train_digit_max,
        ood_test_digit_min=config.ood_test_digit_min,
        quick_mode=True,
    )


def run_experiment(config: ExperimentConfig):
    config = maybe_shorten_for_quick_mode(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_dataset, test_dataset = load_mnist(config.data_dir)
    train_loader_single = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )

    print("\n== Pre-training perception layer on single digits ==")
    perception = PerceptionNet().to(device)
    train_single_digit_model(perception, train_loader_single, device, config.perception_epochs)

    print("\n== Standard MNIST addition benchmark ==")
    standard_train_dataset, standard_test_dataset, standard_train_loader, standard_test_loader = make_addition_loaders(
        train_dataset,
        test_dataset,
        config.standard_train_pairs,
        config.standard_test_pairs,
        config.batch_size,
        config.seed,
        config.num_workers,
    )

    standard_nesy = NeuroSymbolicAddition(perception).to(device)
    standard_baseline = PureNeuralBaseline().to(device)

    train_sum_model(
        standard_nesy,
        standard_train_loader,
        device,
        config.standard_epochs,
        learning_rate=5e-4,
        label="NeSy-Standard",
    )
    train_sum_model(
        standard_baseline,
        standard_train_loader,
        device,
        config.standard_epochs,
        learning_rate=1e-3,
        label="Baseline-Standard",
    )

    standard_nesy_acc, standard_targets, standard_predictions = evaluate_sum_model(
        standard_nesy, standard_test_loader, device
    )
    standard_baseline_acc, baseline_targets, baseline_predictions = evaluate_sum_model(
        standard_baseline, standard_test_loader, device
    )

    print(f"[Standard Test] NeSy accuracy: {standard_nesy_acc:.2f}%")
    print(f"[Standard Test] Baseline accuracy: {standard_baseline_acc:.2f}%")

    print("\n== OOD digit-range benchmark ==")
    ood_train_subset = filter_by_digit_range(train_dataset, 0, config.ood_train_digit_max)
    ood_test_subset = filter_by_digit_range(test_dataset, config.ood_test_digit_min, 9)
    ood_train_loader = DataLoader(
        MNISTAdditionDataset(ood_train_subset, config.ood_train_pairs, seed=config.seed),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    ood_test_loader = DataLoader(
        MNISTAdditionDataset(ood_test_subset, config.ood_test_pairs, seed=config.seed + 1),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    ood_perception = PerceptionNet().to(device)
    train_single_digit_model(ood_perception, train_loader_single, device, config.perception_epochs)
    freeze_module(ood_perception)
    ood_perception.eval()

    ood_nesy = NeuroSymbolicAddition(ood_perception).to(device)
    ood_baseline = PureNeuralBaseline().to(device)

    train_sum_model(
        ood_baseline,
        ood_train_loader,
        device,
        config.ood_epochs,
        learning_rate=1e-3,
        label="Baseline-OOD",
    )

    ood_nesy_acc, ood_targets, ood_nesy_predictions = evaluate_sum_model(ood_nesy, ood_test_loader, device)
    ood_baseline_acc, _, ood_baseline_predictions = evaluate_sum_model(ood_baseline, ood_test_loader, device)

    print(f"[OOD Test] NeSy accuracy: {ood_nesy_acc:.2f}%")
    print(f"[OOD Test] Baseline accuracy: {ood_baseline_acc:.2f}%")

    print("\n== Noise robustness on the frozen OOD NeSy model ==")
    nesy_noise, baseline_noise = evaluate_noise_robustness(
        ood_nesy, ood_baseline, ood_test_loader, device, config.noise_levels
    )
    for noise_level, nesy_value, baseline_value in zip(config.noise_levels, nesy_noise, baseline_noise):
        print(
            f"Noise Level: {noise_level:.1f} | NeSy Accuracy: {nesy_value:.2f}% | "
            f"Baseline Accuracy: {baseline_value:.2f}%"
        )

    save_bar_chart(
        ["NeSy", "Baseline"],
        [ood_nesy_acc, ood_baseline_acc],
        "OOD Digit-Range Accuracy (Train: 0-7, Test: 8-9)",
        "Accuracy (%)",
        config.output_dir / "ood_accuracy_chart.png",
    )

    save_line_chart(
        list(config.noise_levels),
        [
            (nesy_noise, "Neuro-Symbolic", "o", "#1f4e79", "-"),
            (baseline_noise, "Pure Neural Baseline", "s", "#b22222", "--"),
        ],
        ["Neuro-Symbolic", "Pure Neural Baseline"],
        "Accuracy Under Increasing Gaussian Noise",
        "Noise Level",
        "Accuracy (%)",
        config.output_dir / "noise_robustness_chart.png",
    )

    summary = {
        "standard_test_accuracy": {
            "nesy": standard_nesy_acc,
            "baseline": standard_baseline_acc,
        },
        "ood_test_accuracy": {
            "nesy": ood_nesy_acc,
            "baseline": ood_baseline_acc,
        },
        "noise_robustness": {
            "noise_levels": list(config.noise_levels),
            "nesy": nesy_noise,
            "baseline": baseline_noise,
        },
        "artifacts": {
            "ood_accuracy_chart": str(config.output_dir / "ood_accuracy_chart.png"),
            "noise_robustness_chart": str(config.output_dir / "noise_robustness_chart.png"),
        },
    }

    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved results to {summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MNIST neuro-symbolic addition experiments.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory for MNIST data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plots/results.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for all loaders.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--perception-epochs", type=int, default=1, help="Single-digit pretraining epochs.")
    parser.add_argument("--standard-train-pairs", type=int, default=10_000, help="Training pairs for the standard benchmark.")
    parser.add_argument("--standard-test-pairs", type=int, default=2_000, help="Test pairs for the standard benchmark.")
    parser.add_argument("--standard-epochs", type=int, default=5, help="Training epochs for the standard benchmark.")
    parser.add_argument("--ood-train-pairs", type=int, default=10_000, help="Training pairs for the OOD benchmark.")
    parser.add_argument("--ood-test-pairs", type=int, default=2_000, help="Test pairs for the OOD benchmark.")
    parser.add_argument("--ood-epochs", type=int, default=2, help="Training epochs for the OOD benchmark.")
    parser.add_argument("--quick", action="store_true", help="Use a small configuration for smoke testing.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        perception_epochs=args.perception_epochs,
        standard_train_pairs=args.standard_train_pairs,
        standard_test_pairs=args.standard_test_pairs,
        standard_epochs=args.standard_epochs,
        ood_train_pairs=args.ood_train_pairs,
        ood_test_pairs=args.ood_test_pairs,
        ood_epochs=args.ood_epochs,
        quick_mode=args.quick,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
