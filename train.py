"""Train and evaluate the rs-fMRI connectivity transformer with cross-validation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adhd_transformer.data import load_connectivity_npz, make_dataset
from adhd_transformer.metrics import binary_metrics
from adhd_transformer.model import FunctionalConnectivityTransformer, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="NPZ file containing X and y")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--balanced-loss",
        action="store_true",
        help="Weight ADHD examples to counter class imbalance",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.numpy())
    return binary_metrics(
        np.concatenate(labels),
        np.concatenate(probabilities),
        threshold=threshold,
    )


def train_fold(
    fold: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    dataset: torch.utils.data.Dataset,
    labels: np.ndarray,
    config: TransformerConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(args.seed + fold)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = FunctionalConnectivityTransformer(config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    positive_weight = None
    if args.balanced_loss:
        fold_labels = labels[train_indices]
        negatives = np.count_nonzero(fold_labels == 0)
        positives = np.count_nonzero(fold_labels == 1)
        positive_weight = torch.tensor([negatives / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            mean_loss = running_loss / len(train_indices)
            print(f"fold={fold} epoch={epoch:03d} train_loss={mean_loss:.5f}")

    metrics = evaluate(model, validation_loader, device, args.threshold)
    metrics["fold"] = fold
    checkpoint = args.output_dir / f"fold_{fold}.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": asdict(config)},
        checkpoint,
    )
    return metrics


def summarize(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metric_names = ["accuracy", "sensitivity", "specificity", "f1", "auroc", "auprc"]
    return {
        name: {
            "mean": float(np.mean([fold[name] for fold in fold_metrics])),
            "std": float(np.std([fold[name] for fold in fold_metrics], ddof=1)),
        }
        for name in metric_names
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.folds < 2:
        raise ValueError("epochs must be positive and folds must be at least 2")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, labels = load_connectivity_npz(args.data)
    dataset = make_dataset(features, labels)

    smallest_class = int(np.bincount(labels).min())
    if args.folds > smallest_class:
        raise ValueError(f"folds cannot exceed the smallest class count ({smallest_class})")

    config = TransformerConfig(
        n_rois=features.shape[1],
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dim_feedforward=args.feedforward_dim,
        dropout=args.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"participants={len(labels)} controls={(labels == 0).sum()} ADHD={(labels == 1).sum()}")
    print(f"device={device}")

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels), start=1
    ):
        fold_metrics.append(
            train_fold(
                fold,
                train_indices,
                validation_indices,
                dataset,
                labels,
                config,
                args,
                device,
            )
        )

    results = {
        "label_convention": {"0": "control", "1": "ADHD"},
        "config": asdict(config),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "folds": args.folds,
            "seed": args.seed,
            "threshold": args.threshold,
            "balanced_loss": args.balanced_loss,
        },
        "folds": fold_metrics,
        "aggregate": summarize(fold_metrics),
    }
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
