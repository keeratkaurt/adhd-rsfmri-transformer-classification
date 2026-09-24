"""Data loading and validation for connectivity matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_connectivity_npz(path: str | Path, n_rois: int = 116) -> tuple[np.ndarray, np.ndarray]:
    """Load ``X`` matrices and binary ``y`` labels from a compressed NumPy file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        if not {"X", "y"}.issubset(archive.files):
            raise ValueError("NPZ file must contain arrays named 'X' and 'y'")
        features = np.asarray(archive["X"], dtype=np.float32)
        labels = np.asarray(archive["y"], dtype=np.int64)

    if features.ndim != 3 or features.shape[1:] != (n_rois, n_rois):
        raise ValueError(
            f"X must have shape (n_participants, {n_rois}, {n_rois}); "
            f"received {features.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("y must be one-dimensional and aligned with X")
    if not np.isfinite(features).all():
        raise ValueError("X contains NaN or infinite values")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y must use 0 = control and 1 = ADHD")
    if len(np.unique(labels)) != 2:
        raise ValueError("Both control and ADHD samples are required")

    return features, labels


def make_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
