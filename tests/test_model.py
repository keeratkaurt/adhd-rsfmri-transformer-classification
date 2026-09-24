import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhd_transformer.model import FunctionalConnectivityTransformer, TransformerConfig


def test_model_returns_one_logit_per_participant():
    config = TransformerConfig(
        n_rois=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        classifier_hidden=8,
        dropout=0.0,
    )
    model = FunctionalConnectivityTransformer(config)
    output = model(torch.randn(3, 8, 8))
    assert output.shape == (3,)
    assert torch.isfinite(output).all()


def test_model_rejects_wrong_matrix_shape():
    model = FunctionalConnectivityTransformer(TransformerConfig(n_rois=8))
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(2, 8, 7))


def test_config_requires_heads_to_divide_model_dimension():
    with pytest.raises(ValueError, match="divisible"):
        FunctionalConnectivityTransformer(
            TransformerConfig(n_rois=8, d_model=10, n_heads=4)
        )
