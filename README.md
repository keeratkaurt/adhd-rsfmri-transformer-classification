# Transformer-Based ADHD Classification from Resting-State fMRI

PyTorch reference implementation of a transformer encoder for classifying ADHD from AAL-116 resting-state functional-connectivity matrices.

## Project overview

This Neural Data Science team project investigated whether self-attention can capture distributed connectivity patterns that distinguish ADHD from healthy controls. Resting-state fMRI scans were preprocessed with DPARSF, regional time series were extracted with the AAL-116 atlas, and Pearson correlations produced one 116 × 116 functional-connectivity matrix per participant.

The dataset described in the report contained 162 usable participants after excluding scans with missing imaging data. The experimental configuration used five-fold cross-validation, Adam optimization, a learning rate of 0.001, a batch size of 32, and 400 epochs.

## Model

Each row of a connectivity matrix is treated as a brain-region token:

1. A linear layer projects each 116-feature token into the model dimension.
2. Sinusoidal positional encodings retain ROI order.
3. Transformer encoder blocks apply multi-head self-attention, residual connections, layer normalization, and GELU feed-forward layers.
4. Mean pooling creates a participant-level representation.
5. A LeakyReLU multilayer classifier produces one ADHD logit.

The implementation is in [`src/adhd_transformer/model.py`](src/adhd_transformer/model.py). The cross-validation training pipeline is in [`train.py`](train.py).

## Data format

The source neuroimaging data are not redistributed in this repository. Prepare a NumPy `.npz` file containing:

- `X`: float array with shape `(n_participants, 116, 116)`
- `y`: binary array with shape `(n_participants,)`

The code uses the explicit convention `1 = ADHD` and `0 = control`.

```python
import numpy as np

np.savez_compressed("data/connectivity.npz", X=connectivity_matrices, y=labels)
```

## Run the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py \
  --data data/connectivity.npz \
  --epochs 400 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --folds 5 \
  --output-dir outputs
```

For a quick smoke test:

```bash
python train.py --data data/connectivity.npz --epochs 2 --folds 2
```

The training script writes per-fold metrics, aggregate metrics, run configuration, and model checkpoints to the output directory. Accuracy, sensitivity, specificity, F1, AUROC, and AUPRC are computed using the same documented label convention.

## Reported project results

The archived course report recorded mean five-fold performance of 80.2% accuracy, 85.0% sensitivity, 25.0% specificity, 0.505 AUROC, and 0.419 AUPRC. The low specificity and near-chance AUROC indicate weak discrimination despite the reported accuracy.

Those values are preserved as historical results rather than generated outputs. The original training source and exact fold assignments were not retained in this repository, so the reference implementation cannot guarantee numerical reproduction. New runs should be interpreted from the generated fold-level metrics, with particular attention to AUROC, AUPRC, specificity, and class imbalance rather than accuracy alone.

## Repository structure

```text
.
├── ADHD Classification Paper.pdf
├── README.md
├── requirements.txt
├── train.py
├── data/
│   └── README.md
├── src/
│   └── adhd_transformer/
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       └── model.py
└── tests/
    └── test_model.py
```

## Project provenance

This was a team project by Keerat Kaur, Julianne Lee, and Shamei Lu. According to the submitted report, Shamei Lu completed data preprocessing, Julianne Lee extracted the functional-connectivity features and adapted, trained, and evaluated the transformer, and Keerat Kaur conducted background research and created and interpreted the performance visualizations. The code in this repository is a later, clean-room reference reconstruction from the documented methodology, not the unrecovered original nine-module implementation.

## Limitations and next steps

- The sample is small and imbalanced for a transformer.
- Static Pearson connectivity may omit time-varying information.
- A single atlas constrains spatial resolution.
- Nested validation or a held-out test cohort is needed for unbiased model selection.
- Useful extensions include class-weighted loss, calibration, SVM/logistic-regression baselines, dynamic connectivity, and external validation.

## Reference

Dai, P. et al. (2024). Classification of MDD using a Transformer classifier with large-scale multisite resting-state fMRI data. *Human Brain Mapping, 45*(1), e26542. https://doi.org/10.1002/hbm.26542
