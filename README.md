# Transformer-Based ADHD Classification Using Resting-State fMRI

Deep learning approach for ADHD classification using resting-state fMRI functional connectivity data.

---

## Overview

Attention-Deficit/Hyperactivity Disorder (ADHD) is associated with alterations in large-scale brain networks that can be measured through functional connectivity analysis.

This project investigates whether transformer-based deep learning models can distinguish individuals with ADHD from healthy controls using resting-state functional MRI (rs-fMRI) data.

By leveraging self-attention mechanisms, the model learns relationships between brain regions and identifies connectivity patterns that may serve as predictive biomarkers for ADHD.

---

## Abstract

This project explores the use of transformer-based neural networks for ADHD classification using resting-state functional connectivity data derived from fMRI recordings. Functional connectivity matrices were generated from neuroimaging data and used as input to a transformer encoder architecture. The project evaluates whether attention-based deep learning models can identify meaningful patterns in brain connectivity associated with ADHD and demonstrates the potential of transformer models for computational psychiatry and neuroimaging classification tasks.

---

## Research Question

**Can transformer-based neural networks accurately classify ADHD using resting-state functional connectivity data derived from resting-state fMRI recordings?**

---

## Key Concepts

- Resting-State fMRI (rs-fMRI)
- Functional Connectivity
- ADHD Classification
- Transformer Networks
- Self-Attention
- Computational Neuroscience
- Neuroimaging
- Deep Learning

---

## Skills & Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook
- Functional Connectivity Analysis
- Resting-State fMRI
- Deep Learning
- Computational Neuroscience

---

## Dataset

- Resting-state fMRI data from ADHD and healthy control participants
- ROI-based brain parcellation
- Functional connectivity matrices generated from regional time series
- Neuroimaging features used for machine learning classification

---

## Methods

### Data Processing

- Preprocessed resting-state fMRI recordings
- Generated functional connectivity matrices from ROI time series
- Extracted and normalized connectivity features for model training

### Model Development

- Implemented a Transformer encoder architecture
- Applied self-attention mechanisms to model relationships between brain regions
- Trained a binary classifier to distinguish ADHD and healthy control participants

### Evaluation

Model performance was evaluated using:

- Accuracy
- F1 Score
- ROC-AUC

---

## Project Workflow

![Project Workflow](figures/workflow.png)

---

## Results

Key findings from the project include:

- Transformer models successfully learned patterns from functional connectivity data.
- Attention-based architectures demonstrated potential for neuroimaging classification tasks.
- Functional connectivity features contained predictive information relevant to ADHD classification.
- Results support the use of deep learning approaches for computational psychiatry applications.

### Model Performance

![ROC Curve](figures/roc_curve.png)

![Confusion Matrix](figures/confusion_matrix.png)

---

## Repository Contents

| File | Description |
|--------|--------|
| `adhd_transformer_classification.ipynb` | Data preprocessing, model development, training, and evaluation |
| `ADHD_Classification_Report.pdf` | Final project report |
| `transformer/` | Supporting model code and components |
| `figures/` | Project visualizations and results |

---

## Project Report

📄 [Read the Full Report](ADHD_Classification_Report.pdf)

---

## Project Status

Completed as part of **BIPN 162: Neural Data Science** at the University of California, San Diego.

---

## Future Work

- Improve model generalization using larger neuroimaging datasets
- Explore Graph Neural Networks (GNNs) for brain connectivity analysis
- Incorporate explainability methods and attention visualization
- Compare transformer performance against traditional machine learning approaches
- Investigate clinically interpretable biomarkers for ADHD

---

## Citation

If you use or reference this work, please cite:

> Kaur, K. (2026). *Transformer-Based ADHD Classification Using Resting-State fMRI*. University of California, San Diego.

---

## Author

**Keerat Kaur**

University of California, San Diego  
B.S. Cognitive Science — Machine Learning & Neural Computation

### Interests

- Neurotechnology
- Computational Neuroscience
- Brain-Computer Interfaces
- Neural Signal Processing
- Clinical AI

---

### Related Projects

- EEG Channel Reduction for Brain-Computer Interfaces
- Motor Intent Detection Using EEG Event-Related Desynchronization
- LLM Medical Schema Probing
