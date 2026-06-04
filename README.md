# Transformer-Based ADHD Classification Using Resting-State fMRI

Deep learning approach for ADHD classification using resting-state functional connectivity data.

---

## Overview

Attention-Deficit/Hyperactivity Disorder (ADHD) is associated with alterations in large-scale brain networks that can be measured through functional connectivity analysis.

This project investigates whether transformer-based deep learning models can distinguish individuals with ADHD from healthy controls using resting-state fMRI data. By leveraging self-attention mechanisms, the model learns relationships between brain regions and identifies connectivity patterns that may serve as predictive biomarkers for ADHD.

---

## Research Question

**Can transformer-based neural networks accurately classify ADHD using resting-state functional connectivity data derived from fMRI recordings?**

---

## Key Concepts

- Resting-State fMRI (rs-fMRI)
- Functional Connectivity
- ADHD Classification
- Transformer Networks
- Self-Attention
- Computational Neuroscience

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

- Accuracy
- F1 Score
- ROC-AUC

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## Results

The transformer model demonstrated the feasibility of using attention-based deep learning approaches for neuroimaging classification tasks and highlighted the potential of functional connectivity as a predictive biomarker for ADHD.

---

## Repository Contents

- `adhd_transformer_classification.ipynb` — model development and experimentation
- `ADHD_Classification_Report.pdf` — final project report
- `transformer/` — supporting model code and components

---

## Future Work

- Improve model generalization with larger datasets
- Explore Graph Neural Networks (GNNs)
- Incorporate explainability and attention visualization
- Compare against traditional machine learning approaches

---

## Author

**Keerat Kaur**

UC San Diego — Cognitive Science (Machine Learning & Neural Computation)

Interests: Neurotechnology, Computational Neuroscience, Brain-Computer Interfaces, Neural Signal Processing, Clinical AI
