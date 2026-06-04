Transformer-Based ADHD Classification Using Resting-State fMRI

Deep learning approach for ADHD classification using resting-state fMRI functional connectivity data.

Overview

Attention-Deficit/Hyperactivity Disorder (ADHD) is associated with alterations in large-scale brain networks that can be measured through functional connectivity analysis. This project investigates whether transformer-based deep learning models can distinguish individuals with ADHD from healthy controls using resting-state functional MRI (rs-fMRI) data.

By leveraging self-attention mechanisms, the model learns relationships between brain regions and identifies connectivity patterns that may serve as biomarkers for ADHD.

Research Question

Can transformer-based neural networks accurately classify ADHD using resting-state functional connectivity data derived from fMRI recordings?

Methods
Data Processing
Resting-state fMRI data collected from ADHD and control participants
Brain parcellation used to define regions of interest (ROIs)
Functional connectivity matrices generated from ROI time series
Connectivity features extracted and prepared for model training
Model Architecture
Transformer encoder architecture
Self-attention mechanisms used to model relationships between brain regions
Classification head for ADHD vs. Control prediction
Evaluation
Training and testing performed on functional connectivity datasets
Performance evaluated using classification metrics such as accuracy, F1-score, and ROC-AUC
Technologies Used
Python
PyTorch
NumPy
Pandas
Scikit-learn
Jupyter Notebook
Repository Structure
adhd-rsfmri-transformer-classification/
│
├── README.md
├── ADHD_Classification_Report.pdf
├── adhd_transformer.ipynb
├── figures/
└── data_processing/
Results

The transformer model was evaluated on functional connectivity features derived from resting-state fMRI data. Results demonstrate the feasibility of applying attention-based deep learning approaches to neuroimaging classification problems and highlight the potential of functional connectivity as a source of predictive biomarkers.

Future Work
Improve model generalization through larger datasets
Explore graph neural networks for brain connectivity analysis
Incorporate explainability methods to identify influential brain regions
Compare transformer performance against traditional machine learning approaches
Author

Keerat Kaur

UC San Diego — Cognitive Science (Machine Learning & Neural Computation)

Interested in neurotechnology, computational neuroscience, neural signal processing, and clinical AI.
