"""
glacier_melt
============
A Python package for tracking and predicting glacial melt in the
Peruvian Andes using AlphaEarth geospatial embeddings and machine learning.

Companion code for:
    Daly, M. (2026). Tracking and Predicting Glacial Melt in the Peruvian
    Andes Using Machine Learning. MSci Thesis, University of Cambridge.

Modules
-------
sampling    : data loading, stratified sampling, PCA fitting
models      : MLP and CNN architecture definitions (PyTorch)
train       : training loops and checkpoint saving
evaluate    : spatial overlap, AUC, per-glacier metrics
projection  : future melt simulation under linear retreat assumption
visualise   : map plots, histograms, vulnerability figures
"""

__version__ = "0.1.0"
__author__ = "Matt Daly"

from glacier_melt import sampling, models, train, evaluate, projection, visualise

__all__ = ["sampling", "models", "train", "evaluate", "projection", "visualise"]
