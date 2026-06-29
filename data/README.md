# Data

This directory contains small reference files that are tracked in git.
Large data files (parquets, model weights) are hosted externally — see below.

## Files in this directory

| File | Description |
|------|-------------|
| `rgi70_peru.geojson` | RGI Version 7.0 glacier outlines for Peru (500 m buffered). Source: RGI Consortium (2023). doi:10.5067/F6JMOVY5NAVZ |

## External data (Zenodo)

The following large files are archived at [Zenodo record TBD]:

| File(s) | Description | Size (approx.) |
|---------|-------------|----------------|
| `Merged/r1a.parquet`, `r1b.parquet`, `r2.parquet`, `r3.parquet` | 2017 ice pixel features: 64-dim AlphaEarth embeddings, EASD features (elevation, aspect, slope, edge distance), binary melt label, lat/lon coordinates | ~GB |
| `Sim_Merged/r1a.parquet`, `r1b.parquet`, `r2.parquet`, `r3.parquet` | 2025 ice pixel features: 64-dim AlphaEarth embeddings and coordinates, used for future melt simulation | ~GB |
| `Predictions/predictions_full_peru_r3holdout.parquet` | Per-pixel predicted melt probabilities for all models (RF EASD, RF AE, MLP, CNN 3x3, CNN 5x5) | ~GB |
| `Final_Models/mlp_final.pt` | Trained MLP weights (PyTorch state dict) | ~MB |
| `Final_Models/rf_baseline.joblib` | Trained RF baseline (scikit-learn) | ~MB |
| `Final_Models/cnn_3x3.pt` | Trained CNN 3x3 weights | ~MB |

## Reproducing the data from scratch

All large files can be reproduced from scratch using the pipeline:

1. **GEE classification** (`gee/classification.py`): Reproduces the annual glacier coverage dataset extending Elliott (2024) to 2025 using Sentinel-2 imagery.
2. **GEE export** (`gee/export_training.py`, `gee/export_simulation.py`): Exports AlphaEarth embeddings, EASD features, and labels as GeoTIFF files.
3. **Parquet conversion** (`notebooks/01_pca.ipynb` / `src/glacier_melt/sampling.py`): Converts GeoTIFFs to per-region parquet files.
4. **Model training** (`notebooks/02_training.ipynb` / `src/glacier_melt/train.py`): Trains all models and saves checkpoints.

See the root `README.md` for full pipeline instructions.

## Data access requirements

- **Google Earth Engine**: A GEE account is required to reproduce steps 1–2. AlphaEarth embeddings are accessed via the GEE asset `projects/google/models/alpha_earth_foundations/v1`.
- **Elliott (2024) classifications**: Available from the Cambridge group on request. Required for step 1 (used as RF training labels for the coverage classifier).
