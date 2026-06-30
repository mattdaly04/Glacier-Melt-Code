"""
glacier_melt.sampling
=====================
Data loading, training data selection, and PCA fitting.

This module covers two distinct sampling strategies used in the project:

Strategy 1 (edge-distance stratified):
    Used for initial model testing and RF/PCA models. Samples pixels by
    edge-distance quintile, targeting 20k pixels per (quintile x melt class)
    bin with strict 50/50 melt balance. Consistent with Andriychenko (2024).

Strategy 2 (geographic hold-out):
    Used for all full-embedding models (RF AE, MLP, CNN). A geographic region
    (R3) is withheld entirely for validation. All remaining ~12.2M pixels are
    used for training. No edge-distance stratification, since edge-distance is
    not used as a feature in these models.
"""

from __future__ import annotations

import gc
import time
import joblib
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS = ["r1a", "r1b", "r2", "r3"]
VALIDATION_REGION = "r3"

# AlphaEarth embedding columns are named A00–A63 in the parquet files
EMBEDDING_COLS = [f"A{i:02d}" for i in range(64)]
AE_COLS = EMBEDDING_COLS  # alias used by train.py, projection.py, notebooks
EASD_COLS = ["elevation", "aspect", "slope", "edge_distance"]
LABEL_COL = "melt_label"
COORD_COLS = ["lon", "lat"]

# Strategy 1 parameters
S1_SAMPLES_PER_BIN = 20_000
S1_N_QUINTILES = 5
S1_TRAIN_FRAC = 0.8

# Pixel area
PIXEL_AREA_KM2 = (10e-3) ** 2   # 10m × 10m = 0.0001 km²


# ---------------------------------------------------------------------------
# Data loading — 2017 training parquets
# ---------------------------------------------------------------------------

def load_parquet_regions(
    data_dir: str | Path,
    regions: list[str] = REGIONS,
) -> pd.DataFrame:
    """
    Load and concatenate per-region parquet files from the Merged/ directory.

    Each file is expected at ``{data_dir}/{region}_combined.parquet`` and
    contains one row per 2017 ice pixel with 64 AlphaEarth embedding columns
    (A00–A63), EASD features (elevation, aspect, slope, edge_distance), a
    binary melt label (melt_label), and coordinates (lon, lat).

    Parameters
    ----------
    data_dir : str or Path
        Directory containing per-region parquet files.
    regions : list of str
        Region identifiers to load. Defaults to all four regions
        ('r1a', 'r1b', 'r2', 'r3').

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe with a ``region`` column added to identify
        the source of each row.
    """
    data_dir = Path(data_dir)
    dfs = []
    for region in regions:
        path = data_dir / f"{region}_combined.parquet"
        df = pd.read_parquet(path)
        df["region"] = region
        dfs.append(df)
        print(f"{region}: {len(df):,} pixels, "
              f"{df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(df_all):,} pixels")
    print(f"Memory: {df_all.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"Melt rate: {df_all[LABEL_COL].mean() * 100:.2f}%")
    return df_all


# ---------------------------------------------------------------------------
# Data loading — 2025 simulation parquets
# ---------------------------------------------------------------------------

def load_simulation_parquets(
    sim_dir: str | Path,
    regions: list[str] = REGIONS,
) -> pd.DataFrame:
    """
    Load per-region parquet files from the Sim_Merged/ directory.

    These contain 2025 ice pixels with AlphaEarth 2025 embeddings and
    coordinates, used as input to the future melt projection. Files are
    named ``sim_2025_{region}.parquet``.

    Pixels shared between regions are deduplicated by (lon, lat), with R3
    given priority (loaded first, kept on dedup) to preserve the held-out
    validation region's integrity.

    Parameters
    ----------
    sim_dir : str or Path
        Directory containing simulation parquet files.
    regions : list of str
        Regions to load. R3 is always loaded first for deduplication priority
        regardless of the order specified here.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe of 2025 ice pixels with a ``region`` column.
    """
    sim_dir = Path(sim_dir)

    # R3 first so it wins deduplication
    ordered = ["r3"] + [r for r in regions if r != "r3"]

    dfs = []
    for region in ordered:
        path = sim_dir / f"sim_2025_{region}.parquet"
        df = pd.read_parquet(path)
        df["region"] = region
        dfs.append(df)
        print(f"{region}: {len(df):,} pixels")

    df_sim = pd.concat(dfs, ignore_index=True)
    print(f"\nBefore dedup: {len(df_sim):,}")
    df_sim = df_sim.drop_duplicates(
        subset=["lon", "lat"], keep="first"
    ).reset_index(drop=True)
    print(f"After dedup:  {len(df_sim):,}")
    return df_sim


# ---------------------------------------------------------------------------
# GeoTIFF → parquet conversion (2017 training data)
# ---------------------------------------------------------------------------

def convert_tiffs_to_parquet(
    tiff_dir: str | Path,
    output_dir: str | Path,
    regions: list[str] = REGIONS,
    tiff_pattern: str = "{region}_combined*.tif",
    float32: bool = True,
) -> None:
    """
    Convert GeoTIFF exports from GEE into per-region parquet files.

    Reads all tiff tiles for each region, extracts valid (non-NaN, non-nodata)
    pixels with their lat/lon coordinates, concatenates, casts to float32 to
    reduce memory, and saves as ``{region}_combined.parquet``.

    This is the first step in the data pipeline, run once on the remote
    machine after GEE exports are complete.

    Parameters
    ----------
    tiff_dir : str or Path
        Directory containing the exported GeoTIFF tiles.
    output_dir : str or Path
        Directory to write the output parquet files.
    regions : list of str
        Region identifiers to process.
    tiff_pattern : str
        Glob pattern for tiff tiles, with ``{region}`` as a placeholder.
    float32 : bool
        If True, casts all float64 columns to float32 before saving.
        Roughly halves file size with negligible precision loss for
        embedding values.
    """
    tiff_dir = Path(tiff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partial"
    partial_dir.mkdir(exist_ok=True)

    for region in regions:
        print(f"\n{'=' * 50}")
        print(f"Processing {region.upper()}...")

        # Clear partial dir between regions
        for f in partial_dir.glob("*.parquet"):
            f.unlink()

        pattern = tiff_pattern.format(region=region.upper())
        tile_paths = sorted(tiff_dir.glob(pattern))
        print(f"Found {len(tile_paths)} tiles")

        if not tile_paths:
            print(f"No tiles found for {region}, skipping")
            continue

        t_start = time.time()
        for i, path in enumerate(tile_paths):
            out_path = partial_dir / f"tile_{i:04d}.parquet"
            if out_path.exists():
                print(f"  [{i+1}/{len(tile_paths)}] {path.name}: skipped")
                continue

            t0 = time.time()
            try:
                with rasterio.open(path) as src:
                    band_names = list(src.descriptions)
                    transform = src.transform
                    all_bands = src.read()   # (n_bands, height, width)

                    first_band = all_bands[0]
                    valid_mask = ~np.isnan(first_band) & (first_band != -9999)
                    n_valid = valid_mask.sum()

                    if n_valid == 0:
                        pd.DataFrame().to_parquet(out_path)
                        del all_bands
                        gc.collect()
                        continue

                    rows_idx, cols_idx = np.where(valid_mask)
                    data = all_bands[:, rows_idx, cols_idx]  # (n_bands, n_valid)
                    lons, lats = rasterio.transform.xy(
                        transform, rows_idx, cols_idx
                    )

                    df_tile = pd.DataFrame(data.T, columns=band_names)
                    df_tile["lon"] = lons
                    df_tile["lat"] = lats
                    df_tile.to_parquet(out_path)

                    del all_bands, data, df_tile
                    gc.collect()

                elapsed = time.time() - t_start
                avg = elapsed / (i + 1)
                eta = avg * (len(tile_paths) - i - 1)
                print(f"  [{i+1}/{len(tile_paths)}] {path.name}: "
                      f"{n_valid:,} px ({time.time()-t0:.1f}s, "
                      f"ETA {eta/60:.1f}min)")

            except Exception as e:
                print(f"  FAILED: {path.name}: {e}")
                continue

        # Concatenate tiles for this region
        print(f"\nConcatenating tiles for {region.upper()}...")
        parquet_files = sorted(partial_dir.glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in parquet_files if pd.read_parquet(f).shape[0] > 0]

        if not dfs:
            print(f"No data found for {region}")
            continue

        df_region = pd.concat(dfs, ignore_index=True)

        if float32:
            for col in df_region.columns:
                if df_region[col].dtype == "float64":
                    df_region[col] = df_region[col].astype("float32")

        out_path = output_dir / f"{region}_combined.parquet"
        df_region.to_parquet(out_path, compression="snappy", index=False)
        print(f"Saved: {out_path.name} ({len(df_region):,} pixels, "
              f"{out_path.stat().st_size / 1e9:.2f} GB)")

        del df_region, dfs
        gc.collect()

    print("\nAll regions done!")


# ---------------------------------------------------------------------------
# GeoTIFF → parquet conversion (2025 simulation data)
# ---------------------------------------------------------------------------

def convert_sim_tiffs_to_parquet(
    tiff_dir: str | Path,
    output_dir: str | Path,
    regions: list[str] = REGIONS,
) -> None:
    """
    Convert 2025 simulation GeoTIFF exports into per-region parquet files.

    Identical pipeline to ``convert_tiffs_to_parquet`` but for the simulation
    tiffs (named ``sim_pixels_{REGION}_2025*.tif``), writing output files
    named ``sim_2025_{region}.parquet``.

    Parameters
    ----------
    tiff_dir : str or Path
        Directory containing simulation GeoTIFF tiles (sim_tiffs/).
    output_dir : str or Path
        Directory to write output parquet files (Sim_Merged/).
    regions : list of str
        Regions to process.
    """
    tiff_dir = Path(tiff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "partial"
    partial_dir.mkdir(exist_ok=True)

    for region in regions:
        print(f"\n{'=' * 50}")
        print(f"Processing {region.upper()}...")

        for f in partial_dir.glob("*.parquet"):
            f.unlink()

        tile_paths = sorted(
            tiff_dir.glob(f"sim_pixels_{region.upper()}_2025*.tif")
        )
        print(f"Found {len(tile_paths)} tiles")

        if not tile_paths:
            print(f"No tiles found for {region}, skipping")
            continue

        t_start = time.time()
        for i, path in enumerate(tile_paths):
            out_path = partial_dir / f"tile_{i:04d}.parquet"
            if out_path.exists():
                print(f"  [{i+1}/{len(tile_paths)}] {path.name}: skipped")
                continue

            t0 = time.time()
            try:
                with rasterio.open(path) as src:
                    band_names = list(src.descriptions)
                    transform = src.transform
                    all_bands = src.read()

                    first_band = all_bands[0]
                    valid_mask = ~np.isnan(first_band) & (first_band != -9999)
                    n_valid = valid_mask.sum()

                    if n_valid == 0:
                        pd.DataFrame().to_parquet(out_path)
                        del all_bands
                        gc.collect()
                        continue

                    rows_idx, cols_idx = np.where(valid_mask)
                    data = all_bands[:, rows_idx, cols_idx]
                    lons, lats = rasterio.transform.xy(
                        transform, rows_idx, cols_idx
                    )

                    df_tile = pd.DataFrame(data.T, columns=band_names)
                    df_tile["lon"] = lons
                    df_tile["lat"] = lats
                    df_tile.to_parquet(out_path)

                    del all_bands, data, df_tile
                    gc.collect()

                elapsed = time.time() - t_start
                avg = elapsed / (i + 1)
                eta = avg * (len(tile_paths) - i - 1)
                print(f"  [{i+1}/{len(tile_paths)}] {path.name}: "
                      f"{n_valid:,} px ({time.time()-t0:.1f}s, "
                      f"ETA {eta/60:.1f}min)")

            except Exception as e:
                print(f"  FAILED: {path.name}: {e}")
                continue

        print(f"\nConcatenating tiles for {region.upper()}...")
        parquet_files = sorted(partial_dir.glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in parquet_files
               if pd.read_parquet(f).shape[0] > 0]

        if not dfs:
            print(f"No data found for {region}")
            continue

        df_region = pd.concat(dfs, ignore_index=True)
        out_path = output_dir / f"sim_2025_{region}.parquet"
        df_region.to_parquet(out_path, index=False)
        print(f"Saved: {out_path.name} ({len(df_region):,} pixels)")

        del df_region, dfs
        gc.collect()

    print("\nAll regions done!")


# ---------------------------------------------------------------------------
# Strategy 1: edge-distance stratified sampling
# ---------------------------------------------------------------------------

def strategy1_sample(
    df: pd.DataFrame,
    samples_per_bin: int = S1_SAMPLES_PER_BIN,
    n_quintiles: int = S1_N_QUINTILES,
    train_frac: float = S1_TRAIN_FRAC,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified sampling by edge-distance quintile (Strategy 1).

    Splits edge_distance into ``n_quintiles`` equal-frequency bins, then
    samples up to ``samples_per_bin`` melt and non-melt pixels from each bin,
    enforcing a strict 50/50 melt balance within each quintile. The scarcity
    of pixels in the rarest (quintile x melt class) cell limits the overall
    sample size. The combined sample is then randomly split into train/val.

    Used for: RF baseline (EASD), RF + AE PCA, PCA exploration models.

    Parameters
    ----------
    df : pd.DataFrame
        Full pixel dataframe containing ``edge_distance`` and ``melt_label``
        columns.
    samples_per_bin : int
        Target number of pixels per (quintile x melt class) cell.
    n_quintiles : int
        Number of edge-distance quintile bands.
    train_frac : float
        Fraction of the sampled data to use for training.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    train : pd.DataFrame
    val : pd.DataFrame
    """
    rng = np.random.default_rng(random_state)

    df = df.copy()
    df["_quintile"] = pd.qcut(
        df["edge_distance"], q=n_quintiles, labels=False, duplicates="drop"
    )

    sampled_parts = []
    for q in range(n_quintiles):
        q_mask = df["_quintile"] == q
        for label in [0, 1]:
            pool = df[q_mask & (df[LABEL_COL] == label)]
            n = min(samples_per_bin, len(pool))
            sampled_parts.append(pool.sample(n=n, random_state=int(rng.integers(1e6))))

    df_sampled = pd.concat(sampled_parts, ignore_index=True).drop(
        columns=["_quintile"]
    )
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    n_train = int(len(df_sampled) * train_frac)
    train = df_sampled.iloc[:n_train].reset_index(drop=True)
    val = df_sampled.iloc[n_train:].reset_index(drop=True)

    print(f"Strategy 1: {len(df_sampled):,} total pixels "
          f"({len(train):,} train, {len(val):,} val)")
    print(f"Train melt rate: {train[LABEL_COL].mean()*100:.1f}%, "
          f"Val melt rate: {val[LABEL_COL].mean()*100:.1f}%")
    return train, val


# ---------------------------------------------------------------------------
# Strategy 2: geographic hold-out
# ---------------------------------------------------------------------------

def strategy2_split(
    df: pd.DataFrame,
    validation_region: str = VALIDATION_REGION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Geographic hold-out split (Strategy 2).

    Withholds all pixels from ``validation_region`` (R3, containing 158
    glaciers and ~1.8M pixels) for validation. All remaining pixels (~12.2M
    across R1a, R1b, R2) form the training set.

    This approach prevents spatial data leakage arising from random
    pixel-level splitting of spatially correlated data (Roberts et al. 2017),
    giving a more honest estimate of cross-regional generalisation.

    Used for: RF AE, MLP, CNN 3x3, CNN 5x5 full-embedding models.

    Parameters
    ----------
    df : pd.DataFrame
        Full pixel dataframe with a ``region`` column.
    validation_region : str
        Region identifier to hold out. Default is ``'r3'``.

    Returns
    -------
    train : pd.DataFrame
    val : pd.DataFrame
    """
    val = df[df["region"] == validation_region].reset_index(drop=True)
    train = df[df["region"] != validation_region].reset_index(drop=True)

    print(f"Strategy 2 split: {len(train):,} train, {len(val):,} val "
          f"(held-out region: {validation_region})")
    print(f"Train melt rate: {train[LABEL_COL].mean()*100:.1f}%, "
          f"Val melt rate: {val[LABEL_COL].mean()*100:.1f}%")
    return train, val


# ---------------------------------------------------------------------------
# CNN patch construction
# ---------------------------------------------------------------------------

def build_patch_index(
    coords: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """
    Build a k-d tree index mapping each pixel to its N×N spatial patch.

    For each pixel, identifies the surrounding ``patch_size × patch_size``
    neighbourhood by querying a k-d tree over pixel coordinates. Pixels at
    glacier boundaries whose patches extend outside the glacier mask are
    zero-padded, since AlphaEarth embeddings are only available for ice pixels.
    Padded positions are indicated by index -1 in the returned array.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        (lon, lat) coordinates of all training pixels.
    patch_size : int
        Side length of the square patch (3 or 5 in this study).

    Returns
    -------
    np.ndarray, shape (N, patch_size * patch_size)
        Row indices into the coordinate array giving the patch neighbours
        for each pixel. Padded entries (boundary pixels) are -1.
    """
    tree = KDTree(coords)
    n_pixels = len(coords)
    n_neighbours = patch_size * patch_size

    # Query for patch_size^2 nearest neighbours (includes self)
    distances, indices = tree.query(coords, k=n_neighbours)

    # Estimate the expected nearest-neighbour distance (pixel spacing)
    # from the median 2nd-nearest-neighbour distance
    pixel_spacing = np.median(distances[:, 1])
    max_patch_radius = pixel_spacing * (patch_size // 2 + 0.5) * np.sqrt(2)

    # Mark neighbours that are too far away as padding (-1)
    patch_indices = indices.copy()
    patch_indices[distances > max_patch_radius] = -1

    return patch_indices


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def fit_pca(
    df: pd.DataFrame,
    n_components: int | None = 64,
    embedding_cols: list[str] = EMBEDDING_COLS,
    save_path: str | Path | None = None,
    random_state: int = 42,
) -> tuple[PCA, StandardScaler]:
    """
    Fit a PCA and standard scaler on AlphaEarth embeddings.

    Fits a ``StandardScaler`` first (AE embeddings are not zero-mean across
    pixels), then fits PCA on the scaled embeddings. Optionally saves both
    fitted objects to disk with joblib.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe containing the 64 embedding columns (A00–A63).
    n_components : int or None
        Number of principal components to retain. Default 64 (all).
    embedding_cols : list of str
        Column names for the 64-dimensional AE embedding vector.
    save_path : str, Path, or None
        If provided, saves fitted objects as ``{save_path}_pca.joblib``
        and ``{save_path}_scaler.joblib``.
    random_state : int
        Random seed passed to PCA.

    Returns
    -------
    pca : sklearn.decomposition.PCA
    scaler : sklearn.preprocessing.StandardScaler
    """
    t0 = time.time()
    X = df[embedding_cols].values.astype(np.float32)
    print(f"AE matrix: {X.shape}, {X.nbytes / 1e9:.2f} GB")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Standardised in {time.time()-t0:.1f}s")

    t1 = time.time()
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_scaled)
    print(f"PCA fit in {time.time()-t1:.1f}s")

    # Print cumulative variance summary
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    for k in [1, 3, 6, 10, 20, 32, 64]:
        if k <= len(cum_var):
            print(f"  PC1-PC{k:2d}: {cum_var[k-1]:5.2f}% cumulative variance")

    if save_path is not None:
        save_path = Path(save_path)
        joblib.dump(pca, str(save_path) + "_pca.joblib")
        joblib.dump(scaler, str(save_path) + "_scaler.joblib")
        print(f"Saved PCA and scaler to {save_path}_*.joblib")

    return pca, scaler


def transform_pca(
    df: pd.DataFrame,
    pca: PCA,
    scaler: StandardScaler,
    embedding_cols: list[str] = EMBEDDING_COLS,
    n_components: int | None = None,
) -> np.ndarray:
    """
    Apply a fitted scaler + PCA to embedding columns.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe containing the embedding columns.
    pca : PCA
        Fitted PCA object from ``fit_pca``.
    scaler : StandardScaler
        Fitted scaler object from ``fit_pca``.
    embedding_cols : list of str
        Column names for the embedding vector.
    n_components : int or None
        How many components to return. None returns all retained by the PCA.

    Returns
    -------
    np.ndarray, shape (N, n_components)
        Transformed embedding matrix.
    """
    X = df[embedding_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    if n_components is not None:
        X_pca = X_pca[:, :n_components]
    return X_pca


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check(df: pd.DataFrame, embedding_cols: list[str] = EMBEDDING_COLS) -> None:
    """
    Run basic sanity checks on a loaded pixel dataframe and print a report.

    Checks: shape, null counts, AE band statistics, melt/non-melt class
    separation, and whether any AE bands have near-zero variance (which
    would indicate a broken export).

    Parameters
    ----------
    df : pd.DataFrame
        Loaded pixel dataframe.
    embedding_cols : list of str
        AE band column names to inspect.
    """
    print("=== STRUCTURE ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n=== NULLS ===")
    nulls = df.isna().sum()
    nulls_nonzero = nulls[nulls > 0]
    if len(nulls_nonzero) == 0:
        print("No nulls anywhere ✓")
    else:
        print(f"Columns with nulls:\n{nulls_nonzero}")

    print("\n=== ALPHAEARTH SUMMARY ===")
    ae_data = df[embedding_cols].values
    print(f"Overall min: {ae_data.min():.4f}")
    print(f"Overall max: {ae_data.max():.4f}")
    print(f"Overall mean: {ae_data.mean():.4f}")
    print(f"Overall std: {ae_data.std():.4f}")

    print("\n=== CONSTANT BANDS CHECK ===")
    band_stds = df[embedding_cols].std()
    near_zero = band_stds[band_stds < 1e-6]
    if len(near_zero) > 0:
        print(f"WARNING: {len(near_zero)} bands have near-zero std:")
        print(near_zero)
    else:
        print(f"All {len(embedding_cols)} AE bands have meaningful variance ✓")
        print(f"Std range: {band_stds.min():.4f} to {band_stds.max():.4f}")

    print("\n=== ICE/MELT SEPARATION ===")
    melt_means = df[df[LABEL_COL] == 1][embedding_cols].mean()
    nonmelt_means = df[df[LABEL_COL] == 0][embedding_cols].mean()
    mean_diff = (melt_means - nonmelt_means).abs()
    print("Bands with largest mean difference (melt vs non-melt):")
    print(mean_diff.nlargest(5).round(4))
