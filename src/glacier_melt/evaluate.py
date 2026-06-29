"""
glacier_melt.evaluate
=====================
Evaluation metrics and per-glacier analysis.

Two primary metrics are used throughout:

Spatial overlap (recall under calibrated top-N threshold)
    The fraction of observed melt pixels correctly predicted as melt.
    Computed by selecting the top-N pixels by predicted probability,
    where N equals the number of observed melt pixels. This calibration
    ensures the predicted melt area matches the observed melt area, making
    overlap equivalent to recall and directly comparable to the Andriychenko
    (2024) baseline methodology.

    Computed both Peru-wide and per-glacier. Per-glacier calibration is
    applied independently to each glacier (each glacier's top-N count
    matches its own observed melt count).

AUC (area under ROC curve)
    Threshold-independent ranking quality. Probability that a randomly
    chosen melt pixel receives a higher predicted probability than a
    randomly chosen non-melt pixel. AUC = 1 for a perfect ranker.

Note on small glaciers
    Glaciers that completely melted over 2017-2025 have all pixels labelled
    melt=1. For these glaciers, any non-zero prediction achieves 100% overlap
    regardless of spatial accuracy. Filtering by minimum area (e.g. 0.001 km²)
    removes this artefact. With the filter applied, the train/validation
    performance ordering reverses, which motivated raising the minimum area
    threshold in the paper's per-glacier analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import roc_auc_score

from glacier_melt.sampling import LABEL_COL, COORD_COLS

# Pixel area: 10m × 10m = 0.0001 km²
PIXEL_AREA_KM2 = (10 * 10) / 1e6


# ---------------------------------------------------------------------------
# Peru-wide metrics
# ---------------------------------------------------------------------------

def compute_overlap(
    probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Compute spatial overlap (recall under calibrated top-N threshold).

    Selects the top-N pixels by predicted probability where N is the number
    of observed melt pixels, then computes the fraction of observed melt
    pixels that are correctly predicted.

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Predicted melt probabilities.
    labels : np.ndarray, shape (N,)
        Binary ground-truth melt labels (0 = ice, 1 = melt).

    Returns
    -------
    overlap_pct : float
        Overlap as a percentage (0–100).
    iou : float
        Intersection over union.
    """
    actual = labels == 1
    n_melt = actual.sum()

    top_n = np.argsort(probs)[::-1][:n_melt]
    pred = np.zeros(len(probs), dtype=bool)
    pred[top_n] = True

    n_intersect = (pred & actual).sum()
    overlap_pct = n_intersect / n_melt * 100
    iou = n_intersect / (pred | actual).sum()
    return overlap_pct, iou


def compute_auc(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the area under the ROC curve (AUC).

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Predicted melt probabilities.
    labels : np.ndarray, shape (N,)
        Binary ground-truth melt labels.

    Returns
    -------
    float
        AUC in [0, 1].
    """
    return roc_auc_score(labels, probs)


def evaluate_model(
    probs: np.ndarray,
    labels: np.ndarray,
    region_mask: np.ndarray | None = None,
    model_name: str = "",
) -> dict[str, float]:
    """
    Compute overlap, IoU, and AUC for a set of predictions.

    Optionally also computes metrics restricted to a validation region mask,
    reproducing the Peru-wide vs R3 breakdown shown in Table 2 of the paper.

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Predicted melt probabilities over all pixels.
    labels : np.ndarray, shape (N,)
        Ground-truth binary labels.
    region_mask : np.ndarray of bool, shape (N,), or None
        If provided, also computes metrics restricted to pixels where this
        mask is True (e.g. R3 validation region only).
    model_name : str
        Optional label for printed output.

    Returns
    -------
    dict with keys:
        ``'overlap'``     : float — Peru-wide overlap %
        ``'iou'``         : float — Peru-wide IoU
        ``'auc'``         : float — Peru-wide AUC
        ``'val_overlap'`` : float or None — validation-region overlap %
        ``'val_iou'``     : float or None — validation-region IoU
        ``'val_auc'``     : float or None — validation-region AUC
    """
    overlap, iou = compute_overlap(probs, labels)
    auc = compute_auc(probs, labels)

    result = {
        "overlap": overlap,
        "iou": iou,
        "auc": auc,
        "val_overlap": None,
        "val_iou": None,
        "val_auc": None,
    }

    if region_mask is not None:
        val_overlap, val_iou = compute_overlap(
            probs[region_mask], labels[region_mask]
        )
        val_auc = compute_auc(probs[region_mask], labels[region_mask])
        result["val_overlap"] = val_overlap
        result["val_iou"] = val_iou
        result["val_auc"] = val_auc

    prefix = f"{model_name}: " if model_name else ""
    print(
        f"{prefix}Peru-wide: overlap={overlap:.2f}%, "
        f"IoU={iou:.4f}, AUC={auc:.4f}"
    )
    if region_mask is not None:
        print(
            f"{prefix}Val (R3):  overlap={result['val_overlap']:.2f}%, "
            f"IoU={result['val_iou']:.4f}, AUC={result['val_auc']:.4f}"
        )
    return result


def print_summary_table(
    results: dict[str, dict[str, float]],
) -> None:
    """
    Print a formatted summary table of model results.

    Reproduces Table 2 from the paper.

    Parameters
    ----------
    results : dict of model_name → metrics dict
        Output of multiple ``evaluate_model`` calls, keyed by model name.
    """
    print(f"\n{'Model':<20} {'Peru Overlap':>12} {'Peru AUC':>9} "
          f"{'Val Overlap':>12} {'Val AUC':>9}")
    print("-" * 66)
    for name, m in results.items():
        val_ovlp = f"{m['val_overlap']:.2f}%" if m["val_overlap"] is not None else "  N/A"
        val_auc  = f"{m['val_auc']:.4f}"      if m["val_auc"]     is not None else "   N/A"
        print(f"{name:<20} {m['overlap']:>11.2f}% {m['auc']:>9.4f} "
              f"{val_ovlp:>12} {val_auc:>9}")


# ---------------------------------------------------------------------------
# RGI spatial join
# ---------------------------------------------------------------------------

def assign_pixels_to_glaciers(
    df: pd.DataFrame,
    rgi_gdf: gpd.GeoDataFrame,
    coord_cols: tuple[str, str] = ("lon", "lat"),
) -> pd.Series:
    """
    Spatially join pixels to RGI glacier polygons.

    Each pixel is assigned to the RGI glacier polygon that contains it.
    Pixels not falling within any polygon are assigned NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe with coordinate columns.
    rgi_gdf : gpd.GeoDataFrame
        RGI Version 7.0 glacier polygons for Peru (buffered 500 m).
    coord_cols : tuple of str
        Column names for (longitude, latitude) in ``df``.

    Returns
    -------
    pd.Series
        RGI glacier ID for each pixel (same index as ``df``).
        Named ``'rgi_id'``.
    """
    lon_col, lat_col = coord_cols
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    gdf = gpd.GeoDataFrame(
        df[[lon_col, lat_col]].copy(),
        geometry=geometry,
        crs="EPSG:4326",
    )
    rgi_polys = rgi_gdf[["rgi_id", "geometry"]].copy()
    joined = gpd.sjoin(gdf, rgi_polys, how="left", predicate="within")
    return joined["rgi_id"].values


# ---------------------------------------------------------------------------
# Per-glacier overlap
# ---------------------------------------------------------------------------

def per_glacier_metrics(
    df: pd.DataFrame,
    probs: np.ndarray,
    rgi_col: str = "rgi_id",
    label_col: str = LABEL_COL,
    min_area_km2: float | None = 0.001,
    min_pixels: int = 10,
) -> pd.DataFrame:
    """
    Compute overlap and IoU for each individual glacier.

    The top-N calibration is applied per-glacier: for each glacier,
    the N pixels with highest predicted probability are selected as
    predicted melt, where N equals the number of observed melt pixels
    in that glacier. This is equivalent to per-glacier recall under a
    glacier-specific threshold.

    Note on small glaciers
    ----------------------
    Glaciers smaller than ``min_area_km2`` that completely melted over
    2017-2025 have melt_rate = 1.0. For these, any prediction achieves
    100% overlap regardless of spatial accuracy, inflating the mean.
    The default filter (0.001 km²) excludes the most extreme cases.
    Raising the threshold (e.g. 0.1 km²) reverses the train/validation
    ordering seen in the unfiltered data.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe with RGI ID and melt label columns. Must have
        ``rgi_id`` assigned via ``assign_pixels_to_glaciers``.
    probs : np.ndarray, shape (N,)
        Predicted melt probabilities aligned with ``df``.
    rgi_col : str
        Column containing RGI glacier identifiers.
    label_col : str
        Column containing binary melt labels.
    min_area_km2 : float or None
        Exclude glaciers with area below this threshold. Area is inferred
        from pixel count (each pixel is 10m × 10m = 0.0001 km²).
    min_pixels : int
        Minimum number of pixels for a glacier to be included. Glaciers
        with fewer pixels are excluded regardless of area.

    Returns
    -------
    pd.DataFrame
        One row per glacier, columns:
        ``rgi_id``, ``n_pixels``, ``area_km2``, ``n_melt``,
        ``melt_rate``, ``overlap_pct``, ``iou``.
    """
    df = df.copy()
    df["_prob"] = probs

    records = []
    for rgi_id, grp in df.groupby(rgi_col):
        n_total = len(grp)
        n_melt = (grp[label_col] == 1).sum()

        if n_melt == 0 or n_total < min_pixels:
            continue

        area_km2 = n_total * PIXEL_AREA_KM2
        if min_area_km2 is not None and area_km2 < min_area_km2:
            continue

        top_n = np.argsort(grp["_prob"].values)[::-1][:n_melt]
        pred = np.zeros(n_total, dtype=bool)
        pred[top_n] = True
        actual = grp[label_col].values == 1

        n_intersect = (pred & actual).sum()
        overlap_pct = n_intersect / n_melt * 100
        iou = n_intersect / (pred | actual).sum()

        records.append({
            "rgi_id":      rgi_id,
            "n_pixels":    n_total,
            "area_km2":    area_km2,
            "n_melt":      int(n_melt),
            "melt_rate":   n_melt / n_total,
            "overlap_pct": overlap_pct,
            "iou":         iou,
        })

    df_out = pd.DataFrame(records)
    print(f"Per-glacier metrics: {len(df_out):,} glaciers")
    print(f"Mean overlap: {df_out['overlap_pct'].mean():.2f}%")
    print(f"Median overlap: {df_out['overlap_pct'].median():.2f}%")
    return df_out


# ---------------------------------------------------------------------------
# Physical feature aggregation
# ---------------------------------------------------------------------------

def glacier_physical_features(
    df: pd.DataFrame,
    rgi_col: str = "rgi_id",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute mean physical features per glacier from pixel-level data.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe with EASD columns and RGI IDs.
    rgi_col : str
        Column identifying glacier membership.
    feature_cols : list of str or None
        Columns to aggregate. Defaults to EASD columns plus ``mlp_score``
        if present.

    Returns
    -------
    pd.DataFrame
        One row per glacier with mean feature values and pixel count.
        Columns: ``rgi_id``, ``n_pixels``, ``area_km2``,
        ``mean_elevation``, ``mean_slope``, ``mean_aspect``,
        ``mean_edge_distance``, and ``mean_mlp_score`` (if available).
    """
    if feature_cols is None:
        feature_cols = ["elevation", "slope", "aspect", "edge_distance"]
        if "mlp_score" in df.columns:
            feature_cols.append("mlp_score")

    agg = {col: (col, "mean") for col in feature_cols}
    agg["n_pixels"] = ("lon", "count")

    result = df.groupby(rgi_col).agg(**agg).reset_index()
    result.columns = (
        [rgi_col, "n_pixels"]
        + [f"mean_{c}" for c in feature_cols]
    )
    result["area_km2"] = result["n_pixels"] * PIXEL_AREA_KM2
    return result


# ---------------------------------------------------------------------------
# Vulnerability ranking
# ---------------------------------------------------------------------------

def vulnerability_ranking(
    df: pd.DataFrame,
    probs: np.ndarray,
    rgi_col: str = "rgi_id",
    score_col: str = "mlp_score",
) -> pd.DataFrame:
    """
    Rank glaciers by vulnerability to melt.

    Vulnerability is measured as the mean MLP predicted melt probability
    per glacier. Higher mean score = more likely to melt = more vulnerable.
    This is primarily driven by glacier size and edge geometry (fraction of
    pixels near the glacier boundary), not slope.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe with RGI IDs. Must have ``mlp_score`` column
        (predicted probabilities assigned prior to calling this function).
    probs : np.ndarray, shape (N,)
        MLP predicted melt probabilities for each pixel.
    rgi_col : str
        Column identifying glacier membership.
    score_col : str
        Column name to assign probabilities to.

    Returns
    -------
    pd.DataFrame
        One row per glacier, sorted by vulnerability rank (1 = most at risk).
        Columns: ``rgi_id``, ``n_pixels``, ``area_km2``, ``mean_mlp_score``,
        ``vulnerability_rank``.
    """
    df = df.copy()
    df[score_col] = probs

    glacier_scores = (
        df.groupby(rgi_col)[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "mean_mlp_score"})
    )

    pixel_counts = (
        df.groupby(rgi_col)
        .size()
        .reset_index(name="n_pixels")
    )

    result = glacier_scores.merge(pixel_counts, on=rgi_col)
    result["area_km2"] = result["n_pixels"] * PIXEL_AREA_KM2
    result = result.sort_values("mean_mlp_score", ascending=False).reset_index(drop=True)
    result["vulnerability_rank"] = result.index + 1
    return result
