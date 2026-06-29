"""
glacier_melt.visualise
======================
Figure generation for all paper plots.

All functions return a ``matplotlib.figure.Figure`` so the caller controls
saving and display. No function calls ``plt.show()`` directly.

Pixel rendering
---------------
Glacier map plots use a custom rasteriser (``_plot_as_grid``) that converts
pixel coordinates to a Web Mercator image array and overlays it on a
contextily basemap. This avoids the performance penalty of per-pixel scatter
plots on figures containing millions of points, and produces a clean pixel-
accurate rendering that matches the 10m satellite resolution.

Coordinate system
-----------------
All data is stored in WGS84 (EPSG:4326). Map axes are in Web Mercator
(EPSG:3857) to match contextily basemap tiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.decomposition import PCA

import contextily as ctx

from glacier_melt.projection import PIXEL_AREA_KM2, SCENARIOS, TIMESTEPS

Figure = matplotlib.figure.Figure

# Colours for glacier map plots
OBSERVED_PREDICTED_COLOURS = {
    "both":         "#008722",   # true positive  — green
    "actual_only":  "#ffb700",   # false negative — amber
    "pred_only":    "#ff0d00",   # false positive — red
    "neither":      "#ffffff",   # surviving ice  — white
}

FUTURE_MELT_COLOURS = {
    2035: "#d73027",
    2045: "#fdae61",
    2055: "#fee090",
    2065: "#abd9e9",
}
OBSERVED_MELT_COLOUR = "#1a1a4e"   # dark navy — observed 2017-2025 melt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Convert a single WGS84 point to Web Mercator coordinates."""
    pt = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    return float(pt.geometry.x[0]), float(pt.geometry.y[0])


def _mercator_corners(
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
) -> tuple[float, float, float, float]:
    """Convert a WGS84 bounding box to Web Mercator (x_min, x_max, y_min, y_max)."""
    corners = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [lon_min, lon_max], [lat_min, lat_max]
        ), crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    xs = corners.geometry.x.values
    ys = corners.geometry.y.values
    return xs.min(), xs.max(), ys.min(), ys.max()


def _square_extent(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> tuple[float, float, float, float]:
    """Pad the shorter axis so the extent is square in Web Mercator."""
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        diff = (x_range - y_range) / 2
        y_min -= diff
        y_max += diff
    else:
        diff = (y_range - x_range) / 2
        x_min -= diff
        x_max += diff
    return x_min, x_max, y_min, y_max


def _plot_as_grid(
    df_pixels: pd.DataFrame,
    ax: plt.Axes,
    lon_min: float, lat_min: float,
    lon_max: float, lat_max: float,
    resolution: float = 0.00009155,
) -> None:
    """
    Rasterise pixel-level colour data and overlay on a Web Mercator axis.

    Converts each pixel's (lon, lat) and ``plot_colour`` to an RGBA image
    array and renders it with ``imshow``, producing a pixel-accurate map
    overlay without per-point scatter plot overhead.

    Parameters
    ----------
    df_pixels : pd.DataFrame
        Must have columns ``lon``, ``lat``, ``plot_colour`` (hex or named
        colour string, or None to skip the pixel).
    ax : plt.Axes
        Axis in Web Mercator (EPSG:3857) projection.
    lon_min, lat_min, lon_max, lat_max : float
        Load bounding box in WGS84 degrees.
    resolution : float
        Grid cell size in degrees. Default matches the 10m pixel spacing
        at Andean latitudes (~0.00009155°).
    """
    n_cols = int(np.ceil((lon_max - lon_min) / resolution)) + 1
    n_rows = int(np.ceil((lat_max - lat_min) / resolution)) + 1

    img = np.zeros((n_rows, n_cols, 4), dtype=np.float32)

    col_idx = np.floor(
        (df_pixels["lon"].values - lon_min) / resolution
    ).astype(int)
    row_idx = np.floor(
        (lat_max - df_pixels["lat"].values) / resolution
    ).astype(int)

    valid = (
        (col_idx >= 0) & (col_idx < n_cols) &
        (row_idx >= 0) & (row_idx < n_rows)
    )

    for i in np.where(valid)[0]:
        colour = df_pixels["plot_colour"].iloc[i]
        if colour is not None:
            img[row_idx[i], col_idx[i]] = mcolors.to_rgba(colour)

    x_min, x_max, y_min, y_max = _mercator_corners(
        lon_min, lat_min, lon_max, lat_max
    )
    ax.imshow(
        img, extent=[x_min, x_max, y_min, y_max],
        origin="upper", aspect="auto",
        interpolation="nearest", zorder=3,
    )


def _add_map_furniture(
    ax: plt.Axes,
    lon_min_display: float, lat_min_display: float,
    lon_max_display: float, lat_max_display: float,
) -> None:
    """
    Add north arrow, scale bar, gridlines, and tick labels to a map axis.

    Parameters
    ----------
    ax : plt.Axes
        Web Mercator axis.
    lon_min_display, lat_min_display, lon_max_display, lat_max_display : float
        Display bounding box in WGS84 degrees (controls tick spacing).
    """
    # North arrow
    ax.annotate(
        "N", xy=(0.95, 0.92), xytext=(0.95, 0.82),
        xycoords="axes fraction",
        fontsize=10, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
    )

    # Scale bar
    scalebar = ScaleBar(
        1, units="m", dimension="si-length",
        length_fraction=0.25, location="lower right",
        box_alpha=0.7, color="black",
        font_properties={"size": 7},
    )
    ax.add_artist(scalebar)
    ax.set_aspect(1)

    # Gridlines and tick labels
    lon_extent = lon_max_display - lon_min_display
    if lon_extent < 0.05:
        tick_spacing = 0.01
    elif lon_extent < 0.15:
        tick_spacing = 0.02
    else:
        tick_spacing = 0.05

    lon_ticks = np.arange(
        np.ceil(lon_min_display / tick_spacing) * tick_spacing,
        lon_max_display, tick_spacing,
    )
    lat_ticks = np.arange(
        np.ceil(lat_min_display / tick_spacing) * tick_spacing,
        lat_max_display, tick_spacing,
    )

    ax.set_xticks([_to_mercator(lon, 0)[0] for lon in lon_ticks])
    ax.set_yticks([_to_mercator(0, lat)[1] for lat in lat_ticks])
    ax.set_xticklabels([f"{lon:.2f}°" for lon in lon_ticks], fontsize=6)
    ax.set_yticklabels([f"{lat:.2f}°" for lat in lat_ticks], fontsize=6)
    ax.tick_params(left=True, bottom=True, labelleft=True,
                   labelbottom=True, labelsize=6, length=3)
    ax.grid(True, color="gray", linewidth=0.5, alpha=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_zorder(10)
        spine.set_visible(True)


def _glacier_bbox(
    df: pd.DataFrame,
    pad: float,
    load_pad: float,
    lon_offset: float = 0.0,
    lat_offset: float = 0.0,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Compute display and load bounding boxes for a glacier.

    Returns
    -------
    lon_min_display, lat_min_display, lon_max_display, lat_max_display,
    lon_min, lat_min, lon_max, lat_max
    """
    centre_lon = df["lon"].mean() + lon_offset
    centre_lat = df["lat"].mean() + lat_offset
    half_lon = (df["lon"].max() - df["lon"].min()) / 2
    half_lat = (df["lat"].max() - df["lat"].min()) / 2

    lon_min_d = centre_lon - half_lon - pad
    lon_max_d = centre_lon + half_lon + pad
    lat_min_d = centre_lat - half_lat - pad
    lat_max_d = centre_lat + half_lat + pad

    lon_min = centre_lon - half_lon - load_pad
    lon_max = centre_lon + half_lon + load_pad
    lat_min = centre_lat - half_lat - load_pad
    lat_max = centre_lat + half_lat + load_pad

    return (lon_min_d, lat_min_d, lon_max_d, lat_max_d,
            lon_min, lat_min, lon_max, lat_max)


# ---------------------------------------------------------------------------
# PCA scree plot
# ---------------------------------------------------------------------------

def plot_pca_scree(
    pca: PCA,
    n_components_highlight: int = 10,
) -> Figure:
    """
    Plot PCA explained variance ratio (scree) and cumulative variance.

    Reproduces Figure 1 of the paper: two panels showing per-component
    explained variance and cumulative explained variance, with a vertical
    marker at ``n_components_highlight`` (10 components explain ~80% of
    variance in the AE embedding space).

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object (all 64 components retained).
    n_components_highlight : int
        Component count to annotate with a vertical line. Default 10.

    Returns
    -------
    Figure
    """
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    axes[0].bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_ * 100,
        color="steelblue", edgecolor="black", linewidth=0.5,
    )
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title(
        "Scree plot — AlphaEarth embeddings (Peru-wide ice pixels)"
    )
    axes[0].grid(axis="y", alpha=0.3)

    # Cumulative variance
    axes[1].plot(
        range(1, len(cum_var) + 1), cum_var,
        color="darkorange", linewidth=2, marker="o", markersize=4,
    )
    for threshold in [80, 90, 95]:
        axes[1].axhline(threshold, linestyle="--", color="red",
                        alpha=0.4, label=f"{threshold}%")
    axes[1].axvline(
        n_components_highlight, linestyle=":", color="gray", alpha=0.6,
        label=f"PC{n_components_highlight}: {cum_var[n_components_highlight-1]:.1f}%",
    )
    axes[1].set_xlabel("Number of principal components")
    axes[1].set_ylabel("Cumulative variance explained (%)")
    axes[1].set_title("Cumulative variance — AlphaEarth embeddings")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, len(cum_var) + 1)
    axes[1].set_ylim(0, 102)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Overlap histogram
# ---------------------------------------------------------------------------

def plot_overlap_histogram(
    df_overlap: pd.DataFrame,
    df_melt: pd.DataFrame,
    overlap_col: str = "overlap_pct",
    val_region: str = "r3",
) -> Figure:
    """
    Histogram of per-glacier overlap, split by training and validation regions.

    Parameters
    ----------
    df_overlap : pd.DataFrame
        Output of ``evaluate.per_glacier_metrics``, with a ``rgi_id`` column.
    df_melt : pd.DataFrame
        Simulation melt year dataframe, used to identify R3 glacier IDs.
    overlap_col : str
        Column to plot. Default ``'overlap_pct'``.
    val_region : str
        Region identifier for the validation set. Default ``'r3'``.

    Returns
    -------
    Figure
    """
    r3_ids = df_melt[df_melt["region"] == val_region]["rgi_id"].unique()
    df_r3 = df_overlap[df_overlap["rgi_id"].isin(r3_ids)]
    df_train = df_overlap[~df_overlap["rgi_id"].isin(r3_ids)]

    bins = np.linspace(0, 100, 21)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(df_train[overlap_col], bins=bins, alpha=0.6,
            color="steelblue", edgecolor="black", linewidth=0.5,
            label=f"Training regions (n={len(df_train)})")
    ax.hist(df_r3[overlap_col], bins=bins, alpha=0.6,
            color="orange", edgecolor="black", linewidth=0.5,
            label=f"Validation region (n={len(df_r3)})")

    ax.axvline(df_train[overlap_col].mean(), color="blue", linestyle="--",
               label=f"Train mean: {df_train[overlap_col].mean():.1f}%")
    ax.axvline(df_r3[overlap_col].mean(), color="darkorange", linestyle="--",
               label=f"Val mean: {df_r3[overlap_col].mean():.1f}%")

    ax.set_xlabel("Overlap %", fontsize=12)
    ax.set_ylabel("Number of glaciers", fontsize=12)
    ax.set_xlim(left=bins[0])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Vulnerability characteristics
# ---------------------------------------------------------------------------

def plot_vulnerability_characteristics(
    df_vuln: pd.DataFrame,
) -> Figure:
    """
    Compare physical characteristics of high vs low vulnerability glaciers.

    Splits glaciers into top and bottom quartile by mean MLP score, then
    plots histograms of elevation, slope, aspect, edge distance, and area
    for each group. Shows that small size and high edge fraction (not slope)
    are the primary drivers of vulnerability.

    Parameters
    ----------
    df_vuln : pd.DataFrame
        Dataframe with columns: ``mean_mlp_score``, ``mean_elevation``,
        ``mean_slope``, ``mean_aspect``, ``mean_edge_distance``,
        ``area_km2``. Typically produced by merging
        ``evaluate.vulnerability_ranking`` with
        ``evaluate.glacier_physical_features``.

    Returns
    -------
    Figure
    """
    q75 = df_vuln["mean_mlp_score"].quantile(0.75)
    q25 = df_vuln["mean_mlp_score"].quantile(0.25)
    high = df_vuln[df_vuln["mean_mlp_score"] >= q75]
    low = df_vuln[df_vuln["mean_mlp_score"] <= q25]

    features = ["mean_elevation", "mean_slope", "mean_aspect",
                "mean_edge_distance", "area_km2"]
    labels = ["Mean elevation (m)", "Mean slope (°)", "Mean aspect (°)",
              "Mean edge distance (m)", "Area (km²)"]
    units = ["m", "°", "°", "m", "km²"]
    fmts = [".0f", ".1f", ".1f", ".1f", ".2f"]

    bins_list = [
        np.linspace(df_vuln["mean_elevation"].min(),
                    df_vuln["mean_elevation"].max(), 25),
        np.linspace(0, df_vuln["mean_slope"].max(), 25),
        np.linspace(0, 360, 25),
        np.linspace(0, df_vuln["mean_edge_distance"].quantile(0.99), 25),
        np.linspace(0, df_vuln["area_km2"].quantile(0.95), 25),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes_flat = axes.flatten()

    for ax, feat, label, bins, unit, fmt in zip(
        axes_flat[:5], features, labels, bins_list, units, fmts
    ):
        ax.hist(high[feat].clip(upper=bins[-1]), bins=bins, alpha=0.6,
                color="red", edgecolor="black", linewidth=0.4)
        ax.hist(low[feat].clip(upper=bins[-1]), bins=bins, alpha=0.6,
                color="steelblue", edgecolor="black", linewidth=0.4)
        ax.axvline(high[feat].mean(), color="darkred", linestyle="--",
                   linewidth=1.5,
                   label=f"High mean: {high[feat].mean():{fmt}}{unit}")
        ax.axvline(low[feat].mean(), color="darkblue", linestyle="--",
                   linewidth=1.5,
                   label=f"Low mean: {low[feat].mean():{fmt}}{unit}")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Number of glaciers", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(left=bins[0])
        ax.set_ylim(bottom=0)

    axes_flat[5].axis("off")
    legend_elements = [
        mpatches.Patch(facecolor="red", alpha=0.6, edgecolor="black",
                       linewidth=0.4, label="High vulnerability quartile"),
        mpatches.Patch(facecolor="steelblue", alpha=0.6, edgecolor="black",
                       linewidth=0.4, label="Low vulnerability quartile"),
    ]
    axes_flat[5].legend(
        handles=legend_elements, loc="lower left",
        fontsize=11, frameon=True, framealpha=0.8,
        title="Key:", title_fontsize=12,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Glacier panel: observed vs predicted melt
# ---------------------------------------------------------------------------

def plot_observed_vs_predicted_panel(
    df_2017: pd.DataFrame,
    glacier_configs: dict,
    figsize: tuple[int, int] = (10, 14),
) -> Figure:
    """
    6-panel figure of observed vs predicted melt 2017-2025 for selected glaciers.

    Each panel shows true positives (green), false negatives / missed melt
    (amber), false positives (red), and surviving ice (white), overlaid on
    a satellite basemap. Per-glacier calibrated top-N thresholding is applied
    within each RGI polygon.

    Parameters
    ----------
    df_2017 : pd.DataFrame
        2017 pixel dataframe with columns: ``lon``, ``lat``, ``melt_label``,
        ``rgi_id``, ``mlp_score``. Must have MLP scores pre-computed.
    glacier_configs : dict
        Mapping of RGI ID → config dict with keys:
        ``name``, ``zoom``, ``pad``, ``load_pad``,
        ``lon_offset``, ``lat_offset``.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(
        3, 2, figsize=figsize,
        gridspec_kw={"wspace": 0.05, "hspace": 0.15},
    )
    axes_flat = axes.flatten()

    for idx, (rgi_id, config) in enumerate(glacier_configs.items()):
        ax = axes_flat[idx]
        target = df_2017[df_2017["rgi_id"] == rgi_id]
        if len(target) == 0:
            ax.set_visible(False)
            continue

        pad = config["pad"]
        load_pad = config.get("load_pad", pad)
        lon_offset = config.get("lon_offset", 0.0)
        lat_offset = config.get("lat_offset", 0.0)

        (lon_min_d, lat_min_d, lon_max_d, lat_max_d,
         lon_min, lat_min, lon_max, lat_max) = _glacier_bbox(
            target, pad, load_pad, lon_offset, lat_offset
        )

        # All pixels in load bbox
        box = df_2017[
            (df_2017["lat"] >= lat_min) & (df_2017["lat"] <= lat_max) &
            (df_2017["lon"] >= lon_min) & (df_2017["lon"] <= lon_max)
        ].copy().reset_index(drop=True)

        # Per-glacier calibrated top-N prediction
        box["predicted_melt"] = False
        for gid, grp in box.groupby("rgi_id"):
            if pd.isna(gid):
                continue
            n_actual = (grp["melt_label"] == 1).sum()
            if n_actual == 0:
                continue
            top_n = np.argsort(grp["mlp_score"].values)[::-1][:n_actual]
            box.loc[grp.index[top_n], "predicted_melt"] = True

        actual = box["melt_label"] == 1
        predicted = box["predicted_melt"]
        box["plot_colour"] = None
        box.loc[~actual & ~predicted, "plot_colour"] = OBSERVED_PREDICTED_COLOURS["neither"]
        box.loc[actual & predicted,   "plot_colour"] = OBSERVED_PREDICTED_COLOURS["both"]
        box.loc[actual & ~predicted,  "plot_colour"] = OBSERVED_PREDICTED_COLOURS["actual_only"]
        box.loc[~actual & predicted,  "plot_colour"] = OBSERVED_PREDICTED_COLOURS["pred_only"]

        # Set Web Mercator extent
        x_min, x_max, y_min, y_max = _mercator_corners(
            lon_min_d, lat_min_d, lon_max_d, lat_max_d
        )
        x_min, x_max, y_min, y_max = _square_extent(x_min, x_max, y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ctx.add_basemap(
            ax, source=ctx.providers.Esri.WorldTopoMap,
            zoom=config["zoom"], alpha=0.9,
        )
        _plot_as_grid(box, ax, lon_min, lat_min, lon_max, lat_max)
        _add_map_furniture(ax, lon_min_d, lat_min_d, lon_max_d, lat_max_d)

        # Overlap annotation in title
        n_melt = actual.sum()
        if n_melt > 0:
            top_n = np.argsort(box["mlp_score"].values)[::-1][:n_melt]
            pred = np.zeros(len(box), dtype=bool)
            pred[top_n] = True
            overlap = (pred & actual.values).sum() / n_melt * 100
            ax.set_title(f"{overlap:.1f}% Overlap", fontsize=10,
                         pad=3, fontweight="bold")

        # Panel label
        ax.text(0.02, 0.97, f"({chr(97+idx)})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", color="black",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.7))

    # Shared legend
    handles = [
        mpatches.Patch(facecolor=OBSERVED_PREDICTED_COLOURS["both"],
                       label="Predicted & observed melt"),
        mpatches.Patch(facecolor=OBSERVED_PREDICTED_COLOURS["actual_only"],
                       label="Observed melt only (false negative)"),
        mpatches.Patch(facecolor=OBSERVED_PREDICTED_COLOURS["pred_only"],
                       label="Predicted melt only (false positive)"),
        mpatches.Patch(facecolor=OBSERVED_PREDICTED_COLOURS["neither"],
                       edgecolor="grey", label="Surviving ice"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.05))
    plt.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.12)
    return fig


# ---------------------------------------------------------------------------
# Glacier panel: future melt projection
# ---------------------------------------------------------------------------

def plot_future_melt_panel(
    df_melt: pd.DataFrame,
    df_2017: pd.DataFrame,
    glacier_configs: dict,
    timesteps: list[int] = TIMESTEPS,
    figsize: tuple[int, int] = (10, 14),
) -> Figure:
    """
    6-panel figure of projected melt 2025-2065 with observed melt overlay.

    Each panel colours pixels by the decade in which they are projected to
    melt (warm = sooner, cool = later), with observed 2017-2025 melt shown
    in dark navy on top. Pixels surviving past 2065 are not coloured.

    Parameters
    ----------
    df_melt : pd.DataFrame
        Output of ``projection.run_simulation`` for the central scenario.
        Columns: ``lon``, ``lat``, ``rgi_id``, ``region``, ``melt_year``.
    df_2017 : pd.DataFrame
        2017 pixel dataframe with ``lon``, ``lat``, ``melt_label`` columns.
    glacier_configs : dict
        Same format as ``plot_observed_vs_predicted_panel``.
    timesteps : list of int
        Decade end-years to colour. Default [2035, 2045, 2055, 2065].
    figsize : tuple

    Returns
    -------
    Figure
    """
    plot_timesteps = [t for t in timesteps if t in FUTURE_MELT_COLOURS]
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes_flat = axes.flatten()

    for idx, (rgi_id, config) in enumerate(glacier_configs.items()):
        ax = axes_flat[idx]
        target = df_melt[df_melt["rgi_id"] == rgi_id]
        if len(target) == 0:
            ax.set_visible(False)
            continue

        pad = config["pad"]
        load_pad = config.get("load_pad", pad)
        lon_offset = config.get("lon_offset", 0.0)
        lat_offset = config.get("lat_offset", 0.0)

        (lon_min_d, lat_min_d, lon_max_d, lat_max_d,
         lon_min, lat_min, lon_max, lat_max) = _glacier_bbox(
            target, pad, load_pad, lon_offset, lat_offset
        )

        # Simulation pixels in bbox
        sim_box = df_melt[
            (df_melt["lat"] >= lat_min) & (df_melt["lat"] <= lat_max) &
            (df_melt["lon"] >= lon_min) & (df_melt["lon"] <= lon_max)
        ].copy().reset_index(drop=True)

        sim_box["plot_colour"] = None
        for year in plot_timesteps:
            sim_box.loc[sim_box["melt_year"] == year, "plot_colour"] = \
                FUTURE_MELT_COLOURS[year]

        # Observed melt pixels in bbox (overlay in dark navy)
        obs_box = df_2017[
            (df_2017["lat"] >= lat_min) & (df_2017["lat"] <= lat_max) &
            (df_2017["lon"] >= lon_min) & (df_2017["lon"] <= lon_max) &
            (df_2017["melt_label"] == 1)
        ].copy().reset_index(drop=True)
        obs_box["plot_colour"] = OBSERVED_MELT_COLOUR

        combined = pd.concat([sim_box, obs_box], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["lon", "lat"], keep="last"
        )

        x_min, x_max, y_min, y_max = _mercator_corners(
            lon_min_d, lat_min_d, lon_max_d, lat_max_d
        )
        x_min, x_max, y_min, y_max = _square_extent(x_min, x_max, y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ctx.add_basemap(
            ax, source=ctx.providers.Esri.WorldTopoMap,
            zoom=config["zoom"], alpha=0.9,
        )
        _plot_as_grid(combined, ax, lon_min, lat_min, lon_max, lat_max)
        _add_map_furniture(ax, lon_min_d, lat_min_d, lon_max_d, lat_max_d)

        ax.text(0.02, 0.97, f"({chr(97+idx)})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", color="black",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.7))

    # Shared legend
    melt_labels = {
        2035: "2025-2035", 2045: "2035-2045",
        2055: "2045-2055", 2065: "2055-2065",
    }
    handles = [mpatches.Patch(color=OBSERVED_MELT_COLOUR,
                               label="Observed melt 2017-2025")]
    handles += [
        mpatches.Patch(facecolor=FUTURE_MELT_COLOURS[y],
                       label=melt_labels[y])
        for y in plot_timesteps
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.02))
    plt.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.08)
    return fig


# ---------------------------------------------------------------------------
# Retreat projection plot
# ---------------------------------------------------------------------------

def plot_retreat_projection(
    start_area_km2: float,
    scenarios: dict[str, float] = SCENARIOS,
    start_year: int = 2025,
) -> Figure:
    """
    Plot projected ice area from 2025 to deglaciation under each scenario.

    Draws straight lines from the 2025 starting area to zero for each
    scenario (reflecting the linear retreat assumption), with a shaded
    uncertainty envelope between the lower and upper bound scenarios.

    Parameters
    ----------
    start_area_km2 : float
        Ice area in km² at the start of the simulation (2025).
    scenarios : dict of str → float
        Retreat rate scenarios (km²/year). Default uses the three scenarios
        from the paper.
    start_year : int
        Start year for the projection. Default 2025.

    Returns
    -------
    Figure
    """
    from glacier_melt.projection import deglaciation_year_linear

    deglac_years = {
        name: deglaciation_year_linear(start_year, start_area_km2, rate)
        for name, rate in scenarios.items()
    }

    scenario_styles = {
        "lower":   {"color": "#2166ac", "linestyle": "--"},
        "central": {"color": "#1a1a1a", "linestyle": "-"},
        "upper":   {"color": "#d6604d", "linestyle": ":"},
    }
    scenario_labels = {
        "lower":   f"Lower bound ({scenarios['lower']} km²/yr)",
        "central": f"Central estimate ({scenarios['central']} km²/yr)",
        "upper":   f"Upper bound ({scenarios['upper']} km²/yr)",
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    # Uncertainty envelope between lower and upper bounds
    x_fill = np.linspace(start_year, deglac_years["lower"], 500)
    y_lo = np.clip(
        np.interp(x_fill,
                  [start_year, deglac_years["upper"]],
                  [start_area_km2, 0]),
        0, None,
    )
    y_hi = np.clip(
        np.interp(x_fill,
                  [start_year, deglac_years["lower"]],
                  [start_area_km2, 0]),
        0, None,
    )
    ax.fill_between(x_fill, y_lo, y_hi, alpha=0.15, color="steelblue")

    for name, style in scenario_styles.items():
        if name not in deglac_years:
            continue
        deglac = deglac_years[name]
        label = f"{scenario_labels[name]} (~{deglac})"
        ax.plot([start_year, deglac], [start_area_km2, 0],
                color=style["color"], linestyle=style["linestyle"],
                linewidth=2.5, label=label, zorder=3)

    ax.axhline(start_area_km2, color="red", linestyle="--",
               alpha=0.5, linewidth=1,
               label=f"{start_year} ice area ({start_area_km2:.0f} km²)")

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Remaining ice area (km²)", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.set_xlim(start_year - 3, max(deglac_years.values()) + 3)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(
        range(start_year, max(deglac_years.values()) + 5, 5)
    )
    plt.tight_layout()
    return fig
