"""
glacier_melt.projection
=======================
Future melt simulation under a linear retreat assumption.

The simulation applies a fixed number of pixels per 10-year timestep,
selected globally by descending MLP vulnerability score (predicted melt
probability on 2025 AE embeddings). The pixel count per step is derived
from the retreat rate scenario divided by pixel area.

Retreat rate scenarios
----------------------
Three scenarios bracket the plausible range of future retreat:

    Lower bound   : 29.6 km²/yr
    Central        : 32.0 km²/yr  (this study — linear regression R²=0.961)
    Upper bound    : 34.4 km²/yr

The central estimate is derived from a linear regression of total glaciated
area against year over 2017–2025 (excluding 2024 due to a GEE export range
error; sensitivity analysis confirms minimal impact on slope given R²=0.961).

Under the central scenario, complete deglaciation is projected by 2060 ± 3
years, with uncertainty propagated from the scenario bounds.

Simulation mechanics
--------------------
At each 10-year timestep, a fixed number of surviving pixels are designated
as melted, chosen by descending MLP score (highest vulnerability first).
The selection is global across all glaciers simultaneously, not per-glacier.
Pixels surviving to 2085 receive melt_year = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

from glacier_melt.sampling import AE_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIXEL_AREA_KM2 = (10 * 10) / 1e6    # 10m × 10m = 0.0001 km²
TIMESTEP = 10                         # years per simulation step
START_YEAR = 2025
END_YEAR = 2085
TIMESTEPS = list(range(START_YEAR, END_YEAR + 1, TIMESTEP))
MIN_GLACIER_KM2 = 0.01               # minimum glacier size to include

# Retreat rate scenarios (km²/year)
SCENARIOS = {
    "lower":   29.6,   # lower bound
    "central": 32.0,   # this study
    "upper":   34.4,   # upper bound
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pixels_per_step(retreat_rate_km2_yr: float) -> int:
    """
    Convert an annual retreat rate to a pixel count per 10-year timestep.

    Parameters
    ----------
    retreat_rate_km2_yr : float
        Annual ice area loss in km²/year.

    Returns
    -------
    int
        Number of pixels to melt per 10-year step.
    """
    return int(retreat_rate_km2_yr / PIXEL_AREA_KM2) * TIMESTEP


def deglaciation_year_linear(
    start_year: int,
    start_area_km2: float,
    retreat_rate_km2_yr: float,
) -> int:
    """
    Compute the projected deglaciation year under a linear retreat assumption.

    Used both directly (for the projection plot) and as a fallback when
    the simulation does not reach zero area within the simulation window.

    Parameters
    ----------
    start_year : int
        Year of the starting ice extent (2025).
    start_area_km2 : float
        Ice area in km² at ``start_year``.
    retreat_rate_km2_yr : float
        Annual ice area loss in km²/year.

    Returns
    -------
    int
        Projected year of complete deglaciation.
    """
    return int(start_year + start_area_km2 / retreat_rate_km2_yr)


# ---------------------------------------------------------------------------
# RGI spatial join and size filtering
# ---------------------------------------------------------------------------

def prepare_simulation_pixels(
    df_sim: pd.DataFrame,
    rgi_path: str | Path,
    min_glacier_km2: float = MIN_GLACIER_KM2,
) -> pd.DataFrame:
    """
    Assign RGI glacier IDs to 2025 simulation pixels and apply size filter.

    Pixels not falling within any RGI polygon are excluded. Glaciers smaller
    than ``min_glacier_km2`` are also excluded.

    Parameters
    ----------
    df_sim : pd.DataFrame
        2025 ice pixel dataframe from ``sampling.load_simulation_parquets``.
        Must have ``lon`` and ``lat`` columns.
    rgi_path : str or Path
        Path to the RGI 7.0 GeoJSON file for Peru.
    min_glacier_km2 : float
        Minimum glacier area to include. Default 0.01 km².

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with ``rgi_id`` column added. Contains only
        RGI-matched pixels from glaciers exceeding the size threshold.
    """
    rgi = gpd.read_file(str(rgi_path))[["rgi_id", "geometry"]]
    print(f"RGI glaciers loaded: {len(rgi):,}")

    geometry = gpd.points_from_xy(df_sim["lon"], df_sim["lat"])
    gdf = gpd.GeoDataFrame(
        df_sim[["lon", "lat"]], geometry=geometry, crs="EPSG:4326"
    )
    joined = gpd.sjoin(gdf, rgi, how="left", predicate="within")
    df_sim = df_sim.copy()
    df_sim["rgi_id"] = joined["rgi_id"].values

    df_matched = df_sim[df_sim["rgi_id"].notna()].reset_index(drop=True)

    glacier_sizes = df_matched.groupby("rgi_id").size() * PIXEL_AREA_KM2
    valid = glacier_sizes[glacier_sizes >= min_glacier_km2].index
    df_matched = df_matched[df_matched["rgi_id"].isin(valid)].reset_index(drop=True)

    print(f"After RGI match and size filter (>= {min_glacier_km2} km²):")
    print(f"  Valid glaciers: {df_matched['rgi_id'].nunique():,}")
    print(f"  Valid pixels:   {len(df_matched):,}")
    print(f"  Ice area:       {len(df_matched) * PIXEL_AREA_KM2:.2f} km²")
    return df_matched


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(
    df_matched: pd.DataFrame,
    scores: np.ndarray,
    timesteps: list[int] = TIMESTEPS,
    n_pixels_per_step: int | None = None,
    retreat_rate_km2_yr: float = SCENARIOS["central"],
) -> tuple[list[tuple[int, float]], pd.DataFrame]:
    """
    Iterative glacier melt simulation using global MLP vulnerability ranking.

    At each 10-year timestep, a fixed number of surviving pixels are
    designated as melted, selected by descending MLP vulnerability score
    (highest vulnerability first). The selection is global across all
    glaciers simultaneously — the model determines *which* pixels melt,
    the retreat rate scenario determines *how many* melt per step.

    Parameters
    ----------
    df_matched : pd.DataFrame
        2025 ice pixel dataframe with ``lon``, ``lat``, ``rgi_id``,
        ``region`` columns. From ``prepare_simulation_pixels``.
    scores : np.ndarray, shape (N,)
        MLP predicted melt probabilities for each pixel, computed on
        2025 AlphaEarth embeddings.
    timesteps : list of int
        Sequence of years to simulate. Default [2025, 2035, ..., 2085].
    n_pixels_per_step : int or None
        Number of pixels to melt per timestep. If None, computed from
        ``retreat_rate_km2_yr``.
    retreat_rate_km2_yr : float
        Annual ice area loss in km²/year. Used only when
        ``n_pixels_per_step`` is None. Default is the central estimate
        (32.0 km²/yr).

    Returns
    -------
    area_history : list of (year, ice_area_km2) tuples
        Ice area remaining at the end of each timestep.
    df_out : pd.DataFrame
        One row per pixel with a ``melt_year`` column giving the timestep
        end-year in which that pixel melted. Pixels surviving to the final
        timestep have ``melt_year = 0``.
    """
    if n_pixels_per_step is None:
        n_pixels_per_step = pixels_per_step(retreat_rate_km2_yr)

    surviving = np.ones(len(df_matched), dtype=bool)
    melt_year = np.zeros(len(df_matched), dtype=np.int32)
    area_history = [(timesteps[0], surviving.sum() * PIXEL_AREA_KM2)]

    for t_idx in range(len(timesteps) - 1):
        year = timesteps[t_idx]
        next_year = timesteps[t_idx + 1]
        n_current = surviving.sum()
        n_melt = min(n_current, n_pixels_per_step)

        # Select top-N surviving pixels by vulnerability score
        surv_idx = np.where(surviving)[0]
        surv_scores = scores[surv_idx]
        top_local = np.argsort(surv_scores)[::-1][:n_melt]
        top_global = surv_idx[top_local]

        surviving[top_global] = False
        melt_year[top_global] = next_year

        ice_area = surviving.sum() * PIXEL_AREA_KM2
        area_history.append((next_year, ice_area))

        pct_lost = (1 - ice_area / (len(df_matched) * PIXEL_AREA_KM2)) * 100
        print(f"  {year}→{next_year}: {surviving.sum():,} px, "
              f"{ice_area:.1f} km² ({pct_lost:.1f}% lost)")

    df_out = df_matched[["lon", "lat", "rgi_id", "region"]].copy()
    df_out["melt_year"] = melt_year
    return area_history, df_out


def run_all_scenarios(
    df_matched: pd.DataFrame,
    scores: np.ndarray,
    scenarios: dict[str, float] = SCENARIOS,
    timesteps: list[int] = TIMESTEPS,
) -> dict[str, tuple[list[tuple[int, float]], pd.DataFrame]]:
    """
    Run the simulation under all retreat rate scenarios.

    Parameters
    ----------
    df_matched : pd.DataFrame
        2025 ice pixel dataframe from ``prepare_simulation_pixels``.
    scores : np.ndarray
        MLP predicted melt probabilities.
    scenarios : dict of str → float
        Mapping of scenario name → retreat rate (km²/year).
    timesteps : list of int
        Years to simulate.

    Returns
    -------
    dict of scenario_name → (area_history, df_melt_years)
    """
    results = {}
    for name, rate in scenarios.items():
        print(f"\n=== Scenario: {name} ({rate} km²/yr) ===")
        area_history, df_out = run_simulation(
            df_matched, scores, timesteps,
            retreat_rate_km2_yr=rate,
        )
        results[name] = (area_history, df_out)
    return results


# ---------------------------------------------------------------------------
# Deglaciation year
# ---------------------------------------------------------------------------

def find_deglaciation_year(
    area_history: list[tuple[int, float]],
    retreat_rate_km2_yr: float,
) -> int:
    """
    Find or extrapolate the year of complete deglaciation.

    If the simulation reaches zero area within the simulation window,
    returns that year directly. Otherwise, extrapolates linearly from
    the final simulation year using the scenario retreat rate.

    Parameters
    ----------
    area_history : list of (year, area_km2) tuples
        Output of ``run_simulation``.
    retreat_rate_km2_yr : float
        Annual retreat rate for the scenario (used for extrapolation).

    Returns
    -------
    int
        Projected year of complete deglaciation.
    """
    for year, area in area_history:
        if area <= 0:
            return year

    last_year, last_area = area_history[-1]
    annual_loss = retreat_rate_km2_yr
    return int(last_year + last_area / annual_loss)


# ---------------------------------------------------------------------------
# Area history export
# ---------------------------------------------------------------------------

def area_history_to_dataframe(
    scenario_results: dict[str, tuple[list[tuple[int, float]], pd.DataFrame]],
    start_area_km2: float,
) -> pd.DataFrame:
    """
    Convert per-scenario area histories to a tidy DataFrame for export.

    Parameters
    ----------
    scenario_results : dict
        Output of ``run_all_scenarios``.
    start_area_km2 : float
        Ice area at the start of the simulation (2025), used to compute
        percentage loss.

    Returns
    -------
    pd.DataFrame
        Columns: ``scenario``, ``year``, ``ice_area_km2``,
        ``ice_loss_km2``, ``ice_loss_pct``.
    """
    rows = []
    for name, (area_history, _) in scenario_results.items():
        for year, area in area_history:
            rows.append({
                "scenario":     name,
                "year":         year,
                "ice_area_km2": area,
                "ice_loss_km2": start_area_km2 - area,
                "ice_loss_pct": (start_area_km2 - area) / start_area_km2 * 100,
            })
    return pd.DataFrame(rows)
