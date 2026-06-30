"""
gee/export_simulation.py
========================
Export 2025 ice pixel data (AlphaEarth embeddings + EASD features) from
Google Earth Engine to Google Drive as GeoTIFF tiles.

These pixels form the starting point for the future melt simulation. The
2025 ice extent (from the filtered classification) is used as the pixel
mask, and 2025 AlphaEarth embeddings are used as features — capturing the
current state of each glacier rather than the 2017 baseline used for
model training.

The exported stack contains:

    elevation       : SRTM elevation (m)
    aspect          : terrain aspect (degrees)
    slope           : terrain slope (degrees)
    edge_distance   : cumulative cost distance to 2025 glacier edge (m)
    A00–A63         : 64-dimensional AlphaEarth 2025 embeddings

Note: no melt label is included, since these pixels are inputs to the
forward simulation, not training data.

The resulting GeoTIFF tiles are downloaded from Drive and converted to
per-region parquet files by ``scripts/convert_2025_tiffs.py``.

GEE assets used
---------------
Filtered classifications: projects/glacier-melt/assets/classifications_filtered/
                          classified_{region}_2025_filtered
AlphaEarth embeddings:   GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (2025)
SRTM:                    USGS/SRTMGL1_003

Usage
-----
Run this script in a Colab or GEE-authenticated Python environment.
Monitor submitted tasks at https://code.earthengine.google.com/tasks
"""

import ee
ee.Authenticate()
ee.Initialize(project='glacier-melt')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_YEAR = 2025
REGIONS = ['R3', 'R1b', 'R1a', 'R2']   # smallest first for testing
SCALE = 10
DRIVE_FOLDER = 'glacier_melt_sim_export'
FILTERED_ASSET_ROOT = 'projects/glacier-melt/assets/classifications_filtered'
RGI_ASSET = 'projects/glacier-melt/assets/rgi70_low_latitudes'


# ---------------------------------------------------------------------------
# Region geometry construction
# ---------------------------------------------------------------------------

def build_region_geometries(regions=REGIONS):
    """
    Build the analysis geometry for each region.

    Uses the 2025 filtered classification extent as the spatial boundary
    (rather than 2017 as in export_training.py), since we want the 2025
    ice extent as the simulation starting point.

    Parameters
    ----------
    regions : list of str

    Returns
    -------
    dict of str → ee.Geometry
    """
    rgi = ee.FeatureCollection(RGI_ASSET)
    geometries = {}
    for region in regions:
        full_geom = ee.Image(
            f'{FILTERED_ASSET_ROOT}/classified_{region}_2017_filtered'
        ).geometry()
        rgi_clipped = rgi.filterBounds(full_geom)
        buffered = rgi_clipped.geometry().buffer(500).intersection(full_geom)
        geometries[region] = buffered
    print(f"Region geometries built for: {list(geometries.keys())}")
    return geometries


# ---------------------------------------------------------------------------
# Feature stack construction
# ---------------------------------------------------------------------------

def build_simulation_stack(region, region_geom):
    """
    Build the per-pixel feature stack for simulation export.

    Pixel mask: 2025 ice pixels only (classified as ice in the filtered
    2025 classification). The edge_distance is computed relative to the
    2025 ice extent, not the 2017 baseline.

    AlphaEarth embeddings are from SIM_YEAR (2025) so that the MLP model
    scores reflect each pixel's current surface state rather than its 2017
    baseline, providing the most up-to-date vulnerability ranking for the
    forward simulation.

    Parameters
    ----------
    region : str
        Region identifier.
    region_geom : ee.Geometry
        Region analysis geometry.

    Returns
    -------
    ee.Image
        Feature stack masked to 2025 ice pixels, bands:
        elevation, aspect, slope, edge_distance, A00–A63.
    """
    ice_2025 = (ee.Image(
        f'{FILTERED_ASSET_ROOT}/classified_{region}_{SIM_YEAR}_filtered'
    ).select('classification'))
    is_ice_2025 = ice_2025.eq(1)

    # Edge distance relative to 2025 ice extent
    edge_distance = (ee.Image(1).toFloat()
                     .cumulativeCost(source=ice_2025.eq(0), maxDistance=2000)
                     .rename('edge_distance'))

    # SRTM topographic features
    srtm = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Terrain.products(srtm)
    elevation = srtm.rename('elevation')
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')

    # AlphaEarth 2025 embeddings
    ae = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
          .filter(ee.Filter.calendarRange(SIM_YEAR, SIM_YEAR, 'year'))
          .filterBounds(region_geom)
          .mosaic())

    # Stack and mask to 2025 ice pixels
    stack = (elevation
             .addBands(aspect)
             .addBands(slope)
             .addBands(edge_distance)
             .addBands(ae)
             .updateMask(is_ice_2025)
             .clip(region_geom)
             .toFloat())

    return stack


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def submit_simulation_export(region, region_geom):
    """
    Submit a GEE batch task to export the simulation feature stack.

    Parameters
    ----------
    region : str
    region_geom : ee.Geometry

    Returns
    -------
    ee.batch.Task
    """
    stack = build_simulation_stack(region, region_geom)

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=f'sim_pixels_{region}_{SIM_YEAR}_tiff',
        folder=DRIVE_FOLDER,
        fileNamePrefix=f'sim_pixels_{region}_{SIM_YEAR}',
        region=region_geom,
        scale=SCALE,
        crs='EPSG:4326',
        maxPixels=1e10,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True, 'noData': -9999},
    )
    task.start()
    print(f"Submitted: {region} — task {task.id} (state: {task.status()['state']})")
    return task


def submit_all_simulation_exports(region_geometries=None):
    """
    Submit simulation export tasks for all regions.

    Parameters
    ----------
    region_geometries : dict or None
        Output of ``build_region_geometries``. If None, builds automatically.

    Returns
    -------
    dict of str → ee.batch.Task
    """
    if region_geometries is None:
        region_geometries = build_region_geometries()

    tasks = {}
    for region, geom in region_geometries.items():
        try:
            tasks[region] = submit_simulation_export(region, geom)
        except Exception as e:
            print(f"{region}: ERROR — {e}")

    return tasks


def check_tasks():
    """Print status of the 10 most recent GEE batch tasks."""
    tasks = ee.batch.Task.list()[:10]
    for t in tasks:
        s = t.status()
        print(f"{s.get('description', 'unknown'):40s} "
              f"{s.get('state', 'unknown'):12s} "
              f"{s.get('error_message', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    region_geometries = build_region_geometries()

    # Verify stack band names before submitting (use R3 as test — smallest)
    test_stack = build_simulation_stack('R3', region_geometries['R3'])
    print(f"\nStack band names: {test_stack.bandNames().getInfo()}")

    # Submit exports for all regions
    tasks = submit_all_simulation_exports(region_geometries)
    print(f"\n{len(tasks)} simulation export tasks submitted.")
    print("Monitor at: https://code.earthengine.google.com/tasks")
