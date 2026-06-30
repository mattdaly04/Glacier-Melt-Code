"""
gee/export_training.py
======================
Export 2017 training data (AlphaEarth embeddings + EASD features + melt
labels) from Google Earth Engine to Google Drive as GeoTIFF tiles.

For each region, a per-pixel feature stack is constructed over 2017 ice
pixels (pixels classified as ice in the 2017 filtered classification) and
exported as a GeoTIFF. The stack contains:

    melt_label      : 1 if ice in 2017 but not in 2025, else 0
    elevation       : SRTM elevation (m)
    aspect          : terrain aspect (degrees)
    slope           : terrain slope (degrees)
    edge_distance   : cumulative cost distance to glacier edge (m)
    A00–A63         : 64-dimensional AlphaEarth 2017 embeddings

The resulting GeoTIFF tiles are downloaded from Drive and converted to
per-region parquet files by ``scripts/convert_2017_tiffs.py``.

GEE assets used
---------------
Filtered classifications: projects/glacier-melt/assets/classifications_filtered/
                          classified_{region}_{year}_filtered
AlphaEarth embeddings:   GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
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

BASELINE_YEAR = 2017    # year used for ice mask and AlphaEarth embeddings
ENDPOINT_YEAR = 2025    # year used to define melt_label (ice in 2017, not 2025)
REGIONS = ['R3', 'R1b', 'R1a', 'R2']   # smallest first for testing
SCALE = 10              # native Sentinel-2 / AlphaEarth resolution (m)
DRIVE_FOLDER = 'glacier_melt_exports_v2'
FILTERED_ASSET_ROOT = 'projects/glacier-melt/assets/classifications_filtered'
RGI_ASSET = 'projects/glacier-melt/assets/rgi70_low_latitudes'


# ---------------------------------------------------------------------------
# Region geometry construction
# ---------------------------------------------------------------------------

def build_region_geometries(regions=REGIONS):
    """
    Build the analysis geometry for each region.

    Identical to ``classify.build_region_geometries`` but using the
    BASELINE_YEAR classification to define the spatial extent (2017 ice
    extent, not 2023).

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
            f'{FILTERED_ASSET_ROOT}/classified_{region}_{BASELINE_YEAR}_filtered'
        ).geometry()
        rgi_clipped = rgi.filterBounds(full_geom)
        buffered = rgi_clipped.geometry().buffer(500).intersection(full_geom)
        geometries[region] = buffered
    print(f"Region geometries built for: {list(geometries.keys())}")
    return geometries


# ---------------------------------------------------------------------------
# Feature stack construction
# ---------------------------------------------------------------------------

def build_training_stack(region, region_geom):
    """
    Build the per-pixel feature stack for training data export.

    Pixel mask: 2017 ice pixels only (classified as ice in filtered
    2017 classification). Non-ice pixels and pixels outside the 500m-
    buffered RGI geometry are excluded.

    Melt label definition: a pixel is labelled melt=1 if it was ice in
    BASELINE_YEAR (2017) but non-ice in ENDPOINT_YEAR (2025), i.e. it
    melted at some point in the 2017–2025 observation window.

    Edge distance: computed as cumulative cost distance from each ice pixel
    to the nearest non-ice pixel, using a uniform cost surface. Captures
    proximity to the glacier edge, which is a key predictor of melt
    vulnerability (Andriychenko 2024).

    AlphaEarth embeddings: 64-dimensional per-pixel embeddings from the
    BASELINE_YEAR (2017) annual composite, providing implicit surface
    reflectance, temporal dynamics, and topographic context without
    hand-engineering additional features.

    Parameters
    ----------
    region : str
        Region identifier.
    region_geom : ee.Geometry
        Region analysis geometry.

    Returns
    -------
    ee.Image
        Feature stack masked to 2017 ice pixels, bands:
        melt_label, elevation, aspect, slope, edge_distance, A00–A63.
    """
    base_path = FILTERED_ASSET_ROOT

    # Ice masks
    ice_t0 = (ee.Image(f'{base_path}/classified_{region}_{BASELINE_YEAR}_filtered')
              .select('classification').rename('ice_t0'))
    ice_t1 = (ee.Image(f'{base_path}/classified_{region}_{ENDPOINT_YEAR}_filtered')
              .select('classification').rename('ice_t1'))
    is_ice_t0 = ice_t0.eq(1)

    # Melt label: ice in 2017, non-ice in 2025
    melt_label = ice_t0.eq(1).And(ice_t1.eq(0)).rename('melt_label')

    # Edge distance: cumulative cost to nearest non-ice pixel
    edge_distance = (ee.Image(1).toFloat()
                     .cumulativeCost(source=ice_t0.eq(0), maxDistance=2000)
                     .rename('edge_distance'))

    # SRTM topographic features
    srtm = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Terrain.products(srtm)
    elevation = srtm.rename('elevation')
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')

    # AlphaEarth 2017 embeddings (64-dimensional per pixel)
    ae = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
          .filterDate(f'{BASELINE_YEAR}-01-01', f'{BASELINE_YEAR}-12-31')
          .filterBounds(region_geom)
          .mosaic())

    # Stack and mask to 2017 ice pixels
    stack = (melt_label
             .addBands(elevation)
             .addBands(aspect)
             .addBands(slope)
             .addBands(edge_distance)
             .addBands(ae)
             .updateMask(is_ice_t0)
             .clip(region_geom)
             .toFloat())

    return stack


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def submit_training_export(region, region_geom):
    """
    Submit a GEE batch task to export the training feature stack for one region.

    Exports to Google Drive as Cloud-Optimised GeoTIFF with nodata=-9999.
    Large regions (R1a, R2) may produce many tiles.

    Parameters
    ----------
    region : str
        Region identifier.
    region_geom : ee.Geometry

    Returns
    -------
    ee.batch.Task
    """
    stack = build_training_stack(region, region_geom)

    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=f'training_pixels_{region}_{BASELINE_YEAR}',
        folder=DRIVE_FOLDER,
        fileNamePrefix=f'all_pixels_{region}_v2',
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


def submit_all_training_exports(region_geometries=None):
    """
    Submit training data export tasks for all regions.

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
            tasks[region] = submit_training_export(region, geom)
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

    # Verify stack band names before submitting (use R3 as test — smallest region)
    test_stack = build_training_stack('R3', region_geometries['R3'])
    print(f"\nStack band names: {test_stack.bandNames().getInfo()}")

    # Submit exports for all regions
    tasks = submit_all_training_exports(region_geometries)
    print(f"\n{len(tasks)} export tasks submitted.")
    print("Monitor at: https://code.earthengine.google.com/tasks")
