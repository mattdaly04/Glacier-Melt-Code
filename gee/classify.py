"""
gee/classify.py
===============
Sentinel-2 glacier coverage classification for Peru, 2016–2025.

Extends Elliott (2024)'s annual glacier coverage dataset from 2023 to 2025
by following the same methodology:

1. Build annual Sentinel-2 mosaics representative of maximum ablation
   (minimum NDSI per pixel across the dry season, May–October)
2. Train a Random Forest classifier per region using Elliott's 2023 ice
   pixels as positive labels and 2016 non-ice pixels as negative labels
3. Apply five post-classification filters to correct temporal and spatial
   artefacts in the raw classifications

Key methodological difference from Elliott
------------------------------------------
Elliott trained one RF classifier per region per year (32 total), using
manually delineated training pixels. This study trains one classifier per
region across all years, using Elliott's existing classifications as labels.
This reduces the risk of overfitting to year-specific cloud or illumination
conditions. The spectral signature of ice (high NDSI, negative NDWI) does
not change year to year — only the spatial extent changes.

GEE assets used
---------------
Sentinel-2:    COPERNICUS/S2_HARMONIZED (Level 1C TOA, 10m)
CloudScore+:   GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED
SRTM:          USGS/SRTMGL1_003 (30m)
RGI:           projects/glacier-melt/assets/rgi70_low_latitudes
Elliott 2016:  projects/ee-cavep-peruproject/assets/Glaciers_data/
               Sentinel_processed/{region}/2016_final
Elliott 2023:  projects/ee-cavep-peruproject/assets/Glaciers_data/
               Sentinel_processed/{region}/2023_final

Usage
-----
Run cells in order. Each section submits GEE batch tasks that run
asynchronously. Monitor progress at https://code.earthengine.google.com/tasks
"""

import ee
ee.Authenticate()
ee.Initialize(project='glacier-melt')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YEARS = list(range(2016, 2026))         # 2016–2025 inclusive
REGIONS = ['R1a', 'R1b', 'R2', 'R3']
SCALE = 10                               # native Sentinel-2 resolution (m)
CS_THRESHOLD = 0.7                       # CloudScore+ clearness threshold
RF_N_TREES = 100                         # Random Forest trees per classifier
RF_N_SAMPLES = 2000                      # training samples per class per region

MOSAIC_ASSET_ROOT = 'projects/glacier-melt/assets/mosaics'
CLASSIFICATION_ASSET_ROOT = 'projects/glacier-melt/assets/classifications'
FILTERED_ASSET_ROOT = 'projects/glacier-melt/assets/classifications_filtered'
ELLIOTT_ASSET_ROOT = 'projects/ee-cavep-peruproject/assets/Glaciers_data/Sentinel_processed'
RGI_ASSET = 'projects/glacier-melt/assets/rgi70_low_latitudes'


# ---------------------------------------------------------------------------
# Region geometry construction
# ---------------------------------------------------------------------------

def build_region_geometries(regions=REGIONS):
    """
    Build the analysis geometry for each region.

    Each region's geometry is the union of 500m-buffered RGI glacier polygons
    intersected with the extent of Elliott's 2023 classification. The 500m
    buffer ensures edge pixels are included in the feature extraction.

    Parameters
    ----------
    regions : list of str
        Region identifiers.

    Returns
    -------
    dict of str → ee.Geometry
    """
    rgi = ee.FeatureCollection(RGI_ASSET)
    geometries = {}

    for region in regions:
        full_geom = ee.Image(
            f'{ELLIOTT_ASSET_ROOT}/{region}/2023_final'
        ).geometry()
        rgi_clipped = rgi.filterBounds(full_geom)
        buffered = rgi_clipped.geometry().buffer(500).intersection(full_geom)
        geometries[region] = buffered

    print(f"Region geometries built for: {list(geometries.keys())}")
    return geometries


# ---------------------------------------------------------------------------
# Annual mosaic construction
# ---------------------------------------------------------------------------

def _apply_cloudscoreplus_mask(image):
    """
    Mask cloud and cloud shadow pixels using CloudScore+.

    CloudScore+ provides a per-pixel clearness score (0=cloudy, 1=clear).
    Threshold 0.7 was chosen by Elliott (2024) to balance cloud removal
    against retention of bright ice pixels, which can be confused with
    clouds by aggressive masking algorithms.
    """
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    cs_image = cs_plus.filter(
        ee.Filter.equals('system:index', image.get('system:index'))
    ).first()
    return image.updateMask(ee.Image(cs_image).select('cs').gte(CS_THRESHOLD))


def _add_spectral_indices(image):
    """
    Add NDSI and NDWI bands to a Sentinel-2 image.

    NDSI = (B3 - B11) / (B3 + B11)
        Normalised Difference Snow Index. High values (> ~0.4) indicate
        snow or ice. Glacial lakes also have high NDSI, so NDWI is needed
        as a discriminator.

    NDWI = (B3 - B8) / (B3 + B8)
        Normalised Difference Water Index. Positive values indicate open
        water (lakes); ice has negative NDWI. Separates glacial lakes from
        glacial ice during classification.
    """
    B3  = image.select('B3').toFloat()    # Green, 10m
    B8  = image.select('B8').toFloat()    # NIR, 10m
    B11 = image.select('B11').toFloat()   # SWIR1, 20m (resampled by GEE)

    ndsi = B3.subtract(B11).divide(B3.add(B11)).rename('NDSI')
    ndwi = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI')
    return image.addBands([ndsi, ndwi])


def create_annual_mosaic(year, aoi):
    """
    Create an annual Sentinel-2 mosaic representative of maximum ablation.

    Follows Elliott (2024) Section 3.1.1:
    1. Filter to dry season (May–October) to capture maximum melt exposure
    2. Apply CloudScore+ cloud masking (clearness >= 0.7)
    3. Compute per-pixel NDSI minimum — the least snow-covered observation
       at each pixel across the year
    4. Compute per-pixel median of remaining bands using only observations
       in the lowest NDSI quartile — stable representative reflectance

    Why Level 1C (TOA) not Level 2A (surface reflectance)?
    Elliott used L1C because L2A was not consistently available for all
    years in 2016–2023. Since the RF classifier is trained and applied on
    the same data type, absolute reflectance calibration is not required.

    Parameters
    ----------
    year : int
        Year to create mosaic for.
    aoi : ee.Geometry
        Area of interest.

    Returns
    -------
    ee.Image
        Multi-band mosaic with bands: B2, B3, B4, B8, B11, B12, NDWI, NDSI_min.
    """
    s2 = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
          .filterBounds(aoi)
          .filterDate(f'{year}-05-01', f'{year}-10-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
          .map(_apply_cloudscoreplus_mask)
          .map(_add_spectral_indices))

    # Per-pixel NDSI minimum (most ice-exposed observation)
    ndsi_min = s2.select('NDSI').reduce(ee.Reducer.min()).rename('NDSI_min')

    # Median of low-NDSI observations (most stable ice reflectance)
    ndsi_p25 = s2.select('NDSI').reduce(ee.Reducer.percentile([25])).rename('NDSI_p25')

    def mask_to_low_ndsi(image):
        return image.updateMask(image.select('NDSI').lte(ndsi_p25))

    bands_for_median = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDWI']
    median_mosaic = s2.map(mask_to_low_ndsi).select(bands_for_median).median()

    return median_mosaic.addBands(ndsi_min).clip(aoi)


def submit_mosaic_exports(region_geometries, years=YEARS):
    """
    Submit GEE batch tasks to export annual mosaics as GEE assets.

    Exports each region×year mosaic to ``MOSAIC_ASSET_ROOT``. The
    ``unmask(-9999)`` call fills non-glacier pixels with a nodata value,
    enabling ``skipEmptyTiles`` to reduce export size significantly.

    Parameters
    ----------
    region_geometries : dict
        Output of ``build_region_geometries``.
    years : list of int
        Years to export.

    Returns
    -------
    dict of region → dict of year → ee.batch.Task
    """
    tasks = {r: {} for r in region_geometries}
    for region, geom in region_geometries.items():
        for year in years:
            mosaic = create_annual_mosaic(year, geom).unmask(-9999)
            asset_id = f'{MOSAIC_ASSET_ROOT}/mosaic_{region}_{year}'
            task = ee.batch.Export.image.toAsset(
                image=mosaic,
                description=f'mosaic_{region}_{year}',
                assetId=asset_id,
                region=geom,
                scale=SCALE,
                maxPixels=1e13,
                pyramidingPolicy={'.default': 'mean'},
            )
            task.start()
            tasks[region][year] = task
            print(f"Submitted: mosaic_{region}_{year} — task {task.id}")

    total = sum(len(v) for v in tasks.values())
    print(f"\n{total} mosaic export tasks submitted.")
    return tasks


# ---------------------------------------------------------------------------
# RF classifier training and classification
# ---------------------------------------------------------------------------

def _get_training_data(region, region_geom, ref_mosaic, n_samples=RF_N_SAMPLES):
    """
    Extract RF training samples using Elliott's classifications as labels.

    Ice labels:     Elliott's 2023 classification (value = 1)
    Non-ice labels: Elliott's 2016 classification (value = 0)

    Using Elliott's 2023 ice pixels as positive labels and 2016 non-ice
    pixels as negative labels is justified because:
    - 2023 ice pixels were ice in all earlier years (glaciers only retreat)
    - 2016 non-ice pixels were non-ice in all later years
    - The spectral features of ice/non-ice are stable across years

    Parameters
    ----------
    region : str
        Region identifier.
    region_geom : ee.Geometry
        Region analysis geometry.
    ref_mosaic : ee.Image
        Reference mosaic to extract spectral features from. A mid-period
        year (2020) is used as the reference for stable feature values.
    n_samples : int
        Number of stratified samples per class.

    Returns
    -------
    ee.FeatureCollection
        Training samples with ``label`` property (0 or 1) and spectral
        features (NDSI_min, NDWI, B8).
    """
    elliott_2023 = ee.Image(f'{ELLIOTT_ASSET_ROOT}/{region}/2023_final')
    elliott_2016 = ee.Image(f'{ELLIOTT_ASSET_ROOT}/{region}/2016_final')
    features = ref_mosaic.select(['NDSI_min', 'NDWI', 'B8'])
    simple_region = region_geom.bounds()

    ice_samples = (features
                   .addBands(elliott_2023.select('classification').rename('label'))
                   .updateMask(elliott_2023.select('classification').eq(1))
                   .stratifiedSample(
                       numPoints=n_samples, classBand='label',
                       region=simple_region, scale=SCALE,
                       tileScale=4, seed=42, geometries=False,
                   ))

    non_ice_samples = (features
                       .addBands(ee.Image(0).rename('label'))
                       .updateMask(elliott_2016.select('classification').eq(0))
                       .stratifiedSample(
                           numPoints=n_samples, classBand='label',
                           region=simple_region, scale=SCALE,
                           tileScale=4, seed=42, geometries=False,
                       ))

    return ice_samples.merge(non_ice_samples)


def submit_classification_exports(region_geometries, years=YEARS):
    """
    Train one RF classifier per region and apply it to all years.

    One classifier per region (not per year) is trained on a 2020 reference
    mosaic and applied to all years 2016–2025. This is more parsimonious
    than Elliott's 32-classifier approach and reduces overfitting to
    year-specific illumination or cloud conditions.

    Parameters
    ----------
    region_geometries : dict
        Output of ``build_region_geometries``.
    years : list of int
        Years to classify.

    Returns
    -------
    dict of region → dict of year → ee.batch.Task
    """
    tasks = {r: {} for r in region_geometries}

    for region, geom in region_geometries.items():
        print(f"\n── {region} ──────────────────────────────")

        # Reference mosaic: 2020 (mid-period, good image availability)
        ref_mosaic = ee.Image(f'{MOSAIC_ASSET_ROOT}/mosaic_{region}_2020')

        print(f"  Training RF classifier on 2020 reference mosaic...")
        training_data = _get_training_data(region, geom, ref_mosaic)

        classifier = (ee.Classifier.smileRandomForest(
            numberOfTrees=RF_N_TREES, bagFraction=0.5, seed=42
        ).train(
            features=training_data,
            classProperty='label',
            inputProperties=['NDSI_min', 'NDWI', 'B8'],
        ))

        for year in years:
            mosaic = ee.Image(f'{MOSAIC_ASSET_ROOT}/mosaic_{region}_{year}')
            classified = (mosaic
                          .select(['NDSI_min', 'NDWI', 'B8'])
                          .classify(classifier)
                          .rename('classification'))

            asset_id = f'{CLASSIFICATION_ASSET_ROOT}/classified_{region}_{year}'
            task = ee.batch.Export.image.toAsset(
                image=classified,
                description=f'classified_{region}_{year}',
                assetId=asset_id,
                region=geom,
                scale=SCALE,
                maxPixels=1e13,
                pyramidingPolicy={'.default': 'mode'},
            )
            task.start()
            tasks[region][year] = task
            print(f"  Submitted: classified_{region}_{year} — task {task.id}")

    total = sum(len(v) for v in tasks.values())
    print(f"\n{total} classification tasks submitted.")
    return tasks


# ---------------------------------------------------------------------------
# Post-classification filters
# ---------------------------------------------------------------------------

def _load_classification_stack(region, years):
    """
    Load annual classifications into a single multi-band image.
    Band names are year strings: '2016', '2017', etc.
    """
    images = [
        ee.Image(f'{CLASSIFICATION_ASSET_ROOT}/classified_{region}_{y}')
        .rename(str(y))
        for y in years
    ]
    return ee.Image.cat(images)


def _apply_gap_fill(stack, year_strs):
    """
    Fill no-data pixels by propagating the nearest valid classification.

    Persistent cloud cover creates gaps (masked pixels) in some years.
    Each gap is filled by looking backwards in time for the nearest valid
    classification, then forwards if no backward fill is available.
    Final fallback: non-ice (0) for completely unobserved pixels.
    """
    filled_bands = []
    for i, year in enumerate(year_strs):
        filled = stack.select(year)
        for j in range(i - 1, -1, -1):
            filled = filled.unmask(stack.select(year_strs[j]))
        for j in range(i + 1, len(year_strs)):
            filled = filled.unmask(stack.select(year_strs[j]))
        filled_bands.append(filled.unmask(0).rename(year))
    return ee.Image.cat(filled_bands)


def _apply_temporal_filter(stack, year_strs):
    """
    Correct physically implausible ice→non-ice→ice transitions.

    Glaciers only retreat — a pixel classified as non-ice should not
    subsequently be classified as ice. Four rules are applied:

    First-year rule : if year 1 is non-ice but years 2 & 3 are ice → set to ice
    Last-year rule  : if last year is ice but years n-2 & n-1 are non-ice → set to non-ice
    3-year rule     : single non-ice year surrounded by ice years → set to ice
    4-year rule     : two consecutive non-ice years surrounded by ice → set to ice
    """
    n = len(year_strs)
    corrected = {y: stack.select(y) for y in year_strs}

    # First-year rule
    y0, y1, y2 = year_strs[0], year_strs[1], year_strs[2]
    corrected[y0] = corrected[y0].where(
        corrected[y0].eq(0).And(corrected[y1].eq(1)).And(corrected[y2].eq(1)), 1
    )

    # Last-year rule
    yn, yn1, yn2 = year_strs[-1], year_strs[-2], year_strs[-3]
    corrected[yn] = corrected[yn].where(
        corrected[yn].eq(1).And(corrected[yn1].eq(0)).And(corrected[yn2].eq(0)), 0
    )

    # 3-year rule
    for i in range(1, n - 1):
        p, c, nx = year_strs[i-1], year_strs[i], year_strs[i+1]
        corrected[c] = corrected[c].where(
            corrected[c].eq(0).And(corrected[p].eq(1)).And(corrected[nx].eq(1)), 1
        )

    # 4-year rule
    for i in range(1, n - 2):
        p, c, nx, a = (year_strs[i-1], year_strs[i],
                       year_strs[i+1], year_strs[i+2])
        fix = (corrected[c].eq(0).And(corrected[nx].eq(0))
               .And(corrected[p].eq(1)).And(corrected[a].eq(1)))
        corrected[c] = corrected[c].where(fix, 1)
        corrected[nx] = corrected[nx].where(fix, 1)

    return ee.Image.cat([corrected[y].rename(y) for y in year_strs])


def _apply_frequency_filter(stack, year_strs, threshold=0.6):
    """
    Correct pixels classified as ice in > 60% of their active years.

    A pixel's 'active years' excludes trailing non-ice years at the end of
    the time series — these represent genuine melt and should not be reset.
    For example, a pixel with classifications [1,1,1,1,1,1,0,0] has 6 active
    years (frequency = 100%), and the trailing non-ice years are preserved.
    """
    n = len(year_strs)
    bands = [stack.select(y) for y in year_strs]

    # Identify trailing non-ice years (consecutive non-ice from end of series)
    trailing_nonice = ee.Image(1)
    trailing_masks = [None] * n
    for i in range(n - 1, -1, -1):
        trailing_nonice = trailing_nonice.And(bands[i].eq(0))
        trailing_masks[i] = trailing_nonice

    ice_count = ee.Image(0)
    active_count = ee.Image(0)
    for i, band in enumerate(bands):
        is_active = trailing_masks[i].Not()
        ice_count = ice_count.add(band.multiply(is_active))
        active_count = active_count.add(is_active)

    ice_freq = ice_count.divide(active_count.max(1))
    high_freq = ice_freq.gt(threshold)

    corrected_bands = []
    for i, year in enumerate(year_strs):
        band = stack.select(year)
        is_active = trailing_masks[i].Not()
        should_correct = high_freq.And(is_active).And(band.eq(0))
        corrected_bands.append(band.where(should_correct, 1).rename(year))

    return ee.Image.cat(corrected_bands)


def _apply_temporal_permanence_filter(stack, year_strs):
    """
    Enforce one-way ice→non-ice transition.

    Once a pixel becomes non-ice, all subsequent years are set to non-ice.
    This removes any remaining re-glaciation artefacts that survived the
    temporal filter, ensuring the time series is physically consistent.
    """
    corrected = {y: stack.select(y) for y in year_strs}
    for i in range(len(year_strs) - 1):
        c, nx = year_strs[i], year_strs[i + 1]
        corrected[nx] = corrected[nx].where(corrected[c].eq(0), 0)
    return ee.Image.cat([corrected[y].rename(y) for y in year_strs])


def _apply_spatial_filter(stack, year_strs, min_cluster_size=5):
    """
    Remove isolated ice pixel clusters smaller than ``min_cluster_size``.

    Small isolated clusters (< 5 pixels = 500 m² at 10m resolution) are
    likely classification errors from shadowed rock or spectral confusion,
    not genuine glaciers. The 5-pixel threshold is well below the 0.01 km²
    minimum glacier size in the RGI inventory.

    Uses GEE's ``connectedPixelCount`` to identify cluster sizes.
    """
    corrected_bands = []
    for year in year_strs:
        band = stack.select(year)
        connected = band.eq(1).connectedPixelCount(
            maxSize=min_cluster_size + 1, eightConnected=True
        )
        small_cluster = connected.lt(min_cluster_size).And(band.eq(1))
        corrected_bands.append(band.where(small_cluster, 0).rename(year))
    return ee.Image.cat(corrected_bands)


def submit_filtered_classification_exports(region_geometries, years=YEARS):
    """
    Apply all five post-classification filters and export results.

    Filters applied in order:
    1. Gap fill          — fills no-data from persistent cloud
    2. Temporal filter   — corrects implausible re-glaciation
    3. Frequency filter  — corrects pixels that are mostly-ice-but-not-always
    4. Temporal permanence — enforces one-way ice→non-ice transition
    5. Spatial filter    — removes isolated salt-and-pepper clusters

    Exports each region×year to ``FILTERED_ASSET_ROOT`` as a single-band
    image with band name 'classification' (1=ice, 0=non-ice).

    Parameters
    ----------
    region_geometries : dict
        Output of ``build_region_geometries``.
    years : list of int
        Years to filter and export.

    Returns
    -------
    list of ee.batch.Task
    """
    year_strs = [str(y) for y in years]
    all_tasks = []

    for region, geom in region_geometries.items():
        print(f"\nProcessing {region}...")
        stack = _load_classification_stack(region, years)
        stack = _apply_gap_fill(stack, year_strs)
        stack = _apply_temporal_filter(stack, year_strs)
        stack = _apply_frequency_filter(stack, year_strs)
        stack = _apply_temporal_permanence_filter(stack, year_strs)
        stack = _apply_spatial_filter(stack, year_strs)

        for year in years:
            asset_id = f'{FILTERED_ASSET_ROOT}/classified_{region}_{year}_filtered'
            single_year = stack.select(str(year)).rename('classification')
            task = ee.batch.Export.image.toAsset(
                image=single_year,
                description=f'filtered_{region}_{year}',
                assetId=asset_id,
                region=geom,
                scale=SCALE,
                maxPixels=1e13,
                pyramidingPolicy={'.default': 'mode'},
            )
            task.start()
            all_tasks.append(task)
            print(f"  Submitted: filtered_{region}_{year}")

    print(f"\n{len(all_tasks)} filtered classification exports submitted.")
    print("Monitor at: https://code.earthengine.google.com/tasks")
    return all_tasks


# ---------------------------------------------------------------------------
# Validation against Elliott
# ---------------------------------------------------------------------------

def validate_against_elliott(region, year, region_geom):
    """
    Compute spatial overlap between our filtered classification and Elliott's.

    Reproduces the validation metric from Andriychenko (2024): overlap is
    the fraction of Elliott's ice pixels that are also classified as ice by
    our method. Computed at 30m resolution for speed (area estimates remain
    accurate).

    Parameters
    ----------
    region : str
        Region identifier (e.g. 'R1a').
    year : int
        Year to validate (2016 or 2023).
    region_geom : ee.Geometry
        Region analysis geometry.

    Returns
    -------
    dict with keys: ``elliott_km2``, ``our_km2``, ``overlap_km2``,
    ``overlap_pct``.
    """
    our_filtered = ee.Image(
        f'{FILTERED_ASSET_ROOT}/classified_{region}_{year}_filtered'
    ).select('classification').eq(1)

    elliott = ee.Image(
        f'{ELLIOTT_ASSET_ROOT}/{region}/{year}_final'
    ).select('classification').eq(1)

    pixel_area = ee.Image.pixelArea().divide(1e6)

    def get_area(image):
        return list(image.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region_geom,
            scale=30,
            maxPixels=1e13,
            tileScale=4,
        ).getInfo().values())[0]

    elliott_km2 = get_area(elliott)
    our_km2 = get_area(our_filtered)
    overlap_km2 = get_area(our_filtered.And(elliott))
    overlap_pct = overlap_km2 / elliott_km2 * 100

    print(f"\n── {region} {year} Validation ──────────────────────")
    print(f"Elliott ice area:    {elliott_km2:.1f} km²")
    print(f"Our filtered area:   {our_km2:.1f} km²")
    print(f"Overlap:             {overlap_pct:.1f}%")

    return {
        'elliott_km2':  elliott_km2,
        'our_km2':      our_km2,
        'overlap_km2':  overlap_km2,
        'overlap_pct':  overlap_pct,
    }


# ---------------------------------------------------------------------------
# Main pipeline (run in order)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Step 1: Build region geometries
    region_geometries = build_region_geometries()

    # Step 2: Export annual mosaics (submit and wait for completion)
    # submit_mosaic_exports(region_geometries)

    # Step 3: Train classifiers and export raw classifications
    # submit_classification_exports(region_geometries)

    # Step 4: Apply post-classification filters and export
    # submit_filtered_classification_exports(region_geometries)

    # Step 5: Validate against Elliott for R1a 2016 and 2023
    # validate_against_elliott('R1a', 2016, region_geometries['R1a'])
    # validate_against_elliott('R1a', 2023, region_geometries['R1a'])
