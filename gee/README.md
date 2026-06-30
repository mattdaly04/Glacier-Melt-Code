# gee/

Google Earth Engine scripts for the glacier coverage classification and data export pipeline. These scripts are run in a GEE-authenticated Python environment (Google Colab or local with `earthengine-api` installed).

## Prerequisites

- A Google Earth Engine account with access to the `glacier-melt` project
- Access to Elliott (2024)'s classification assets at `projects/ee-cavep-peruproject/`
- `earthengine-api` and `geemap` installed (`uv add earthengine-api geemap`)

## Scripts

| Script | Description |
|--------|-------------|
| `classify.py` | Full Sentinel-2 classification pipeline: mosaics → RF classification → 5 post-processing filters → filtered assets |
| `export_training.py` | Export 2017 training data (AE embeddings + EASD + melt labels) to Drive as GeoTIFF |
| `export_simulation.py` | Export 2025 simulation data (AE embeddings + EASD) to Drive as GeoTIFF |

## Pipeline order

Run in the order listed above. Each script submits GEE batch tasks that run asynchronously. Monitor progress at https://code.earthengine.google.com/tasks and wait for completion before running the next step.

```
classify.py
  └── submit_mosaic_exports()          # ~40 tasks, wait ~1hr
  └── submit_classification_exports()  # ~40 tasks, wait ~2hr
  └── submit_filtered_classification_exports()  # ~40 tasks, wait ~1hr

export_training.py
  └── submit_all_training_exports()    # 4 tasks (one per region), wait ~2hr

export_simulation.py
  └── submit_all_simulation_exports()  # 4 tasks, wait ~1hr
```

After the exports complete, download the GeoTIFF tiles from Google Drive and run the parquet conversion scripts:

```
uv run python scripts/convert_2017_tiffs.py
uv run python scripts/convert_2025_tiffs.py
```

## GEE asset structure

```
projects/glacier-melt/assets/
├── rgi70_low_latitudes          # RGI v7.0 polygons for Peru
├── mosaics/
│   └── mosaic_{region}_{year}   # Annual Sentinel-2 mosaics, 2016-2025
├── classifications/
│   └── classified_{region}_{year}           # Raw RF classifications
└── classifications_filtered/
    └── classified_{region}_{year}_filtered  # Post-processed classifications
```
