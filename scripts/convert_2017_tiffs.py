"""
Convert 2017 GEE GeoTIFF exports to per-region parquet files.

This is a one-time data preparation script run on the remote machine
after GEE exports are complete. The resulting parquet files are the
canonical training data, archived on Zenodo.

Usage
-----
    uv run python scripts/convert_2017_tiffs.py

Paths are set for the remote machine. Adjust TIFF_DIR and OUTPUT_DIR
if running elsewhere.
"""

from pathlib import Path
from glacier_melt.sampling import convert_tiffs_to_parquet

TIFF_DIR = Path("/nvme1/users/md962/glacier/Glacier Project/tiffs/")
OUTPUT_DIR = Path("/nvme1/users/md962/glacier/Glacier Project/Merged/")

if __name__ == "__main__":
    convert_tiffs_to_parquet(
        tiff_dir=TIFF_DIR,
        output_dir=OUTPUT_DIR,
    )
