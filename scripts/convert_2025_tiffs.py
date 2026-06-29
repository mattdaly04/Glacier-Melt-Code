"""
Convert 2025 simulation GeoTIFF exports to per-region parquet files.

This is a one-time data preparation script run on the remote machine
after GEE exports are complete. The resulting parquet files are the
canonical simulation inputs, archived on Zenodo.

Usage
-----
    uv run python scripts/convert_sim_tiffs.py

Paths are set for the remote machine. Adjust TIFF_DIR and OUTPUT_DIR
if running elsewhere.
"""

from pathlib import Path
from glacier_melt.sampling import convert_sim_tiffs_to_parquet

TIFF_DIR = Path("/nvme1/users/md962/glacier/Glacier Project/sim_tiffs/sim_tiffs/")
OUTPUT_DIR = Path("/nvme1/users/md962/glacier/Glacier Project/Sim_Merged/")

if __name__ == "__main__":
    convert_sim_tiffs_to_parquet(
        tiff_dir=TIFF_DIR,
        output_dir=OUTPUT_DIR,
    )
