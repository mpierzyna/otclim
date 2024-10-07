from __future__ import annotations

import concurrent.futures
import pathlib
from typing import List, Tuple
import pkg_resources

import cdsapi
import numpy as np


print(f"cdsapi version: {pkg_resources.get_distribution('cdsapi').version}")


def compute_bounding_box(lon: float, lat: float) -> Tuple[float, float, float, float]:
    """Compute the bounding box for a given lon/lat pair."""
    # Find the closest grid point
    grid_lon = round(lon * 4) / 4
    grid_lat = round(lat * 4) / 4

    # Compute the bounding box
    min_lon = grid_lon - 0.5
    max_lon = grid_lon + 0.5
    min_lat = grid_lat - 0.5
    max_lat = grid_lat + 0.5

    # Return NWSE tuple
    return (max_lat, min_lon, min_lat, max_lon)


def get_sl(
    variables: List[str],
    area: Tuple[float, float, float, float],
    years: List[int],
    months: List[int] | None,
    days: List[int] | None,
    output_file: str,
):
    """Download ERA5 single-level data.

    Parameters
    ----------
    variables : List[str]
        List of variables to download.
    area : Tuple[float, float, float, float]
        Area to download data for (N, W, S, E).
    years : List[int]
        List of years to download data for.
    months : List[int]
        List of months to download data for. If `None`, download full year.
    days : List[int]
        List of days to download data for. If `None`, download full month.
    output_file : str
        Path to save the downloaded data to.
    """
    # Default: Download full year
    if months is None:
        months = np.arange(1, 13)

    # Default: Download full month
    if days is None:
        days = np.arange(1, 32)

    if pathlib.Path(output_file).exists():
        print(f"Skipping {output_file} because it already exists.")
        return
    print(f"Requesting {variables}...")

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": variables,
            "area": list(area),
            "year": [f"{y}" for y in years],
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in days],
            "time": [f"{t:02d}:00" for t in np.arange(0, 24)],  # always full days
            "data_format": "netcdf",  # default is grib
            "download_format": "unarchived",
        },
        output_file,
    )


def get_sl_parallel(
    variables: List[str],
    area: Tuple[float, float, float, float],
    years: List[int],
    months: List[int] | None,
    days: List[int] | None,
    root: pathlib.Path,
    n_workers: int = 10,
):
    """Download ERA5 single-level data in parallel.

    - Submitting requests in parallel can accelerate the download process because the next request is already queued
      while the current one is being processed.
    - Downloading one variable at a time allows to later add more variables easily for the same location.
    - See `get_sl` function for parameter descriptions.

    Parameters
    ----------
    root : pathlib.Path
        Root directory to save the downloaded data to.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for v in variables:
            fname = f"ERA5_SL_{v}_{years[0]}_{years[-1]}.nc"
            future = executor.submit(
                get_sl,
                variables=[
                    v,
                ],
                area=area,
                years=years,
                months=months,
                days=days,
                output_file=str(root / fname),
            )
            futures.append(future)

        # Check if errors occurred
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
                raise e
