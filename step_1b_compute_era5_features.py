"""
Step 1b: Compute additional features from ERA5 data

Define all settings for this script in `config.py`.
"""

import logging

import numpy as np
import xarray as xr

import config
import otclim

logging.basicConfig(level=logging.INFO)
logger = otclim.logging.get_logger("data")


def compute_met_features(ds: xr.Dataset) -> xr.Dataset:
    """Feature engineering for ERA5"""
    # Horizontal wind magnitude
    U10 = ds["u10"]
    V10 = ds["v10"]
    M10 = np.sqrt(U10**2 + V10**2)
    ds["m10"] = M10
    ds = ds.drop_vars(["u10", "v10"])  # information redundant

    U100 = ds["u100"]
    V100 = ds["v100"]
    M100 = np.sqrt(U100**2 + V100**2)
    ds["m100"] = M100
    ds = ds.drop_vars(["u100", "v100"])  # information redundant

    # Vertical shear
    alpha = np.log(M100 / M10) / np.log(10)
    ds["alpha"] = alpha

    # Wind direction
    X10 = np.rad2deg(np.arctan2(-U10, -V10))
    X10 = xr.where(X10 < 0, X10 + 360, X10)
    ds["sin_X10"] = np.sin(np.deg2rad(X10))
    ds["cos_X10"] = np.cos(np.deg2rad(X10))

    X100 = np.rad2deg(np.arctan2(-U100, -V100))
    X100 = xr.where(X100 < 0, X100 + 360, X100)
    ds["sin_X100"] = np.sin(np.deg2rad(X100))
    ds["cos_X100"] = np.cos(np.deg2rad(X100))

    # Directional shear
    beta = np.abs(X100 - X10)
    beta = xr.where(beta > 180, 360 - beta, beta)
    ds["beta"] = beta

    # Temperature differences
    tsl = ds["stl1"]  # soil temperature (level 1)
    tsk = ds["skt"]  # skin temperature
    t2m = ds["t2m"]  # 2m temperature

    ds["dT0"] = tsl - tsk
    ds["dT1"] = tsk - t2m
    ds["dTd"] = ds["d2m"] - t2m  # dew-point spread

    return ds


def add_time_features(ds: xr.Dataset) -> xr.Dataset:
    """Add time features to ERA5 dataset"""
    ds_time_features = xr.Dataset(
        {
            "sin_hr": np.sin(2 * np.pi * ds["time"].dt.hour / 24),
            "cos_hr": np.cos(2 * np.pi * ds["time"].dt.hour / 24),
            "sin_day": np.sin(2 * np.pi * ds["time"].dt.dayofyear / 365),
            "cos_day": np.cos(2 * np.pi * ds["time"].dt.dayofyear / 365),
            "sin_month": np.sin(2 * np.pi * ds["time"].dt.month / 12),
            "cos_month": np.cos(2 * np.pi * ds["time"].dt.month / 12),
        }
    )

    # Add station dimension to time features
    ds_time_features = ds_time_features.expand_dims(station=ds["station"])

    # Merge with original dataset
    ds = xr.merge([ds, ds_time_features])
    return ds


if __name__ == "__main__":
    # Load all downloaded ERA5 data
    logger.info(f"Loading ERA5 data from {config.DATA_RAW_ERA5}")
    ds = xr.open_mfdataset(config.DATA_RAW_ERA5.glob("*.nc"), compat="override").load()

    # Collocate stations and ERA5 data
    lat, lon = zip(*config.STATIONS.values())
    lat = xr.DataArray(list(lat), dims="station")
    lon = xr.DataArray(list(lon), dims="station")
    ds = ds.sel(longitude=lon, latitude=lat, method="nearest")
    ds = ds.assign_coords(station=list(config.STATIONS.keys()))

    logger.info(f"Collocated grid points for station: {ds['station']}")

    # Rename ERA5 dim for convenience
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # Compute features
    logger.info("Computing additional features...")
    ds = compute_met_features(ds)
    ds = add_time_features(ds)
    logger.info(f"Features: {[v for v in ds.data_vars]}")

    # Check for NaNs
    has_nan = False
    for v in ds.data_vars:
        n = int(ds[v].isnull().sum())
        if n > 0:
            logger.warning(f"{v} has {n} NaNs")
            has_nan = True
    if not has_nan:
        logger.info("No NaNs found.")

    # Write to disk
    logger.info(f"Writing features to {config.DATA_FE / 'era5_features.nc'}")
    ds.to_netcdf(config.DATA_FE / "era5_features.nc")
