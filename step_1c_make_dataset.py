"""
Step 1c: Merge ERA5 data with observed Cn2.

Define all settings for this script in `config.py`.
"""

import logging

import pandas as pd
import xarray as xr

import config
import otclim

logging.basicConfig(level=logging.INFO)
logger = otclim.logging.get_logger("data")


def resample_30min_mean_center(da_v: xr.DataArray) -> xr.DataArray:
    """Smooth signal by resampling to +-15 min around t. Attach timestamp to **center** of the interval, i.e. t."""
    da_v = da_v.resample(time="30min", label="left", offset="15min", skipna=True).mean()
    da_v = da_v.assign_coords(time=da_v.time + pd.Timedelta("15min"))
    return da_v


def load_cn2() -> xr.Dataset:
    """Load Cn2 data
    
    THIS IS AN EXAMPLE. USERS NEED TO IMPLEMENT THEIR OWN FUNCTION.
    
    This function needs to return a dataset with two dimensions:
    - time: datetime64 (overlapping with ERA5 data)
    - station: str (station identifier, aligning with defined stations in `config.py`)
    
    The dataset should contain at least one target variable, for example:
    - cn2: float32 (Cn2 values in m^-2/3) NON-logarithmic
    - ct2: float32 (Ct2 values in m^-2/3) NON-logarithmic
    """ ""

    # Load CSV file with two columns: DateTime and CN2
    df = pd.read_csv(config.DATA_RAW_CN2ML / "LAS120001_filtered.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])  # it is important that the index is a datetime64 object
    df = df.sort_values(by="DateTime").dropna()
    time = df["DateTime"].values
    cn2 = df["CN2"].values

    # Convert into xarray dataarray
    da_cn2 = xr.DataArray(cn2, dims=["time"], coords={"time": time})

    # Do preprocessing, such as resampling/smoothing, if necessary
    da_cn2 = resample_30min_mean_center(da_cn2)

    # Create dataset (needed if multiple targets exist)
    # and add dimension for station (although we only have one here)
    ds = xr.Dataset({"Cn2": da_cn2})
    ds = ds.expand_dims(station=["AE"])

    return ds


if __name__ == "__main__":
    # Load ERA5 data after feature engineering
    ds_era5 = xr.open_dataset(config.DATA_FE / "era5_features.nc")

    # Load Cn2 data (user implemented!)
    ds_cn2 = load_cn2()

    # Ensure Cn2 data come in expected format: 2 dims (time, station) and time is datetime64
    assert "time" in ds_cn2.dims
    assert "station" in ds_cn2.dims
    assert len(ds_cn2.dims) == 2
    assert ds_cn2["time"].dtype == "datetime64[ns]"

    # Merge datasets
    ds_final = xr.merge([ds_era5, ds_cn2], join="inner")

    if len(ds_final) == 0:
        raise ValueError("No overlapping data between ERA5 and Cn2 data!")
    if len(ds_final) < len(ds_cn2):
        logger.warning(
            f"Only {len(ds_final)} out of {len(ds_cn2)} Cn2 data points overlap with ERA5 data."
            f"This can be expected because ERA5 has 1h temporal resolution and your observations"
            f"are probably finer."
        )

    logger.info(f"Final dataset: {ds_final}")
    logger.info(f"Writing to {config.DATA_FINAL / 'era5_obs_merged.nc'}")
    ds_final.to_netcdf(config.DATA_FINAL / "era5_obs_merged.nc")
