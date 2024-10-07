import logging
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

import config
from otclim.logging import get_logger

logging.basicConfig(level="INFO")
logger = get_logger("data")


def split_test_train_temp(ds: xr.Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split for temporal extrapolation study and convert to pandas DataFrame.
    Use only one station with different train and test intervals.
    """

    def _ds_to_pd(ds: xr.Dataset) -> pd.DataFrame:
        """Stack dataset, convert to pandas, drop NaNs"""
        return ds.to_dataframe().dropna()

    # Select only training station
    ds = ds.sel(station=config.TRAIN_STATION)

    # Create masks for temporal splitting of data
    time_train_mask = np.zeros(ds["time"].shape, dtype=bool)
    for y in config.TRAIN_YEARS:
        time_train_mask |= ds["time"].isin(ds.sel(time=y)["time"])
    if config.TEST_YEARS is None:
        time_test_mask = ~time_train_mask
    else:
        time_test_mask = np.zeros(ds["time"].shape, dtype=bool)
        for y in config.TEST_YEARS:
            time_test_mask |= ds["time"].isin(ds.sel(time=y)["time"])

    # Ensure that there is no temporal overlap
    assert not np.any(time_train_mask & time_test_mask)

    # Split data
    ds_train = ds.sel(time=time_train_mask)
    ds_test = ds.sel(time=time_test_mask)

    # Convert to pandas
    df_train = _ds_to_pd(ds_train)
    df_test = _ds_to_pd(ds_test)

    # Check that datasets are not empty
    if len(df_train) == 0:
        raise ValueError("No training data selected!")
    if len(df_test) == 0:
        raise ValueError("No testing data selected!")

    logger.info(f"Selected {len(df_train)} training samples and {len(df_test)} testing samples.")
    return df_train, df_test
