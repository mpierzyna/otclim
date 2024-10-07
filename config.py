from __future__ import annotations

import pathlib
from typing import List, Tuple, Dict

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import otclim

# # Dataset settings
# Variables to download from ERA5 in step 1a
ERA5_VARS: List[str] = [
    "10u",
    "10v",
    "100u",
    "100v",
    "2d",
    "2t",
    "bld",
    "blh",
    "cape",
    "dctb",
    "dndza",
    "gwd",
    "i10fg",
    "ie",
    "ishf",
    "lcc",
    "mean_surface_direct_short_wave_radiation_flux",  # msdswrf
    "mean_surface_downward_long_wave_radiation_flux",  # msdlwrf
    "mean_surface_downward_short_wave_radiation_flux",  # msdswrf
    "msl",
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",  # msnlwrf
    "mean_surface_net_short_wave_radiation_flux",  # msnswrf
    "skt",
    "stl1",
    "tcc",
    "tplb",
    "tplt",
    "zust",
]

# Compute bounding box around lat/lon pair for which ERA5 data will be downloaded
# Alternatively, provide (max_lat, min_lon, min_lat, max_lon) directly
LAT, LON = 52, 4.37  # TU Delft Aerospace building
ERA5_BOX: Tuple[float, float, float, float] = otclim.era5.compute_bounding_box(lat=LAT, lon=LON)

# Time range for which to download ERA5 data
ERA5_YEARS: List[int] = [2022, 2023, 2024]
ERA5_MONTHS: List[int] | None = None  # all months
ERA5_DAYS: List[int] | None = None  # all days

ERA5_YEARS: List[int] = [2022]
ERA5_MONTHS: List[int] | None = [1]  # all months
ERA5_DAYS: List[int] | None = [1]  # all days

# Define the stations for which we have Cn2 observations
STATIONS: Dict[str, Tuple[float, float]] = {
    "AE": (52, 4.37),  # TU Delft Aerospace building
}

# Path to store raw ERA5 data
DATA_ROOT = pathlib.Path("data/tud_ae")
DATA_RAW_ERA5 = DATA_ROOT / "1_raw_era5"  # does not need to be changed
DATA_RAW_CN2ML = DATA_ROOT / "1_raw_cn2"  # does not need to be changed
DATA_FE = DATA_ROOT / "2_features"  # does not need to be changed
DATA_FINAL = DATA_ROOT / "3_final"  # does not need to be changed


# # ML settings
# Explicitly specify features and target that are expected in dataset.
FEATURES: List[str] = [
    "bld",
    "blh",
    "cape",
    "d2m",
    "dctb",
    "dndza",
    "gwd",
    "i10fg",
    "ie",
    "ishf",
    "lcc",
    "msdrswrf",
    "msdwlwrf",
    "msdwswrf",
    "msl",
    "mslhf",
    "msnlwrf",
    "msnswrf",
    "skt",
    "stl1",
    "t2m",
    "tcc",
    "tplb",
    "tplt",
    "zust",
    "m10",
    "m100",
    "alpha",
    "sin_X10",
    "cos_X10",
    "sin_X100",
    "cos_X100",
    "beta",
    "dT0",
    "dT1",
    "dTd",
    "sin_hr",
    "cos_hr",
    "sin_day",
    "cos_day",
    "sin_month",
    "cos_month",
]
TARGET: str = "Cn2"

# Temporal splitting of data
TRAIN_YEARS: List[str] = ["2022"]
TRAIN_STATION: str = "AE"  # only one station for training supported right now
TEST_YEARS: List[str] | None = ["2022"]  # Specify test years or None to use all years not in TRAIN_YEARS

# Output directory for storing results
OUT_DIR = pathlib.Path("output")

# FLAML settings
FLAML_CONFIG = {
    "time_budget": 30,  # in s
    "task": "regression",
    # 5-fold CV WITHOUT shuffling to not create correlated folds
    "eval_method": "cv",
    "split_type": KFold(n_splits=5, shuffle=False),
    "early_stop": True,
    "ensemble": False,
    "estimator_list": ["lgbm", "xgboost", "xgb_limitdepth", "lgbm_l1", "lgbm_hub", "xgb_l1", "xgb_hub"],
    "metric": "rmse",
    "n_jobs": -1,
    "n_members": None,
    "retrain_full": True,
    "sample": True,
    "seed": 0,
    "verbose": 3,
    "log_file_name": str(OUT_DIR / "flaml.log"),
}

# Metrics for evaluation
METRICS = {
    "rmse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
    "mae": mean_absolute_error,
    "r2": r2_score,
    "pearsonr": lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
}
