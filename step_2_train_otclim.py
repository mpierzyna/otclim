"""
Step 2: Train gradient-boosting model with FLAML

Define all settings for this script in `config.py`.
"""

import logging

import flaml
import joblib
import xarray as xr
from sklearn.pipeline import make_pipeline

import config
from otclim.flaml_custom import add_custom_learners
from otclim.logging import get_logger
from otclim.ml import split_test_train_temp
from otclim.transform import QuantileMinMaxScaler

logging.basicConfig(level="INFO")
logger = get_logger("train")

if __name__ == "__main__":
    # Load dataset and select only features and target
    ds = xr.open_dataset(config.DATA_FINAL / "era5_obs_merged.nc")
    ds = ds[config.FEATURES + [config.TARGET]]

    # Split data (test data not needed for now)
    logger.info("Splitting data...")
    df_train, _ = split_test_train_temp(ds)
    X_train = df_train[config.FEATURES].to_numpy()
    y_train = df_train[config.TARGET].to_numpy().reshape(-1, 1)

    # Fit pre-train transformer and save for later
    logger.info("Fitting pre-train transformer...")
    pre_train_tf = make_pipeline(
        # FunctionTransformer(func=np.log10, inverse_func=power_10, validate=True),  # first log-transform
        QuantileMinMaxScaler(qmin=0.25, qmax=0.75),  # then scale with inter-quartile range
    )
    y_train = pre_train_tf.fit_transform(y_train)
    joblib.dump(pre_train_tf, config.OUT_DIR / "pre_train_tf.joblib")

    # Transform target and start training
    logger.info("Starting training...")
    automl = flaml.AutoML(**config.FLAML_CONFIG)
    add_custom_learners(automl)
    automl.fit(X_train=X_train, y_train=y_train)

    # Save final model
    joblib.dump(automl, config.OUT_DIR / "flaml_trained.joblib")
