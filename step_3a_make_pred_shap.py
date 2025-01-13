"""
Step 3a: Make test predictions, compute scores, and get SHAP-based feature importance for trained model

Define all settings for this script in `config.py`.
"""

import logging
from typing import List

import flaml
import joblib
import numpy as np
import pandas as pd
import shap
import xarray as xr

import config
from otclim.logging import get_logger
from otclim.ml import split_test_train_temp

logging.basicConfig(level="INFO")
logger = get_logger("eval")


def tree_shap_fi(model, features: List, X: np.ndarray) -> shap.Explanation:
    """Compute SHAP values for a tree-based model and return shap.Explanation object."""

    # Generate background dataset by random sampling
    n_bg_samples = 500
    random_state = 42
    if len(X) > n_bg_samples:
        rng = np.random.default_rng(seed=random_state)
        rand_inds = rng.choice(len(X), size=n_bg_samples, replace=False)
        Xe = X[rand_inds]
    else:
        Xe = X

    explainer = shap.TreeExplainer(model, feature_perturbation="interventional", data=Xe)
    try:
        # Having dataframes as inputs to explainer seems to cause additivity check exceptions. Ensure numpy array.
        # Source; https://stackoverflow.com/questions/68233466/shap-exception-additivity-check-failed-in-treeexplainer
        res = explainer(np.array(X))
    except shap.utils._exceptions.ExplainerError as e:
        # Still fallback to disabling additivity check
        logger.error("SHAP additivity check failed! Running without it. Be careful.")
        logger.error("The following exception was raised:")
        logger.error(e)
        res = explainer(np.array(X), check_additivity=False)
    res.feature_names = features  # Assign features to the result
    return res


if __name__ == "__main__":
    # Load model
    logger.info("Loading model...")
    model: flaml.automl.AutoML = joblib.load(config.OUT_DIR / "flaml_trained.joblib")

    # Load dataset, select features and target, get test split
    logger.info("Loading dataset...")
    ds = xr.open_dataset(config.DATA_FINAL / "era5_obs_merged.nc")
    ds = ds[config.FEATURES + [config.TARGET]]
    df_train, df_test = split_test_train_temp(ds)

    # Load transformer
    pre_train_tf = joblib.load(config.OUT_DIR / "pre_train_tf.joblib")
    y_true = df_test[config.TARGET].values
    y_true_tf = pre_train_tf.transform(y_true.reshape(-1, 1)).ravel()

    # Make predictions
    logger.info("Making predictions...")
    X_test = df_test[config.FEATURES].to_numpy()
    y_pred_tf = model.predict(X_test)
    y_pred = pre_train_tf.inverse_transform(y_pred_tf.reshape(-1, 1)).ravel()

    # Save predictions
    df_pred = pd.DataFrame.from_dict(
        {
            f"{config.TARGET}_true": y_true,
            f"{config.TARGET}_pred": y_pred,
        }
    ).set_index(df_test.index)
    df_pred.to_csv(config.OUT_DIR / "test_predictions.csv")

    # Compute scores
    logger.info("Computing scores...")
    scores = {}
    for name, fn in config.METRICS.items():
        score = fn(y_true, y_pred)
        score_tf = fn(y_true_tf, y_pred_tf)
        scores[name] = [score, score_tf]

    df_scores = pd.DataFrame.from_dict(scores, orient="index", columns=["score", "score_tf"])
    df_scores.to_csv(config.OUT_DIR / "test_scores.csv")

    # Compute SHAP values
    logger.info("Computing SHAP values...")
    shap_values = tree_shap_fi(
        model.model.model,
        config.FEATURES,
        X=df_train[config.FEATURES].to_numpy(),  # evaluate on train data for cleanest feature importance
    )
    joblib.dump(shap_values, config.OUT_DIR / "shap_values.joblib")

    # Compute feature importance
    fi_values = np.abs(shap_values.values).mean(axis=0)
    df_fi = pd.DataFrame(data=fi_values, index=config.FEATURES, columns=["fi_shap"])
    df_fi = df_fi.sort_values("fi_shap", ascending=False)
    df_fi.to_csv(config.OUT_DIR / "shap_fi.csv")

    logger.info("Done!")
