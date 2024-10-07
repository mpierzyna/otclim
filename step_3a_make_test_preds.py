"""
Step 3a: Make test predictions with trained model

Define all settings for this script in `config.py`.
"""

import flaml
import joblib
import pandas as pd
import xarray as xr

import config
from otclim.ml import split_test_train_temp

if __name__ == "__main__":
    # Load model
    model: flaml.automl.AutoML = joblib.load(config.OUT_DIR / "flaml_trained.joblib")

    # Load dataset, select features and target, get test split
    ds = xr.open_dataset(config.DATA_FINAL / "era5_obs_merged.nc")
    ds = ds[config.FEATURES + [config.TARGET]]
    _, df_test = split_test_train_temp(ds)

    # Load transformer
    pre_train_tf = joblib.load(config.OUT_DIR / "pre_train_tf.joblib")
    y_true = df_test[config.TARGET].values
    y_true_tf = pre_train_tf.transform(y_true.reshape(-1, 1)).ravel()

    # Make predictions
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
    scores = {}
    for name, fn in config.METRICS.items():
        score = fn(y_true, y_pred)
        score_tf = fn(y_true_tf, y_pred_tf)
        scores[name] = [score, score_tf]

    df_scores = pd.DataFrame.from_dict(scores, orient="index", columns=["score", "score_tf"])
    df_scores.to_csv(config.OUT_DIR / "test_scores.csv")
