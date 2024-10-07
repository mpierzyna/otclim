import flaml.automl.model


class CustomLGBMEstimator(flaml.automl.model.LGBMEstimator):
    """LGBM with custom objective function. Check https://lightgbm.readthedocs.io/en/latest/Parameters.html"""

    OBJECTIVE = None  # Objective function to use

    def __init__(self, task="binary", **config):
        assert task == "regression", "Only regression is supported."
        config.update(
            {
                "objective": self.OBJECTIVE,
            }
        )
        super().__init__(task=task, **config)


class LGBML1Estimator(CustomLGBMEstimator):
    """LGBM Estimator with objective set to l1."""

    OBJECTIVE = "l1"


class LGBMHuberEstimator(CustomLGBMEstimator):
    """LGBM Estimator with objective set to huber."""

    OBJECTIVE = "huber"


class CustomXGBoostEstimator(flaml.automl.model.XGBoostEstimator):
    """XGBoost with custom objective function. Check https://xgboost.readthedocs.io/en/stable/parameter.html"""

    OBJECTIVE = None  # Objective function to use

    def __init__(self, task="binary", **config):
        assert task == "regression", "Only regression is supported."
        config.update(
            {
                "objective": self.OBJECTIVE,
            }
        )
        super().__init__(task=task, **config)


class XGBL1Estimator(CustomXGBoostEstimator):
    OBJECTIVE = "reg:absoluteerror"


class XGBHuberEstimator(CustomXGBoostEstimator):
    OBJECTIVE = "reg:pseudohubererror"


default_learners = [
    "lgbm",
    "xgboost",
    "xgb_limitdepth",
    "catboost",
    "extra_tree",
    "rf",
]

custom_learner_dict = {
    "lgbm_l1": LGBML1Estimator,
    "lgbm_hub": LGBMHuberEstimator,
    "xgb_l1": XGBL1Estimator,
    "xgb_hub": XGBHuberEstimator,
}
custom_learners = list(custom_learner_dict.keys())


def add_custom_learners(automl: flaml.automl.automl.AutoML) -> None:
    for name, learner in custom_learner_dict.items():
        automl.add_learner(name, learner)
