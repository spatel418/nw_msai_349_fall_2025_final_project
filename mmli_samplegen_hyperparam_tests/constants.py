"""
Constants for MMLI Experiment
"""
from scipy.stats import uniform, randint, reciprocal

RANDOM_STATE = 13

MODEL_PARAM_RANGE = {
    "linear": {"fit_intercept": [True, False]},
    "bayesian": {
        "alpha_1": uniform(1e-6, 1e6),
        "alpha_2": uniform(1e-6, 1e6),
        "lambda_1": uniform(1e-6, 1e6),
        "lambda_2": uniform(1e-6, 1e6),
        "fit_intercept": [True, False],
    },
    "elasticnet": {"alpha": uniform(0.0001, 10), "l1_ratio": uniform(0, 1)},
    "xgboost": {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),
        "max_depth": randint(2, 6),
        "n_estimators": randint(100, 150),
        "subsample": uniform(0.6, 0.4),
        "nthread": [1],
    },
    "randomforest": {
        "n_estimators": randint(30, 200),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 7),
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    },
    "svm": {
        "C": uniform(0.1, 100),
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4, 5],
        "epsilon": uniform(0.01, 1.0),
    },
}

MODEL_PARAM_DEFAULT = {
    "linear": {"n_jobs": None},
    "bayesian": {"verbose": False},
    "elasticnet": {"random_state": RANDOM_STATE},
    "xgboost": {"nthread": 1, "verbosity": 0},
    "randomforest": {
        "random_state": RANDOM_STATE,
        "n_estimators": 30,
        "max_depth": 5,
        "max_features": "sqrt",
    },
    "svm": {"kernel": "rbf"},
}

SCORING = {
    "adj_r2": "r2",
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
}

OFFLINE_REGRESSION_METRICS = ["adj_r2", "r2", "mse", "mae"]
