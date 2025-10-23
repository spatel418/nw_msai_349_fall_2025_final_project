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
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4),
        "nthread": [1],
    },
    "randomforest": {
        "n_estimators": randint(30, 200),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 7),
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
        "oob_score": [True, False],
    },
    "svm": {
            "C": uniform(0.1, 100),  # Try values like 0.1 to 100
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto"] + list(reciprocal(1e-4, 1e1).rvs(100)),  # For non-linear kernels
            "degree": [2, 3, 4, 5],  # Only relevant for 'poly' kernel
            'epsilon': uniform(0.01, 1.0),
            "coef0": uniform(0.0, 1.0),  # Relevant for 'poly' and 'sigmoid'

    },
    "tabpfn": {
        "n_estimators": [6,8,10,12],
        "device": ["cpu"]
    }
}
# setting this to accomodate with multiprocecssing
MODEL_PARAM_DEFAULT = {
    "linear": {"n_jobs": None},
    "bayesian": {"verbose": False},
    "elasticnet": {"random_state": RANDOM_STATE},
    "xgboost": {"nthread": 1},
    "randomforest": {
        "random_state": RANDOM_STATE,
        "n_estimators": 30,
        "max_depth": 5,
        "max_features": "sqrt",
    },
    "svm": {"kernel": "rbf"},
    "tabpfn": {"n_jobs": 1,
               "device": "cpu"
    }
}

TREE_ENSEMBLES = ["randomforest", "xgboost"]

OFFLINE_REGRESSION_METRICS = ["adj_r2", "r2", "mse", "mae"]

SCORING = {
    "adj_r2": "r2",
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
}

# we do not want to display these parameters to the users as these are computationally necessary but not for the model learning.
MODEL_HYPERPARAMTERS_TO_NOT_DISPLAY = {
    "linear": ["copy_X", "fit_intercept", "n_jobs", "positive"],
    "bayesian": [
        "compute_score",
        "copy_X",
        "verbose",
        "fit_intercept",
        "max_iter",
        "tol",
    ],
    "elasticnet": [
        "copy_X",
        "random_state",
        "warm_start",
        "fit_intercept",
        "max_iter",
        "positive",
        "precompute",
        "tol",
        "selection",
    ],
    "xgboost": [
        "callbacks",
        "early_stopping_rounds",
        "eval_metric",
        "feature_types",
        "importance_type",
        "interaction_constraints",
        "multi_strategy",
        "n_jobs",
        "random_state",
        "sampling_method",
        "validate_parameters",
        "verbosity",
        "n_thread",
    ],
    "randomforest": ["n_jobs", "random_state", "verbose", "warm_start"],
    "svm":["cache_size", "verbose", "max_iter"],
    "tabpfn":["model_path", "device", "ignore_pretraining_limits","fit_mode", 
              "memory_saving_mode", "n_jobs", "random_state", "inference_config", "differentiable_input"] 
}
