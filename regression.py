import os
import shutil
import json
from collections import Counter
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import (
    LinearRegression,
    BayesianRidge,
    ElasticNet,
)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
import xgboost as xgb
#from tabpfn import TabPFNRegressor
#from tabpfn_feature_importance_helper import TabPFNRegressorWithImportance
from regression_file_validator import (
    RegressionFileValidator,
)
from constants import (
    MODEL_PARAM_DEFAULT,
    MODEL_PARAM_RANGE,
    MODEL_HYPERPARAMTERS_TO_NOT_DISPLAY,
    OFFLINE_REGRESSION_METRICS,
    SCORING,
)
import logging

from data_augmentation_upsampling_vae_kde import data_augment_vae as data_augment_vae_kde
from data_augmentation_upsampling_vae_binning import data_augment_vae as data_augment_vae_binning
from data_augmentation_upsamping_gaussian_noise import add_gaussian_noise
from data_augmentation_upsampling_smoter import smoter_augmentation
# Import the diagnostic functions
from kde_debug import (
    diagnose_kde_augmentation,
    analyze_neighbor_interpolation,
    analyze_balance_factor_impact,
    test_custom_kde_parameters,
    simulate_augmentation_process
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get the root logger
logger = logging.getLogger()  # Or logging.getLogger(__name__) for module-specific


warnings.filterwarnings("ignore", category=DataConversionWarning)

def run_kde_diagnostics(df_train, target_column, bin_edges, num_bins=10, region_assignments=None):
    """
    Run a complete diagnostic on your KDE-based augmentation
    
    Parameters:
    -----------
    df_train : pandas DataFrame
        Input training dataframe with features and target
    target_column : str
        Name of the target column
    bin_edges : numpy array
        Region boundaries
    num_bins : int, default=10
        Number of regions
    region_assignments : numpy array, optional
        Region assignments for each sample
    """
    # Separate features and target
    X = df_train.drop(columns=[target_column])
    y = df_train[target_column]
    
    print("Starting KDE-based augmentation diagnostics...")
    print(f"Data shape: {X.shape}, Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"Number of regions: {num_bins}")
    
    # If region assignments are not provided, calculate them
    if region_assignments is None:
        region_assignments = np.digitize(y, bin_edges[1:-1])
    
    # 1. First, examine the distribution and KDE behavior
    print("\n1. Testing different KDE parameters to find optimal bandwidth...")
    kde_params = test_custom_kde_parameters(y.values, bin_edges)
    
    # 2. Initialize your DataAugmentationVAE
    print("\n2. Creating and fitting VAE augmenter...")
    augmenter = DataAugmentationVAE(
        num_regions=num_bins,
        balance_strategy='increase',  # Use 'increase' for uniform distribution
        min_samples_per_region=50,  # Set target count per region
        balance_factor=0.3,  # Balance between density and uniform sampling
    )
    
    # Fit the augmenter
    augmenter.fit(X, y, bin_edges, region_assignments)
    
    # 3. Diagnose the current KDE augmentation behavior
    print("\n3. Diagnosing current KDE augmentation behavior...")
    diagnostic_results = diagnose_kde_augmentation(augmenter)
    
    # 4. Analyze interpolation for underrepresented regions
    print("\n4. Analyzing interpolation for underrepresented regions...")
    
    # Find the most underrepresented region
    region_counts = np.bincount(region_assignments, minlength=num_bins)
    underrep_region = np.argmin(region_counts)
    
    print(f"Most underrepresented region: {underrep_region+1} with {region_counts[underrep_region]} samples")
    analyze_neighbor_interpolation(augmenter, underrep_region)
    
    # 5. Analyze impact of balance_factor
    print("\n5. Analyzing impact of balance_factor...")
    analyze_balance_factor_impact(augmenter, underrep_region)
    
    # 6. Simulate augmentation with different parameters
    print("\n6. Simulating augmentation with different parameters...")
    
    # Try different bandwidths
    print("\nTesting different bandwidths:")
    
    # Default bandwidth from augmenter
    default_bw = augmenter.kde.bandwidth
    
    # Try narrower bandwidth
    narrow_bw = default_bw * 0.3
    print(f"\nSimulating with narrow bandwidth ({narrow_bw:.4f})...")
    simulate_augmentation_process(y.values, bin_edges, bandwidth=narrow_bw, balance_factor=0.3)
    
    # Try wider bandwidth
    wide_bw = default_bw * 3.0
    print(f"\nSimulating with wide bandwidth ({wide_bw:.4f})...")
    simulate_augmentation_process(y.values, bin_edges, bandwidth=wide_bw, balance_factor=0.3)
    
    # Try different balance factors
    print("\nTesting different balance factors:")
    
    # More density-focused
    print("\nSimulating with density-focused balance (0.1)...")
    simulate_augmentation_process(y.values, bin_edges, bandwidth=default_bw, balance_factor=0.1)
    
    # More uniform-focused
    print("\nSimulating with uniform-focused balance (0.7)...")
    simulate_augmentation_process(y.values, bin_edges, bandwidth=default_bw, balance_factor=0.7)
    
    # 7. Print recommendations
    print("\n===== RECOMMENDATIONS =====")
    print("Based on the diagnostic results, consider the following changes:")
    
    # Check bandwidth effectiveness
    if diagnostic_results['kde']['bandwidth'] > diagnostic_results['kde']['scott_bandwidth'] * 2:
        print("1. The current bandwidth is quite large, which may be oversmoothing the density estimation.")
        print("   Consider using a smaller bandwidth (around 30-50% of the current value).")
    elif diagnostic_results['kde']['bandwidth'] < diagnostic_results['kde']['scott_bandwidth'] * 0.5:
        print("1. The current bandwidth is quite small, which may be focusing too much on existing data points.")
        print("   Consider using a larger bandwidth (around 2-3x the current value).")
    
    # Check balance factor effectiveness
    region_idx = underrep_region
    
    print(f"2. For the most underrepresented region (Region {region_idx+1}):")
    print("   - If analysis shows poor interpolation potential, consider adjusting the interpolation strategy.")
    print("   - If analysis shows good interpolation potential but poor targeting, adjust the balance_factor.")
    
    # Final recommendations based on simulation results
    print("\n3. Based on simulation results:")
    print("   - Choose the bandwidth and balance_factor combination that produces the most uniform combined distribution.")
    print("   - For severely underrepresented regions, consider using region-specific KDE or adaptive sampling.")


# Simple utility function for integration with your regression platform

def adj_r2(r2, n, p):
    """
    Calculate adjusted R-squared

    Parameters:
    r2 (float): R-squared value
    n (int): Sample size
    p (int): Number of predictors (excluding intercept)

    Returns:
    float: Adjusted R-squared value
    """
    # Check if we have enough samples relative to predictors
    if n <= p + 1:
        return float("nan")  # Not enough degrees of freedom

    # Correct formula for adjusted R-squared
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    # Adjusted R² should never exceed R²
    if adj_r2 > r2:
        return r2

    return adj_r2


class RegressionHyperparameterTune:
    def __init__(self, model_name, model, X, y, goal_metric, k=10):
        self.model_name = model_name
        self.model = model
        self.X = X
        self.y = y
        self.goal_metric = goal_metric
        self.k = k

    def tune_parameters_rscv(self):
        random_search = RandomizedSearchCV(
            self.model(),
            scoring=SCORING[self.goal_metric],
            param_distributions=MODEL_PARAM_RANGE[self.model_name],
            n_iter=50,
            random_state=13,
            cv=self.k if len(self.X) >= self.k else len(self.X),
        )
        random_search.fit(self.X, self.y)
        return random_search.best_params_


class RegressionTrain:
    def __init__(
        self,
        input_df,
        target,
        enable_parameter_tune,
        data_augmentation,
        feature_selection_autoselect,
        feature_selection_num,
        goal_metric,
        data_augmentation_method = None,
        data_augmentation_sectioning = None,
        data_augmentation_region_num = 10,
        data_augmentation_min_samples_per_region = 5,
        balance_strategy = "equal",
        k = 10,
        sampling_method = 'kfold'
    ):
        self.input = input_df
        self.target = target
        self.enable_parameter_tune = enable_parameter_tune
        self.data_augmentation = data_augmentation
        self.feature_selection_autoselect = feature_selection_autoselect
        self.feature_selection_num = feature_selection_num
        self.goal_metric = goal_metric
        self.data_augmentation_method = data_augmentation_method
        self.data_augmentation_sectioning = data_augmentation_sectioning
        self.data_augmentation_region_num = data_augmentation_region_num
        self.data_augmentation_min_samples_per_region = data_augmentation_min_samples_per_region
        self.balance_strategy = balance_strategy
        self.k = k 
        self.sampling_method = sampling_method

    def normalize(self, X, scaler=None):
        if scaler is None:
            scaler = MinMaxScaler()
            # need to still remove columns that have all the same value
            X_scaled = scaler.fit_transform(X)
            return X_scaled, scaler
        else:
            X_scaled = scaler.transform(X)
            return X_scaled

    def find_best_model(self, model_dict):
        X = (
            self.input.drop(columns=[self.target])
            .select_dtypes(include=["number"])
            .fillna(0)
        )
        y = self.input[[self.target]].fillna(self.input[self.target].mean())
        best_model = {
            "model_name": None,
            f"test_{self.goal_metric}": None,
            "goal_metric_per_folds": None,
            "model": None,
            "metrics": None,
            "selected_features": None,
            "test_true_predicted": None,
        }
        num_splits = self.k if len(X) >= self.k else len(X)
        for k in model_dict.keys():
            model_dict[k]["metrics"], goal_metric_per_folds = self.train_model(
                k, model_dict[k], X, y, n_splits=num_splits
            )
            # identifies which is the best model according to test goal metric. At the moment this only uses 1 model so code is not relavant but in the future we ewill compare multiple models
            # Check if this is the first model or if we need to compare with previous best
            current_metric = model_dict[k]["metrics"][f"test_{self.goal_metric}"]
            is_first_model = best_model[f"test_{self.goal_metric}"] is None

            # Determine if current model is better than previous best
            # Similar to previous code logic for error vs fit metrics
            if self.goal_metric in ["mae", "mse", "rmse"]:
                # For error metrics, lower is better
                is_better = (
                    is_first_model
                    or current_metric < best_model[f"test_{self.goal_metric}"]
                )
            else:  # 'r2' or 'adj_r2'
                # For fit metrics, higher is better
                is_better = (
                    is_first_model
                    or current_metric > best_model[f"test_{self.goal_metric}"]
                )

            # If this is the first model or it's better than the previous best, update all fields
            if is_better:
                best_model[f"test_{self.goal_metric}"] = current_metric
                best_model["model_name"] = k
                best_model["model"] = model_dict[k]["metrics"]["model"]
                best_model["metrics"] = model_dict[k]["metrics"]
                best_model["selected_features"] = model_dict[k]["metrics"][
                    "selected_features"
                ]
                best_model["goal_metric_per_folds"] = goal_metric_per_folds
                best_model["test_true_predicted"] = model_dict[k]["metrics"][
                    "test_true_predicted"
                ]

        # performing scaler creation twice, during training to find the best model then after training, chooseing the best model and performing normalization on the entire passed in training set.
        best_model["metrics"] = {
            k: 0 if np.isnan(v) else v
            for k, v in best_model["metrics"].items()
            if any(metric in k for metric in OFFLINE_REGRESSION_METRICS)
        }
        # remove non metric features from best_model["metrics"]
        scaler = self.normalize(
            X[best_model["selected_features"]]
        )
        best_model["scaler"] = scaler[1]
        return best_model
    
    def mark_organic_rows(self, augmented_df, organic_df, write_location):
        """
        Marks rows in augmented_df that exist in organic_df with an 'is_organic' column.
        
        Parameters:
        -----------
        augmented_df : pandas DataFrame
            The DataFrame containing both organic and synthetic rows
        organic_df : pandas DataFrame
            The DataFrame containing only original/organic rows
        write_location : string
            Location to write this dataframe as a CSV with the new "is_organic" column
            
        Returns:
        --------
        None
        """
        # Create a copy to avoid modifying the original
        result_df = augmented_df.copy()
        
        # Find columns that exist in both dataframes for comparison
        common_cols = list(set(result_df.columns).intersection(set(organic_df.columns)))
        
        # Don't use special columns like 'target_region' if they exist
        merge_cols = [col for col in common_cols if col not in ["target_region"]]
        
        # Add is_organic column to organic_df
        organic_df_with_flag = organic_df.copy()
        organic_df_with_flag['is_organic'] = True
        
        # Merge to identify original rows
        result_df = pd.merge(
            result_df,
            organic_df_with_flag[merge_cols + ['is_organic']],
            on=merge_cols,
            how='left'
        )
        
        # Fill NaN values with False
        result_df['is_organic'] = result_df['is_organic'].fillna(False)
        
        result_df.to_csv(write_location, index=False)

    def train_model(self, model_name, model_dict, X, y, n_splits):
        if hasattr(self, 'sampling_method') and self.sampling_method == "stratified":
            return self._train_model_stratified(model_name, model_dict, X, y)
        else:
            return self._train_model_kfold(model_name, model_dict, X, y, n_splits)

    def _train_model_stratified(self, model_name, model_dict, X, y):
        """
        Train model using stratified sampling based on K-means clustering.
        Uses 70% train, 30% test split and ensures proper representation of all clusters.
        """

        # Set up metrics collection
        goal_metric_per_folds = {"train": [], "test": []}
        
        # Create directory for stratified sampling results
        stratified_dir = "stratified-sampling"
        os.makedirs(stratified_dir, exist_ok=True)

        max_clusters = min(int(len(X) / 2), 5)
        X_scaled, cluster_scaler = self.normalize(X)

        # Try clustering with decreasing number of clusters until stratification works
        train_index, test_index = None, None
        for n_clusters in range(max_clusters, 4, -1):  # Try from max_clusters down to 5
            print(f"Attempting stratified sampling with {n_clusters} clusters...")
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Check if stratification is possible
            cluster_counts = Counter(cluster_labels)
            min_count = min(cluster_counts.values())
            
            # We need at least 2 samples per cluster to split between train and test
            if min_count >= 2:
                print(f"Using {n_clusters} clusters for stratification. Smallest cluster has {min_count} samples.")
                
                # Perform stratified train/test split (70/30)
                train_index, test_index = train_test_split(
                    np.arange(len(X)), 
                    test_size=0.3, 
                    random_state=42,
                    stratify=cluster_labels
                )
                
                # Create directory for this stratification
                iteration_dir = os.path.join(stratified_dir, "stratified_split")
                os.makedirs(iteration_dir, exist_ok=True)
                break
        
        # If stratification failed, fall back to random split
        if train_index is None:
            print("Stratification failed. Falling back to random 70/30 split.")
            train_index, test_index = train_test_split(
                np.arange(len(X)), 
                test_size=0.3, 
                random_state=13
            )
            
            # Create directory for random split
            iteration_dir = os.path.join(stratified_dir, "random_split")
            os.makedirs(iteration_dir, exist_ok=True)

        if self.data_augmentation:
            df_train_organic = pd.concat([X.iloc[train_index,], y.iloc[train_index]], axis=1)
            # Write organic data when augmentation is true
            df_train_organic.to_csv(os.path.join(iteration_dir, "organic_train.csv"), index=False)
            
            # Apply data augmentation (similar to your existing code)
            target_values = df_train_organic[self.target].values
            if self.data_augmentation_method == "gaussian":
                df_train = add_gaussian_noise(df_train_organic, self.target, n_samples_multiplier=3)
            elif self.data_augmentation_method == "smote":
                df_train = smoter_augmentation(df_train_organic, self.target)
            elif self.data_augmentation_method == "vae":
                # Your existing VAE code
                if self.data_augmentation_sectioning == "binning":
                    bin_edges = np.percentile(target_values, np.linspace(0, 100, self.data_augmentation_region_num + 1))
                    if len(np.unique(bin_edges)) < len(bin_edges):
                        # Fall back to equal-width regions
                        bin_edges = np.linspace(np.min(target_values), np.max(target_values), self.data_augmentation_region_num + 1)
                    bin_assignments = np.digitize(target_values, bin_edges[1:-1])
                    df_train = data_augment_vae_binning(
                        df_train=df_train_organic,
                        target_column=self.target,
                        bin_edges=bin_edges,
                        num_bins=self.data_augmentation_region_num,
                        balance_strategy=self.balance_strategy,
                        augmented_bucket_size=self.data_augmentation_min_samples_per_region,
                        bin_assignments=bin_assignments
                    )
                else:
                    # KDE
                    region_edges = np.percentile(target_values, np.linspace(0, 100, self.data_augmentation_region_num + 1))
                    if len(np.unique(region_edges)) < len(region_edges):
                        # Fall back to equal-width regions
                        region_edges = np.linspace(np.min(target_values), np.max(target_values), self.data_augmentation_region_num + 1)
                    region_assignments = np.digitize(target_values, region_edges[1:-1])
                    df_train = data_augment_vae_kde(
                        df_train=df_train_organic,
                        target_column=self.target,
                        bin_edges=region_edges,
                        num_bins=self.data_augmentation_region_num,
                        balance_strategy=self.balance_strategy,
                        augmented_bucket_size=self.data_augmentation_min_samples_per_region,
                        region_assignments=region_assignments
                    )
            
            # Mark organic rows versus non-organic
            self.mark_organic_rows(df_train, df_train_organic, os.path.join(iteration_dir, "train.csv"))
            
            # Process training data
            columns_to_drop = [col for col in [self.target, "target_region", "target_bucket"] if col in df_train.columns]
            X_train = df_train.drop(columns=columns_to_drop)
            columns = X_train.columns
            X_train, scaler = self.normalize(X_train)
            X_train = pd.DataFrame(X_train, columns=columns)
            y_train = df_train[[self.target]]
            
            X_train_organic = df_train_organic.drop(columns=[self.target])
            X_train_organic = pd.DataFrame(self.normalize(X_train_organic, scaler), columns=columns)
            y_train_organic = df_train_organic[[self.target]]
        else:
            X_train = X.iloc[train_index,]
            # Writing train to disk without augmentation
            pd.DataFrame(X_train, columns=X.columns).to_csv(os.path.join(iteration_dir, "train.csv"), index=False)
            X_train, scaler = self.normalize(X_train)
            X_train = pd.DataFrame(X_train, columns=X.columns)
            y_train = y.iloc[train_index]
        
        # Process test data
        X_test = X.iloc[test_index,]
        # Writing test fold to disk
        pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(iteration_dir, "test.csv"), index=False)
        X_test = pd.DataFrame(self.normalize(X_test, scaler), columns=X.columns)
        y_test = y.iloc[test_index]
        
        # Feature selection (if enabled)
        if self.feature_selection_autoselect:
            if len(X_train) < 10:  # Use sample size as CV if smaller than 10
                cv = len(X_train)
            else:
                cv = 10
            
            models = model_dict.copy()
            #if model_name == "tabpfn":
                #models["model"] = TabPFNRegressorWithImportance
                
            if self.feature_selection_num > 0:
                rfe = RFE(
                    estimator=models["model"](**MODEL_PARAM_DEFAULT[model_name]),
                    n_features_to_select=self.feature_selection_num,
                    step=1,
                )
            else:
                rfe = RFECV(
                    estimator=models["model"](**MODEL_PARAM_DEFAULT[model_name]),
                    step=1,
                    cv=cv,
                    scoring=SCORING[self.goal_metric],
                    min_features_to_select=1,
                )
            
            rfe.fit(X_train, y_train)
            selected_indices = rfe.get_support(indices=True)
            X_train = X_train.iloc[:, selected_indices]
            X_test = X_test.iloc[:, selected_indices]
        
        # Hyperparameter tuning (if enabled)
        if self.enable_parameter_tune:
            hyperparameter_tune = RegressionHyperparameterTune(
                model_name,
                model_dict["model"],
                X_train,
                y_train,
                self.goal_metric,
            )
            best_params = hyperparameter_tune.tune_parameters_rscv()
            logger.info(f"Best parameters: {best_params}")
            model = model_dict["model"](**best_params)
        else:
            model = model_dict["model"](**MODEL_PARAM_DEFAULT[model_name])
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Get predictions for both train and test
        if self.data_augmentation:
            X_train = X_train_organic[X_train.columns]
            y_train = y_train_organic
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Format predictions consistently
        if isinstance(y_train_pred[0], np.ndarray) and isinstance(y_test_pred[0], np.ndarray):
            y_train_pred = [float(pred[0]) for pred in y_train_pred]
            y_test_pred = [float(pred[0]) for pred in y_test_pred]
        else:
            y_train_pred = [float(pred) for pred in y_train_pred]
            y_test_pred = [float(pred) for pred in y_test_pred]
        
        # Calculate metrics
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "train_adj_r2": adj_r2(
                r2_score(y_train, y_train_pred), len(y_train), len(X.columns)
            ),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred),
            "test_adj_r2": adj_r2(
                r2_score(y_test, y_test_pred), len(y_test), len(X.columns)
            ),
            "test_r2": r2_score(y_test, y_test_pred),
            "test_true_predicted": {
                "train": [
                    {
                        "true": y_train[self.target].tolist()[i],
                        "predicted": y_train_pred[i],
                    }
                    for i in range(len(X_train))
                ],
                "test": [
                    {
                        "true": y_test[self.target].tolist()[i],
                        "predicted": y_test_pred[i],
                    }
                    for i in range(len(X_test))
                ],
            },
            "model": model,
            "selected_features": (
                X.columns.tolist()
                if not self.feature_selection_autoselect
                else X.columns[selected_indices].tolist()
            ),
            "hyperparameters": (
                best_params
                if self.enable_parameter_tune
                else MODEL_PARAM_DEFAULT[model_name]
            ),
        }
        
        # Save metrics
        goal_metric_per_folds["train"].append(metrics[f"train_{self.goal_metric}"])
        goal_metric_per_folds["test"].append(metrics[f"test_{self.goal_metric}"])
                
        # Save metrics to JSON
        json_serializable_metrics = {
            "metrics": {
                "train": {
                    "mae": metrics["train_mae"],
                    "mse": metrics["train_mse"],
                    "r2": metrics["train_r2"],
                    "adj_r2": metrics["train_adj_r2"]
                },
                "test": {
                    "mae": metrics["test_mae"],
                    "mse": metrics["test_mse"],
                    "r2": metrics["test_r2"],
                    "adj_r2": metrics["test_adj_r2"]
                }
            },
            "selected_features": metrics["selected_features"],
            "hyperparameters": metrics["hyperparameters"]
        }
        
        with open(os.path.join(iteration_dir, "scores.json"), "w") as f:
            json.dump(json_serializable_metrics, f, indent=4)
        
        # Return the single set of metrics
        return metrics, goal_metric_per_folds
    
    def _train_model_random(self, model_name, model_dict, X, y):
        """
        Train model using random sampling with optional data augmentation.
        Uses user-defined train percentage (self.train_pct) for the split.
        """
        # Set up metrics collection
        goal_metric_per_folds = {"train": [], "test": []}
        
        # Create directory for random sampling results
        random_dir = "random-sampling"
        os.makedirs(random_dir, exist_ok=True)
        iteration_dir = os.path.join(random_dir, "random_split")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Split the data
        train_index, test_index = train_test_split(
            np.arange(len(X)), 
            train_size=self.train_pct,
            random_state=13
        )
        
        # Handle data augmentation if enabled
        if self.data_augmentation:
            df_train_organic = pd.concat([X.iloc[train_index,], y.iloc[train_index]], axis=1)
            # Write organic data when augmentation is true
            df_train_organic.to_csv(os.path.join(iteration_dir, "organic_train.csv"), index=False)
            
            # Apply data augmentation based on method
            target_values = df_train_organic[self.target].values
            if self.data_augmentation_method == "gaussian":
                df_train = add_gaussian_noise(df_train_organic, self.target, n_samples_multiplier=3)
            elif self.data_augmentation_method == "smote":
                df_train = smoter_augmentation(df_train_organic, self.target)
            elif self.data_augmentation_method == "vae":
                # VAE with binning or KDE
                if self.data_augmentation_sectioning == "binning":
                    bin_edges = np.percentile(target_values, np.linspace(0, 100, self.data_augmentation_region_num + 1))
                    if len(np.unique(bin_edges)) < len(bin_edges):
                        # Fall back to equal-width regions
                        bin_edges = np.linspace(np.min(target_values), np.max(target_values), self.data_augmentation_region_num + 1)
                    bin_assignments = np.digitize(target_values, bin_edges[1:-1])
                    df_train = data_augment_vae_binning(
                        df_train=df_train_organic,
                        target_column=self.target,
                        bin_edges=bin_edges,
                        num_bins=self.data_augmentation_region_num,
                        balance_strategy=self.balance_strategy,
                        augmented_bucket_size=self.data_augmentation_min_samples_per_region,
                        bin_assignments=bin_assignments
                    )
                else:
                    # KDE
                    region_edges = np.percentile(target_values, np.linspace(0, 100, self.data_augmentation_region_num + 1))
                    if len(np.unique(region_edges)) < len(region_edges):
                        # Fall back to equal-width regions
                        region_edges = np.linspace(np.min(target_values), np.max(target_values), self.data_augmentation_region_num + 1)
                    region_assignments = np.digitize(target_values, region_edges[1:-1])
                    df_train = data_augment_vae_kde(
                        df_train=df_train_organic,
                        target_column=self.target,
                        bin_edges=region_edges,
                        num_bins=self.data_augmentation_region_num,
                        balance_strategy=self.balance_strategy,
                        augmented_bucket_size=self.data_augmentation_min_samples_per_region,
                        region_assignments=region_assignments
                    )
            
            # Mark organic rows versus non-organic
            self.mark_organic_rows(df_train, df_train_organic, os.path.join(iteration_dir, "train.csv"))
            
            # Process training data
            columns_to_drop = [col for col in [self.target, "target_region", "target_bucket"] if col in df_train.columns]
            X_train = df_train.drop(columns=columns_to_drop)
            columns = X_train.columns
            X_train, scaler = self.normalize(X_train)
            X_train = pd.DataFrame(X_train, columns=columns)
            y_train = df_train[[self.target]]
            
            X_train_organic = df_train_organic.drop(columns=[self.target])
            X_train_organic = pd.DataFrame(self.normalize(X_train_organic, scaler), columns=columns)
            y_train_organic = df_train_organic[[self.target]]
        else:
            # Without augmentation, use the original data
            X_train = X.iloc[train_index,]
            # Writing train to disk without augmentation
            pd.DataFrame(X_train, columns=X.columns).to_csv(os.path.join(iteration_dir, "train.csv"), index=False)
            X_train, scaler = self.normalize(X_train)
            X_train = pd.DataFrame(X_train, columns=X.columns)
            y_train = y.iloc[train_index]
        
        # Process test data
        X_test = X.iloc[test_index,]
        # Writing test fold to disk
        pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(iteration_dir, "test.csv"), index=False)
        X_test = pd.DataFrame(self.normalize(X_test, scaler), columns=X.columns)
        y_test = y.iloc[test_index]
        
        # Feature selection (if enabled)
        if self.feature_selection_autoselect:
            if len(X_train) < 10:  # Use sample size as CV if smaller than 10
                cv = len(X_train)
            else:
                cv = min(10, len(X_train))
            
            models = model_dict.copy()
            #if model_name == "tabpfn":
                #models["model"] = TabPFNRegressorWithImportance
                
            if self.feature_selection_num > 0:
                rfe = RFE(
                    estimator=models["model"](**MODEL_PARAM_DEFAULT[model_name]),
                    n_features_to_select=self.feature_selection_num,
                    step=1,
                )
            else:
                rfe = RFECV(
                    estimator=models["model"](**MODEL_PARAM_DEFAULT[model_name]),
                    step=1,
                    cv=cv,
                    scoring=SCORING[self.goal_metric],
                    min_features_to_select=1,
                )
            
            rfe.fit(X_train, y_train)
            selected_indices = rfe.get_support(indices=True)
            X_train = X_train.iloc[:, selected_indices]
            X_test = X_test.iloc[:, selected_indices]
            selected_features = X.columns[selected_indices].tolist()
        else:
            selected_features = X.columns.tolist()
        
        # Hyperparameter tuning (if enabled)
        if self.enable_parameter_tune:
            hyperparameter_tune = RegressionHyperparameterTune(
                model_name,
                model_dict["model"],
                X_train,
                y_train,
                self.goal_metric,
            )
            best_params = hyperparameter_tune.tune_parameters_rscv()
            logger.info(f"Best parameters: {best_params}")
            model = model_dict["model"](**best_params)
        else:
            model = model_dict["model"](**MODEL_PARAM_DEFAULT[model_name])
            best_params = MODEL_PARAM_DEFAULT[model_name]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Get predictions
        if self.data_augmentation:
            # Use organic data for evaluation when augmentation was used for training
            X_train = X_train_organic[X_train.columns]
            y_train = y_train_organic
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Format predictions
        if isinstance(y_train_pred[0], np.ndarray) and isinstance(y_test_pred[0], np.ndarray):
            y_train_pred = [float(pred[0]) for pred in y_train_pred]
            y_test_pred = [float(pred[0]) for pred in y_test_pred]
        else:
            y_train_pred = [float(pred) for pred in y_train_pred]
            y_test_pred = [float(pred) for pred in y_test_pred]
        
        # Calculate metrics
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "train_adj_r2": adj_r2(
                r2_score(y_train, y_train_pred), len(y_train), len(X.columns)
            ),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred),
            "test_adj_r2": adj_r2(
                r2_score(y_test, y_test_pred), len(y_test), len(X.columns)
            ),
            "test_r2": r2_score(y_test, y_test_pred),
            "test_true_predicted": {
                "train": [
                    {
                        "true": y_train[self.target].tolist()[i],
                        "predicted": y_train_pred[i],
                    }
                    for i in range(len(X_train))
                ],
                "test": [
                    {
                        "true": y_test[self.target].tolist()[i],
                        "predicted": y_test_pred[i],
                    }
                    for i in range(len(X_test))
                ],
            },
            "model": model,
            "selected_features": selected_features,
            "hyperparameters": best_params,
        }
        
        # Save metrics
        goal_metric_per_folds["train"].append(metrics[f"train_{self.goal_metric}"])
        goal_metric_per_folds["test"].append(metrics[f"test_{self.goal_metric}"])
        
        # Save metrics to JSON
        json_serializable_metrics = {
            "metrics": {
                "train": {
                    "mae": metrics["train_mae"],
                    "mse": metrics["train_mse"],
                    "r2": metrics["train_r2"],
                    "adj_r2": metrics["train_adj_r2"]
                },
                "test": {
                    "mae": metrics["test_mae"],
                    "mse": metrics["test_mse"],
                    "r2": metrics["test_r2"],
                    "adj_r2": metrics["test_adj_r2"]
                }
            },
            "selected_features": metrics["selected_features"],
            "hyperparameters": metrics["hyperparameters"]
        }
        
        with open(os.path.join(iteration_dir, "scores.json"), "w") as f:
            json.dump(json_serializable_metrics, f, indent=4)
        
        # For backward compatibility, create averaged metrics
        avg_fold_metrics = {
            "avg_train_mae": metrics["train_mae"],
            "avg_train_mse": metrics["train_mse"],
            "avg_train_adj_r2": metrics["train_adj_r2"],
            "avg_train_r2": metrics["train_r2"],
            "avg_test_mae": metrics["test_mae"],
            "avg_test_mse": metrics["test_mse"],
            "avg_test_adj_r2": metrics["test_adj_r2"],
            "avg_test_r2": metrics["test_r2"],
        }
        
        # Return the metrics
        return metrics, goal_metric_per_folds, avg_fold_metrics
    def _train_model_kfold(self, model_name, model_dict, X, y, n_splits=10):
        metrics = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=13)
        goal_metric_per_folds = {"train": [], "test": []}
        ###
        # Create main K-folds directory if it doesn't exist
        kfolds_dir = "k-folds"
        os.makedirs(kfolds_dir, exist_ok=True)
        ###
        for iteration, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"Fold {iteration + 1} / {n_splits}")
            ###
            # Create directory for this iteration
            iteration_dir = os.path.join(kfolds_dir, f"iteration_{iteration+1}")
            os.makedirs(iteration_dir, exist_ok=True)
            ###
            if self.data_augmentation:
                df_train_organic = pd.concat([X.iloc[train_index,], y.iloc[train_index]], axis=1)
                ###
                #write organic data with when augmentation is true
                df_train_organic.to_csv(os.path.join(iteration_dir, f"iteration_{iteration+1}_organic_train.csv"), index=False)
                ###
                target_values = df_train_organic[self.target].values
                if self.data_augmentation_method == "gaussian":
                    df_train = add_gaussian_noise(df_train_organic, self.target, n_samples_multiplier=3)
                elif self.data_augmentation_method == "smote":
                    df_train = smoter_augmentation(df_train_organic, self.target)
                elif self.data_augmentation_method == "vae":
                    if self.data_augmentation_sectioning == "binning":
                        bin_edges = np.percentile(target_values, np.linspace(0, 100, self.data_augmentation_region_num + 1))
                        if len(np.unique(bin_edges)) < len(bin_edges):
                            # Fall back to equal-width regions
                            bin_edges = np.linspace(np.min(target_values), np.max(target_values), self.data_augmentation_region_num + 1)
                        bin_assignments = np.digitize(target_values, bin_edges[1:-1])
                        df_train = data_augment_vae_binning(
                            df_train=df_train_organic,
                            target_column=self.target,
                            bin_edges=bin_edges,
                            num_bins=self.data_augmentation_region_num,
                            balance_strategy=self.balance_strategy,
                            augmented_bucket_size=self.data_augmentation_min_samples_per_region,
                            bin_assignments=bin_assignments
                        )
                    else:
                        #kde
                        #min_samples_per_region = data_augmentation_min_samples_per_region  # Changed from augmented_bucket_size for clarity
                        region_edges = np.percentile(target_values, np.linspace(0, 100, self.data_augmentation_region_num + 1))
                        if len(np.unique(region_edges)) < len(region_edges):
                            # Fall back to equal-width regions
                            region_edges = np.linspace(np.min(target_values), np.max(target_values), self.data_augmentation_region_num + 1)
                        region_assignments = np.digitize(target_values, region_edges[1:-1])
                        df_train = data_augment_vae_kde(
                            df_train=df_train_organic,
                            target_column=self.target,
                            bin_edges=region_edges,  # We keep the parameter name bin_edges for backward compatibility
                            num_bins=self.data_augmentation_region_num,    # We keep the parameter name num_bins for backward compatibility
                            balance_strategy=self.balance_strategy,
                            augmented_bucket_size=self.data_augmentation_min_samples_per_region,  # We keep the parameter name for backward compatibility
                            region_assignments=region_assignments
                        )
                ### 
                #marking organic rows versus non organic
                self.mark_organic_rows(df_train, df_train_organic, os.path.join(iteration_dir, f"iteration_{iteration+1}_train.csv"))
                ###
                columns_to_drop = [col for col in [self.target, "target_region", "target_bucket"] if col in df_train.columns]
                X_train = df_train.drop(columns=columns_to_drop)
                columns = X_train.columns
                X_train, scaler = self.normalize(X_train)
                X_train = pd.DataFrame(X_train, columns=columns)
                y_train = df_train[[self.target]]
                #columns_to_drop = [col for col in [self.target, "target_region", "target_bucket"] if col in df_train.columns]
                X_train_organic = df_train_organic.drop(columns=[self.target])
                X_train_organic = pd.DataFrame(self.normalize(X_train_organic, scaler), columns=columns)
                y_train_organic = df_train_organic[[self.target]]
            else:
                X_train = X.iloc[train_index,]
                ###
                # writing train to disk without augmentation
                pd.DataFrame(X_train, columns=X.columns).to_csv(os.path.join(iteration_dir, f"iteration_{iteration+1}_train.csv"), index=False)
                ###
                X_train, scaler = self.normalize(X_train)
                X_train = pd.DataFrame(X_train, columns=X.columns)
                y_train = y.iloc[train_index]
            X_test = X.iloc[test_index,]
            ###
            # writing test fold to disk
            pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(iteration_dir, f"iteration_{iteration+1}_test.csv"), index=False)
            #
            #X_train, scaler = self.normalize(X_train)
            #X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(self.normalize(X_test, scaler), columns=X.columns)
            #y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_test = y.iloc[test_index]
            if self.feature_selection_autoselect:
                if len(X_train) < n_splits:
                    cv = len(X_train)
                else:
                    cv = n_splits
                models = model_dict.copy()
                #if model_name == "tabpfn":
                    #models["model"] = TabPFNRegressorWithImportance
                    
                if self.feature_selection_num > 0:
                    rfe = RFE(
                        estimator=models["model"](**MODEL_PARAM_DEFAULT[model_name]),
                        n_features_to_select=self.feature_selection_num,
                        step=1,
                    )
                else:
                    rfe = RFECV(
                        estimator=models["model"](**MODEL_PARAM_DEFAULT[model_name]),
                        step=1,
                        cv=cv,
                        scoring=SCORING[self.goal_metric],
                        min_features_to_select=1,
                    )
                rfe.fit(X_train, y_train)
                selected_indices = rfe.get_support(indices=True)
                X_train = X_train.iloc[:, selected_indices]
                X_test = X_test.iloc[:, selected_indices]

            if self.enable_parameter_tune:
                hyperparameter_tune = RegressionHyperparameterTune(
                    model_name,
                    model_dict["model"],
                    X_train,
                    y_train,
                    self.goal_metric,
                )
                best_params = hyperparameter_tune.tune_parameters_rscv()
                logger.info(f"Best parameters: {best_params}")
                model = model_dict["model"](**best_params)
            else:
                model = model_dict["model"](**MODEL_PARAM_DEFAULT[model_name])
            model.fit(X_train, y_train)
            # Get predictions for both train and test
            # Change X_train_organic and y_train_organic to X_train and y_train because he original X_train + y_train with synthetic data is not needed anymore
            if self.data_augmentation:
                X_train = X_train_organic[X_train.columns]
                y_train = y_train_organic
            #
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # linear regression returns a format of [[1],[2],[3]] while other models return format [1,2,3]
            if isinstance(y_train_pred[0], np.ndarray) and isinstance(
                y_test_pred[0], np.ndarray
            ):
                y_train_pred = [float(pred[0]) for pred in y_train_pred]
                y_test_pred = [float(pred[0]) for pred in y_test_pred]
            else:
                y_train_pred = [float(pred) for pred in y_train_pred]
                y_test_pred = [float(pred) for pred in y_test_pred]
            fold_metrics = {
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "train_mse": mean_squared_error(y_train, y_train_pred),
                "train_adj_r2": adj_r2(
                    r2_score(y_train, y_train_pred), len(y_train), len(X.columns)
                ),
                "train_r2": r2_score(y_train, y_train_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "test_adj_r2": adj_r2(
                    r2_score(y_test, y_test_pred), len(y_test), len(X.columns)
                ),
                "test_r2": r2_score(y_test, y_test_pred),
                "test_true_predicted": {
                    "train": [
                        {
                            "true": y_train[self.target].tolist()[i],
                            "predicted": y_train_pred[i],
                        }
                        for i in range(len(X_train))
                    ],
                    "test": [
                        {
                            "true": y_test[self.target].tolist()[i],
                            "predicted": y_test_pred[i],
                        }
                        for i in range(len(X_test))
                    ],
                },
                "model": model,
                "selected_features": (
                    X.columns.tolist()
                    if not self.feature_selection_autoselect
                    else X.columns[selected_indices].tolist()
                ),
                "hyperparameters": (
                    best_params
                    if self.enable_parameter_tune
                    else MODEL_PARAM_DEFAULT[model_name]
                ),
            }
            # populating goal_metric_per_folds builds the box and whisker plot representing cross validation scores
            goal_metric_per_folds["train"].append(
                fold_metrics[f"train_{self.goal_metric}"]
            )
            goal_metric_per_folds["test"].append(
                fold_metrics[f"test_{self.goal_metric}"]
            )
            # Store test predictions
            metrics.append(fold_metrics)
            if self.goal_metric in ["mae", "mse", "rmse"]:
                # For error metrics, lower is better
                best_fold_idx = min(
                    range(len(metrics)),
                    key=lambda i: metrics[i][f"test_{self.goal_metric}"],
                )
            else:  # 'r2' or 'adj_r2'
                # For R² metrics, higher is better
                best_fold_idx = max(
                    range(len(metrics)),
                    key=lambda i: metrics[i][f"test_{self.goal_metric}"],
                )
            ###
            ### writing metrics / hyperparameter / features per fold
            # Create JSON-serializable metrics for saving
            json_serializable_metrics = {
                "iteration": iteration + 1,
                "metrics": {
                    "train": {
                        "mae": fold_metrics["train_mae"],
                        "mse": fold_metrics["train_mse"],
                        "r2": fold_metrics["train_r2"],
                        "adj_r2": fold_metrics["train_adj_r2"]
                    },
                    "test": {
                        "mae": fold_metrics["test_mae"],
                        "mse": fold_metrics["test_mse"],
                        "r2": fold_metrics["test_r2"],
                        "adj_r2": fold_metrics["test_adj_r2"]
                    }
                },
                "selected_features": fold_metrics["selected_features"],
                "hyperparameters": fold_metrics["hyperparameters"]  # Convert to string to ensure JSON compatibility
            }
            ### of iteration metrics
            # Save JSON file
            with open(os.path.join(iteration_dir, f"iteration_{iteration+1}_scores.json"), "w") as f:
                json.dump(json_serializable_metrics, f, indent=4)
            ###
        ###
        # save best fold metrics
        # Get the best iteration number (1-based)
        best_iteration = best_fold_idx + 1

        # Source directory path
        best_iteration_dir = os.path.join(kfolds_dir, f"iteration_{best_iteration}")
        # Destination directory path at top level
        destination_dir = "best_fold"

        # Copy the entire directory if it exists
        if os.path.exists(best_iteration_dir):
            # Remove destination directory if it already exists
            if os.path.exists(destination_dir):
                shutil.rmtree(destination_dir)
            
            # Copy the entire directory
            shutil.copytree(best_iteration_dir, destination_dir)
        
        ###
        return metrics[best_fold_idx], goal_metric_per_folds


class RegressionInference:
    def __init__(self, input_df, scaler, model, target, target_provided=False):
        self.input = input_df
        self.scaler = scaler
        self.model = model
        self.target = target
        self.target_provided = target_provided

    def normalize(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def inference(self):
        X = self.input.select_dtypes(include=["number"]).fillna(0)
        X = X[[_ for _ in X.columns if _ != self.target]]
        X = pd.DataFrame(self.normalize(X), columns=X.columns)
        print(X.columns)
        if self.target_provided:
            y_test = self.input[[self.target]] if self.target_provided else None
            y_test_pred = [float(_) for _ in list(self.model.predict(X))]
            test_metrics = {
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "test_adj_r2": adj_r2(
                    r2_score(y_test, y_test_pred), len(y_test), len(X.columns)
                ),
                "test_r2": r2_score(y_test, y_test_pred),
                "test_true_predicted": [
                        {
                            "true": y_test[self.target].tolist()[i],
                            "predicted": y_test_pred[i],
                        }
                        for i in range(len(X))
                    ],
                }
            ###
            #build the test dir
            os.makedirs("test", exist_ok=True)
            #SAVE JSON SCORES
            with open(os.path.join("test", f"test_scores.json"), "w") as f:
                json.dump(test_metrics, f, indent=4)
            ###
            return {"predictions": y_test_pred, "test_metrics": test_metrics}
        else:
            return {"predictions": [float(_) for _ in list(self.model.predict(X))]}


# Builder pattern for clean method chaining interface--very testable
class RegressionBuilder:
    def __init__(
        self,
        input_training_df,
        input_inference_df,
        target,
        hyperparameters_tune,
        feature_selection_autoselect,
        model_choice,
        goal_metric,
        data_augmentation,
        feature_selection_num,
        saved_model,
        saved_scaler,
        data_augmentation_method = None,
        data_augmentation_sectioning = None,
        data_augmentation_region_num = 10,
        data_augmentation_min_samples_per_region = 5,
        balance_strategy = "equal",
        k = 10,
        target_provided = False,
        sampling_method = "kfold"
    ):
        self.input_training = input_training_df
        self.regression_training_file_validator = (
            RegressionFileValidator(input_training_df.columns.tolist())
            if input_training_df is not None
            else None
        )
        self.input_inference = input_inference_df
        self.regression_inference_file_validator = (
            RegressionFileValidator(input_inference_df.columns.tolist())
            if input_inference_df is not None
            else None
        )
        self.target = target
        self.hyperparameters_tune = hyperparameters_tune
        self.feature_selection_autoselect = feature_selection_autoselect
        self.model_choice = model_choice
        self.goal_metric = goal_metric
        self.data_augmentation = data_augmentation
        self.feature_selection_num = feature_selection_num
        self.saved_model = saved_model
        self.saved_scaler = saved_scaler
        self.data_augmentation_method = data_augmentation_method
        self.data_augmentation_sectioning = data_augmentation_sectioning
        self.data_augmentation_region_num = data_augmentation_region_num
        self.data_augmentation_min_samples_per_region = data_augmentation_min_samples_per_region
        self.balance_strategy = balance_strategy
        self.k = k
        self.target_provided = target_provided
        self.sampling_method = sampling_method


    def populate_model_dict(self):
        model_dict = {}
        model_dict["linear"] = LinearRegression
        model_dict["bayesian"] = BayesianRidge
        model_dict["elasticnet"] = ElasticNet
        model_dict["xgboost"] = xgb.XGBRegressor
        model_dict["randomforest"] = RandomForestRegressor
        model_dict["svm"] = SVR
        #model_dict["tabpfn"] = TabPFNRegressor
        if isinstance(self.model_choice, list):
            model_dict = {
                k: {"model": model_dict[k]}
                for k in self.model_choice
                if k in model_dict
            }
        return model_dict

    def prepare_model_dict_for_training(self, model_dict):
        X = (
            self.input_training.drop(columns=[self.target])
            .select_dtypes(include=["number"])
            .fillna(0)
        )
        feature_dict = {k: X.columns for k in model_dict.keys()}
        return {
            model_name: {
                "model": model_dict[model_name],
                "feature_analysis": list(feature_dict.get(model_name, [])),
            }
            for model_name in model_dict
        }

    def build_model(self):
        model_dict = self.populate_model_dict()
        regression_train = RegressionTrain(
            self.input_training,
            self.target,
            self.hyperparameters_tune,
            self.data_augmentation,
            self.feature_selection_autoselect,
            self.feature_selection_num,
            self.goal_metric,
            self.data_augmentation_method,
            self.data_augmentation_sectioning,
            self.data_augmentation_region_num,
            self.data_augmentation_min_samples_per_region,
            self.balance_strategy,
            self.k,
            self.sampling_method

        )
        optimized_model = regression_train.find_best_model(model_dict)
        self.scaler = optimized_model["scaler"]
        self.model = optimized_model["model"]
        self.model_name = optimized_model["model_name"]
        self.metrics = optimized_model["metrics"]
        self.features = optimized_model["selected_features"]
        self.goal_metric_per_folds = optimized_model["goal_metric_per_folds"]
        self.true_prediction_point = optimized_model["test_true_predicted"]

    def run_inference(self):
        regression_inference = RegressionInference(
            self.input_inference,
            self.saved_scaler,
            self.saved_model,
            self.target,
            self.target_provided,
        )
        return regression_inference.inference()

    def run_regression(self):
        if self.input_training is not None:
            validation_status_training = (
                self.regression_training_file_validator.run_validation(
                    self.input_training
                )
            )
            if not isinstance(validation_status_training, pd.DataFrame):
                return {"error": f"{validation_status_training}"}
            self.build_model()
            return {
                "metrics": self.metrics,
                "model_name": self.model_name,
                "model": self.model,
                "hyperparameters": [
                    {"Hyperparameter Name": key, "Hyperparameter Value": value}
                    for key, value in self.model.get_params().items()
                    if key not in MODEL_HYPERPARAMTERS_TO_NOT_DISPLAY[self.model_name]
                ],
                "scaler": self.scaler,
                "features": self.features,
                "goal_metric_per_folds": self.goal_metric_per_folds,
                "true_prediction_point": self.true_prediction_point,
            }
        else:
            validation_status_inference = (
                self.regression_inference_file_validator.run_validation(
                    self.input_inference
                )
            )
            if not isinstance(validation_status_inference, pd.DataFrame):
                return {"error": f"{validation_status_inference}"}
            return self.run_inference()
