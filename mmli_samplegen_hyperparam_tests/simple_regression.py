"""
Simplified Regression Framework for MMLI Experiment
Focused on k-fold cross-validation with optional data augmentation
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


# Default model parameters
MODEL_PARAM_DEFAULT = {
    "linear": {"n_jobs": None},
    "bayesian": {"verbose": False},
    "elasticnet": {"random_state": 13, "alpha": 1.0, "l1_ratio": 0.5},
    "xgboost": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 13},
    "randomforest": {"random_state": 13, "n_estimators": 100, "max_depth": 5, "max_features": "sqrt"},
    "svm": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
}


def adj_r2(r2, n, p):
    """Calculate adjusted R-squared"""
    if n <= p + 1:
        return float("nan")
    adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return min(adj, r2)


def _train_single_fold(fold_data):
    """
    Train a single fold - helper function for parallel execution
    Must be at module level for pickling
    """
    fold_idx, train_idx, test_idx, X_original, y_original, X_augmented, y_augmented, \
        is_synthetic, model_type, output_dir, using_augmentation = fold_data
    
    # Get test data (always from original)
    X_test = X_original.iloc[test_idx].values
    y_test = y_original.iloc[test_idx].values
    
    # Create scaler for this fold
    scaler = MinMaxScaler()
    
    if using_augmentation:
        # Fit scaler on training fold of original data
        X_train_orig = X_original.iloc[train_idx].values
        scaler.fit(X_train_orig)
        
        # Create mask for augmented data
        train_mask = is_synthetic.copy()
        for orig_idx in train_idx:
            train_mask[orig_idx] = True
        
        X_train = X_augmented.iloc[train_mask].values
        y_train = y_augmented.iloc[train_mask].values
    else:
        X_train = X_original.iloc[train_idx].values
        y_train = y_original.iloc[train_idx].values
        scaler.fit(X_train)
    
    # Normalize
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get model
    models = {
        'linear': LinearRegression,
        'bayesian': BayesianRidge,
        'elasticnet': ElasticNet,
        'xgboost': xgb.XGBRegressor,
        'randomforest': RandomForestRegressor,
        'svm': SVR
    }
    model = models[model_type](**MODEL_PARAM_DEFAULT[model_type])
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict on test
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adj_r2_val = adj_r2(r2, len(y_test), X_test.shape[1])
    
    fold_result = {
        'fold': fold_idx + 1,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'adj_r2': adj_r2_val,
        'predictions': y_pred.tolist(),
        'actuals': y_test.tolist()
    }
    
    # Save fold results if output_dir provided
    if output_dir:
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
            json.dump(fold_result, f, indent=2)
    
    return fold_result


class SimpleRegressor:
    """
    Simple regression wrapper for MMLI experiment
    Supports k-fold cross-validation with optional augmented data
    Supports parallel fold training via joblib
    """
    
    def __init__(self, model_type='randomforest', k_folds=5, random_state=13, n_jobs=-1):
        """
        Initialize regressor
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'linear', 'bayesian', 'elasticnet', 'xgboost', 'randomforest', 'svm'
        k_folds : int
            Number of folds for cross-validation
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs for k-fold CV (-1 = all cores, 1 = sequential)
        """
        self.model_type = model_type
        self.k_folds = k_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scaler = MinMaxScaler()
        
    def _get_model(self):
        """Get a new model instance"""
        models = {
            'linear': LinearRegression,
            'bayesian': BayesianRidge,
            'elasticnet': ElasticNet,
            'xgboost': xgb.XGBRegressor,
            'randomforest': RandomForestRegressor,
            'svm': SVR
        }
        return models[self.model_type](**MODEL_PARAM_DEFAULT[self.model_type])
    
    def _normalize(self, X, fit=False):
        """Normalize features"""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def train_with_augmentation(self, X_original, y_original, X_augmented=None, y_augmented=None, 
                                 is_synthetic=None, output_dir=None):
        """
        Train model using k-fold CV with optional parallel fold training
        
        When augmentation is provided:
        - Training: Uses augmented data (original + synthetic)
        - Evaluation: Only on original (organic) samples
        
        Parameters:
        -----------
        X_original : pandas DataFrame
            Original feature data
        y_original : pandas Series
            Original target values
        X_augmented : pandas DataFrame, optional
            Augmented feature data (original + synthetic)
        y_augmented : pandas Series, optional
            Augmented target values
        is_synthetic : numpy array, optional
            Boolean array indicating synthetic samples
        output_dir : str, optional
            Directory to save fold results
            
        Returns:
        --------
        results : dict
            Cross-validation results with metrics
        """
        # Determine if we're using augmentation
        using_augmentation = X_augmented is not None
        
        # If augmentation data provided, we need to handle it specially
        if using_augmentation:
            original_indices = np.where(~is_synthetic)[0]
            n_original = len(original_indices)
        else:
            n_original = len(X_original)
        
        # Set up k-fold on ORIGINAL data only
        kf = KFold(n_splits=min(self.k_folds, n_original), shuffle=True, random_state=self.random_state)
        
        # Prepare fold data for parallel execution
        fold_data_list = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_original)):
            fold_data = (
                fold_idx, train_idx, test_idx,
                X_original, y_original,
                X_augmented, y_augmented,
                is_synthetic, self.model_type, output_dir, using_augmentation
            )
            fold_data_list.append(fold_data)
        
        # Run folds in parallel if n_jobs != 1
        if self.n_jobs != 1 and len(fold_data_list) > 1:
            fold_metrics = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_train_single_fold)(fold_data) for fold_data in fold_data_list
            )
        else:
            # Sequential execution
            fold_metrics = [_train_single_fold(fold_data) for fold_data in fold_data_list]
        
        # Aggregate metrics
        avg_metrics = {
            'mean_mae': np.mean([f['mae'] for f in fold_metrics]),
            'std_mae': np.std([f['mae'] for f in fold_metrics]),
            'mean_mse': np.mean([f['mse'] for f in fold_metrics]),
            'std_mse': np.std([f['mse'] for f in fold_metrics]),
            'mean_rmse': np.mean([f['rmse'] for f in fold_metrics]),
            'std_rmse': np.std([f['rmse'] for f in fold_metrics]),
            'mean_r2': np.mean([f['r2'] for f in fold_metrics]),
            'std_r2': np.std([f['r2'] for f in fold_metrics]),
            'mean_adj_r2': np.mean([f['adj_r2'] for f in fold_metrics]),
            'std_adj_r2': np.std([f['adj_r2'] for f in fold_metrics])
        }
        
        results = {
            'model_type': self.model_type,
            'k_folds': len(fold_metrics),
            'using_augmentation': using_augmentation,
            'fold_metrics': fold_metrics,
            'aggregate_metrics': avg_metrics
        }
        
        return results


def train_baseline(X, y, target_column, model_type='randomforest', k_folds=5, output_dir=None):
    """
    Train baseline model (no augmentation)
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature data
    y : pandas Series
        Target values
    target_column : str
        Name of target column (for logging)
    model_type : str
        Type of model to use
    k_folds : int
        Number of CV folds
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    results : dict
        Training results with metrics
    """
    print(f"\n{'='*60}")
    print(f"Training BASELINE Model ({model_type})")
    print(f"{'='*60}")
    print(f"Data: {len(X)} samples, {len(X.columns)} features")
    print(f"Target: {target_column}")
    
    regressor = SimpleRegressor(model_type=model_type, k_folds=k_folds)
    results = regressor.train_with_augmentation(
        X_original=X,
        y_original=y,
        output_dir=output_dir
    )
    
    print(f"\nResults (Mean ± Std across {k_folds} folds):")
    print(f"  MAE:  {results['aggregate_metrics']['mean_mae']:.4f} ± {results['aggregate_metrics']['std_mae']:.4f}")
    print(f"  RMSE: {results['aggregate_metrics']['mean_rmse']:.4f} ± {results['aggregate_metrics']['std_rmse']:.4f}")
    print(f"  R²:   {results['aggregate_metrics']['mean_r2']:.4f} ± {results['aggregate_metrics']['std_r2']:.4f}")
    
    return results


def train_with_synthetic(X_original, y_original, X_augmented, y_augmented, is_synthetic,
                         target_column, model_type='randomforest', k_folds=5, output_dir=None):
    """
    Train model with augmented data
    
    Parameters:
    -----------
    X_original : pandas DataFrame
        Original feature data
    y_original : pandas Series
        Original target values  
    X_augmented : pandas DataFrame
        Augmented feature data (original + synthetic)
    y_augmented : pandas Series
        Augmented target values
    is_synthetic : numpy array
        Boolean array indicating synthetic samples
    target_column : str
        Name of target column
    model_type : str
        Type of model to use
    k_folds : int
        Number of CV folds
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    results : dict
        Training results with metrics
    """
    n_synthetic = is_synthetic.sum()
    n_original = len(X_original)
    
    print(f"\n{'='*60}")
    print(f"Training with AUGMENTED Data ({model_type})")
    print(f"{'='*60}")
    print(f"Original samples: {n_original}")
    print(f"Synthetic samples: {n_synthetic}")
    print(f"Total training samples: {n_original + n_synthetic}")
    print(f"Features: {len(X_original.columns)}")
    
    regressor = SimpleRegressor(model_type=model_type, k_folds=k_folds)
    results = regressor.train_with_augmentation(
        X_original=X_original,
        y_original=y_original,
        X_augmented=X_augmented,
        y_augmented=y_augmented,
        is_synthetic=is_synthetic,
        output_dir=output_dir
    )
    
    print(f"\nResults (Mean ± Std across {k_folds} folds):")
    print(f"  MAE:  {results['aggregate_metrics']['mean_mae']:.4f} ± {results['aggregate_metrics']['std_mae']:.4f}")
    print(f"  RMSE: {results['aggregate_metrics']['mean_rmse']:.4f} ± {results['aggregate_metrics']['std_rmse']:.4f}")
    print(f"  R²:   {results['aggregate_metrics']['mean_r2']:.4f} ± {results['aggregate_metrics']['std_r2']:.4f}")
    
    return results
