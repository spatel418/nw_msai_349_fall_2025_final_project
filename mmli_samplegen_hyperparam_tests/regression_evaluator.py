"""
Regression Evaluator Module
===========================
Evaluates regression models on molecular datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. XGBoost models will be skipped.")

from constants import MODEL_PARAM_DEFAULT


def adj_r2(r2, n, p):
    """Calculate adjusted R-squared"""
    if n <= p + 1:
        return float("nan")
    adj_r2_value = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return min(adj_r2_value, r2)


class RegressionEvaluator:
    """
    Evaluates regression models on molecular datasets.
    """
    
    def __init__(self, target_column, test_size=0.3, random_state=13, n_folds=5):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        target_column : str
            Name of the target column
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
        n_folds : int
            Number of folds for cross-validation
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.n_folds = n_folds
        
        # Model registry
        self.models = {
            'linear': LinearRegression,
            'bayesian': BayesianRidge,
            'elasticnet': ElasticNet,
            'svm': SVR,
            'randomforest': RandomForestRegressor,
        }
        
        if XGB_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor
    
    def _prepare_data(self, df):
        """Prepare data for training"""
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Remove special columns
        cols_to_drop = [c for c in X.columns if c in ['target_region', 'target_bucket', 'is_organic']]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        
        # Fill NaN
        X = X.fillna(0)
        
        return X, y
    
    def _get_model(self, model_name):
        """Get model instance with default parameters"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = self.models[model_name]
        params = MODEL_PARAM_DEFAULT.get(model_name, {})
        
        return model_class(**params)
    
    def evaluate(self, df, model_names=None):
        """
        Evaluate one or more models on the dataset.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Dataset to evaluate
        model_names : list or None
            List of model names to evaluate. If None, evaluate all.
        
        Returns:
        --------
        dict : Results for each model
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        # Prepare data
        X, y = self._prepare_data(df)
        
        if len(X) < 10:
            print(f"Warning: Very small dataset ({len(X)} samples)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                print(f"Skipping unknown model: {model_name}")
                continue
            
            try:
                # Get model
                model = self._get_model(model_name)
                
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                n_features = X_train.shape[1]
                
                metrics = {
                    'train': {
                        'mae': mean_absolute_error(y_train, y_train_pred),
                        'mse': mean_squared_error(y_train, y_train_pred),
                        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'r2': r2_score(y_train, y_train_pred),
                        'adj_r2': adj_r2(r2_score(y_train, y_train_pred), len(y_train), n_features),
                    },
                    'test': {
                        'mae': mean_absolute_error(y_test, y_test_pred),
                        'mse': mean_squared_error(y_test, y_test_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'r2': r2_score(y_test, y_test_pred),
                        'adj_r2': adj_r2(r2_score(y_test, y_test_pred), len(y_test), n_features),
                    },
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'n_features': n_features,
                }
                
                results[model_name] = metrics
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_cv(self, df, model_names=None):
        """
        Evaluate models using cross-validation.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Dataset to evaluate
        model_names : list or None
            List of model names to evaluate
        
        Returns:
        --------
        dict : CV results for each model
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        X, y = self._prepare_data(df)
        
        # Adjust n_folds if necessary
        n_folds = min(self.n_folds, len(X))
        if n_folds < 2:
            print("Not enough samples for cross-validation")
            return {}
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                continue
            
            fold_results = {
                'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'adj_r2': []
            }
            
            try:
                for train_idx, test_idx in kfold.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Scale
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train and predict
                    model = self._get_model(model_name)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Metrics
                    n_features = X_train.shape[1]
                    fold_results['mae'].append(mean_absolute_error(y_test, y_pred))
                    fold_results['mse'].append(mean_squared_error(y_test, y_pred))
                    fold_results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    fold_results['r2'].append(r2_score(y_test, y_pred))
                    fold_results['adj_r2'].append(adj_r2(r2_score(y_test, y_pred), len(y_test), n_features))
                
                # Aggregate
                results[model_name] = {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                    }
                    for metric, values in fold_results.items()
                }
                
            except Exception as e:
                print(f"Error in CV for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
