#from tabpfn.scripts.tabular_metrics import TabPFNRegressor
from tabpfn import TabPFNRegressor
from sklearn.inspection import permutation_importance

class TabPFNRegressorWithImportance(TabPFNRegressor):
    """Wrapper for TabPFNRegressor that adds feature_importances_ attribute using permutation importance"""
    
    def fit(self, X, y):
        # Call the original fit method
        super().fit(X, y)
        
        # Calculate permutation importance
        # This uses the training data to calculate feature importance
        perm_importance = permutation_importance(self, X, y, n_repeats=3, random_state=42)
        
        # Store the mean importance as the feature_importances_ attribute
        self.feature_importances_ = perm_importance.importances_mean
        
        return self