"""
Molecular Stability Validator for MMLI Dataset
===============================================
Validates synthetic molecular samples for chemical, physical, and statistical stability.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings("ignore")


class MolecularStabilityValidator:
    """
    Validates synthetic molecular samples for stability across multiple dimensions:
    - Physical stability: Feature values within observed physical ranges
    - Chemical stability: Correlation patterns maintained, no impossible combinations
    - Statistical stability: Samples follow learned distribution
    """
    
    def __init__(self, original_data, target_column, tolerance=0.1, outlier_contamination=0.1):
        """
        Initialize the validator with original data
        
        Parameters:
        -----------
        original_data : pandas DataFrame
            Original dataset with features and target
        target_column : str
            Name of the target column
        tolerance : float
            Tolerance for range extension (0.1 = 10% beyond observed range)
        outlier_contamination : float
            Expected proportion of outliers for outlier detection methods
        """
        self.target_column = target_column
        self.tolerance = tolerance
        self.outlier_contamination = outlier_contamination
        
        # Separate features and target
        self.X_original = original_data.drop(columns=[target_column])
        self.y_original = original_data[target_column]
        
        # Store feature names
        self.feature_names = self.X_original.columns.tolist()
        
        # Compute statistics from original data
        self._compute_original_statistics()
        
        # Fit outlier detectors
        self._fit_outlier_detectors()
    
    def _compute_original_statistics(self):
        """Compute statistics from original data for validation"""
        # Feature ranges (with tolerance)
        self.feature_mins = self.X_original.min()
        self.feature_maxs = self.X_original.max()
        self.feature_ranges = self.feature_maxs - self.feature_mins
        
        # Extended ranges with tolerance
        self.extended_mins = self.feature_mins - self.tolerance * self.feature_ranges
        self.extended_maxs = self.feature_maxs + self.tolerance * self.feature_ranges
        
        # Feature means and stds
        self.feature_means = self.X_original.mean()
        self.feature_stds = self.X_original.std()
        
        # Correlation matrix
        self.correlation_matrix = self.X_original.corr()
        
        # Target range
        self.target_min = self.y_original.min()
        self.target_max = self.y_original.max()
        self.target_range = self.target_max - self.target_min
        self.extended_target_min = self.target_min - self.tolerance * self.target_range
        self.extended_target_max = self.target_max + self.tolerance * self.target_range
        
        # Percentile ranges for robust validation
        self.feature_q01 = self.X_original.quantile(0.01)
        self.feature_q99 = self.X_original.quantile(0.99)
        
        # Identify high-correlation pairs (for chemical consistency)
        self.high_corr_pairs = []
        corr_array = self.correlation_matrix.values
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                if abs(corr_array[i, j]) > 0.8:  # High correlation threshold
                    self.high_corr_pairs.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        corr_array[i, j]
                    ))
    
    def _fit_outlier_detectors(self):
        """Fit outlier detection models on original data"""
        X_values = self.X_original.values
        
        # Local Outlier Factor
        n_neighbors = min(5, len(X_values) - 1)
        if n_neighbors >= 1:
            self.lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.outlier_contamination,
                novelty=True
            )
            self.lof.fit(X_values)
            self.lof_available = True
        else:
            self.lof_available = False
        
        # Isolation Forest
        self.iso_forest = IsolationForest(
            contamination=self.outlier_contamination,
            random_state=13
        )
        self.iso_forest.fit(X_values)
        
        # Elliptic Envelope (for Gaussian-like data)
        try:
            self.elliptic = EllipticEnvelope(
                contamination=self.outlier_contamination,
                random_state=13
            )
            self.elliptic.fit(X_values)
            self.elliptic_available = True
        except:
            self.elliptic_available = False
    
    def validate_physical_stability(self, X_synthetic, y_synthetic):
        """
        Validate physical stability: features within plausible physical ranges
        
        Returns:
        --------
        results : dict
            Validation results with pass/fail and details
        """
        results = {
            'test_name': 'Physical Stability',
            'description': 'Features within observed physical ranges',
            'passed': [],
            'failed': [],
            'details': {}
        }
        
        n_samples = len(X_synthetic)
        
        # Convert to numpy if needed
        if isinstance(X_synthetic, pd.DataFrame):
            X_vals = X_synthetic.values
        else:
            X_vals = X_synthetic
            
        if isinstance(y_synthetic, pd.Series):
            y_vals = y_synthetic.values
        else:
            y_vals = y_synthetic
        
        # Check each sample
        for i in range(n_samples):
            sample_valid = True
            violations = []
            
            for j, feat in enumerate(self.feature_names):
                val = X_vals[i, j]
                
                # Check extended range
                if val < self.extended_mins[feat] or val > self.extended_maxs[feat]:
                    sample_valid = False
                    violations.append(f"{feat}: {val:.4f} (range: {self.extended_mins[feat]:.4f} - {self.extended_maxs[feat]:.4f})")
            
            # Check target range
            target_val = y_vals[i]
            if target_val < self.extended_target_min or target_val > self.extended_target_max:
                sample_valid = False
                violations.append(f"target: {target_val:.4f} (range: {self.extended_target_min:.4f} - {self.extended_target_max:.4f})")
            
            if sample_valid:
                results['passed'].append(i)
            else:
                results['failed'].append(i)
                results['details'][i] = violations
        
        results['pass_rate'] = len(results['passed']) / n_samples if n_samples > 0 else 0
        return results
    
    def validate_chemical_stability(self, X_synthetic, y_synthetic):
        """
        Validate chemical stability: correlation patterns maintained
        
        Returns:
        --------
        results : dict
            Validation results with pass/fail and details
        """
        results = {
            'test_name': 'Chemical Stability',
            'description': 'Correlation patterns and chemical relationships maintained',
            'passed': [],
            'failed': [],
            'details': {}
        }
        
        n_samples = len(X_synthetic)
        
        if len(self.high_corr_pairs) == 0:
            # No high correlations to check, all samples pass
            results['passed'] = list(range(n_samples))
            results['pass_rate'] = 1.0
            results['details']['note'] = 'No high-correlation pairs to validate'
            return results
        
        # Convert to numpy if needed
        if isinstance(X_synthetic, pd.DataFrame):
            X_vals = X_synthetic.values
        else:
            X_vals = X_synthetic
        
        # For each sample, check if highly correlated features maintain their relationship
        for i in range(n_samples):
            sample_valid = True
            violations = []
            
            for feat1, feat2, original_corr in self.high_corr_pairs:
                # Get values
                idx1 = self.feature_names.index(feat1)
                idx2 = self.feature_names.index(feat2)
                val1 = X_vals[i, idx1]
                val2 = X_vals[i, idx2]
                
                # Normalize values
                std1 = self.feature_stds[feat1] if self.feature_stds[feat1] > 0 else 1
                std2 = self.feature_stds[feat2] if self.feature_stds[feat2] > 0 else 1
                norm1 = (val1 - self.feature_means[feat1]) / std1
                norm2 = (val2 - self.feature_means[feat2]) / std2
                
                # Check if normalized values have consistent sign relationship
                if original_corr > 0:
                    if norm1 * norm2 < -2:  # Significant opposite deviation
                        sample_valid = False
                        violations.append(f"{feat1}-{feat2}: inconsistent positive correlation")
                else:
                    if norm1 * norm2 > 2:  # Significant same deviation
                        sample_valid = False
                        violations.append(f"{feat1}-{feat2}: inconsistent negative correlation")
            
            if sample_valid:
                results['passed'].append(i)
            else:
                results['failed'].append(i)
                results['details'][i] = violations
        
        results['pass_rate'] = len(results['passed']) / n_samples if n_samples > 0 else 0
        return results
    
    def validate_statistical_stability(self, X_synthetic, y_synthetic):
        """
        Validate statistical stability: samples follow learned distribution
        Uses ensemble of outlier detection methods
        
        Returns:
        --------
        results : dict
            Validation results with pass/fail and details
        """
        results = {
            'test_name': 'Statistical Stability',
            'description': 'Samples follow learned data distribution',
            'passed': [],
            'failed': [],
            'details': {}
        }
        
        # Convert to numpy if needed
        if isinstance(X_synthetic, pd.DataFrame):
            X_values = X_synthetic.values
        else:
            X_values = X_synthetic
            
        n_samples = len(X_values)
        
        # Get predictions from each detector
        if self.lof_available:
            lof_pred = self.lof.predict(X_values)  # 1 = inlier, -1 = outlier
        else:
            lof_pred = np.ones(n_samples)
            
        iso_pred = self.iso_forest.predict(X_values)
        
        if self.elliptic_available:
            try:
                ellip_pred = self.elliptic.predict(X_values)
            except:
                ellip_pred = np.ones(n_samples)
        else:
            ellip_pred = np.ones(n_samples)
        
        # Ensemble: sample is valid if majority of detectors say inlier
        for i in range(n_samples):
            votes = [lof_pred[i], iso_pred[i], ellip_pred[i]]
            inlier_votes = sum(1 for v in votes if v == 1)
            
            if inlier_votes >= 2:  # Majority vote
                results['passed'].append(i)
            else:
                results['failed'].append(i)
                results['details'][i] = f"Outlier votes: LOF={lof_pred[i]}, IsoForest={iso_pred[i]}, Elliptic={ellip_pred[i]}"
        
        results['pass_rate'] = len(results['passed']) / n_samples if n_samples > 0 else 0
        return results
    
    def validate_all(self, X_synthetic, y_synthetic):
        """
        Run all validation tests
        
        Parameters:
        -----------
        X_synthetic : pandas DataFrame or numpy array
            Synthetic feature data
        y_synthetic : pandas Series or numpy array
            Synthetic target values
            
        Returns:
        --------
        results : dict
            Comprehensive validation results
        """
        physical = self.validate_physical_stability(X_synthetic, y_synthetic)
        chemical = self.validate_chemical_stability(X_synthetic, y_synthetic)
        statistical = self.validate_statistical_stability(X_synthetic, y_synthetic)
        
        n_samples = len(X_synthetic)
        
        # Combine results - sample is stable only if passes all tests
        stable_samples = set(physical['passed']) & set(chemical['passed']) & set(statistical['passed'])
        unstable_samples = set(range(n_samples)) - stable_samples
        
        overall_results = {
            'n_total': n_samples,
            'n_stable': len(stable_samples),
            'n_unstable': len(unstable_samples),
            'stability_rate': len(stable_samples) / n_samples if n_samples > 0 else 0,
            'stable_indices': list(stable_samples),
            'unstable_indices': list(unstable_samples),
            'physical_stability': physical,
            'chemical_stability': chemical,
            'statistical_stability': statistical,
            'summary': {
                'physical_pass_rate': physical['pass_rate'],
                'chemical_pass_rate': chemical['pass_rate'],
                'statistical_pass_rate': statistical['pass_rate'],
                'overall_pass_rate': len(stable_samples) / n_samples if n_samples > 0 else 0
            }
        }
        
        return overall_results
    
    def get_stable_samples(self, X_synthetic, y_synthetic, validation_results=None):
        """
        Get only the stable synthetic samples
        
        Parameters:
        -----------
        X_synthetic : pandas DataFrame or numpy array
            Synthetic feature data
        y_synthetic : pandas Series or numpy array
            Synthetic target values
        validation_results : dict, optional
            Pre-computed validation results
            
        Returns:
        --------
        X_stable : same type as X_synthetic
            Only stable samples
        y_stable : same type as y_synthetic
            Only stable targets
        """
        if validation_results is None:
            validation_results = self.validate_all(X_synthetic, y_synthetic)
        
        stable_indices = validation_results['stable_indices']
        
        if len(stable_indices) == 0:
            # Return empty with same type
            if isinstance(X_synthetic, pd.DataFrame):
                X_stable = X_synthetic.iloc[[]].reset_index(drop=True)
            else:
                X_stable = np.array([]).reshape(0, X_synthetic.shape[1])
            
            if isinstance(y_synthetic, pd.Series):
                y_stable = y_synthetic.iloc[[]].reset_index(drop=True)
            else:
                y_stable = np.array([])
            
            return X_stable, y_stable
        
        if isinstance(X_synthetic, pd.DataFrame):
            X_stable = X_synthetic.iloc[stable_indices].reset_index(drop=True)
        else:
            X_stable = X_synthetic[stable_indices]
        
        if isinstance(y_synthetic, pd.Series):
            y_stable = y_synthetic.iloc[stable_indices].reset_index(drop=True)
        else:
            y_stable = y_synthetic[stable_indices]
        
        return X_stable, y_stable


def print_validation_summary(results):
    """Print a formatted summary of validation results"""
    print("\n" + "="*60)
    print("MOLECULAR STABILITY VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal Synthetic Samples: {results['n_total']}")
    print(f"Stable Samples: {results['n_stable']} ({results['stability_rate']*100:.1f}%)")
    print(f"Unstable Samples: {results['n_unstable']} ({(1-results['stability_rate'])*100:.1f}%)")
    
    print("\n--- Individual Test Results ---")
    print(f"Physical Stability Pass Rate:    {results['summary']['physical_pass_rate']*100:.1f}%")
    print(f"Chemical Stability Pass Rate:    {results['summary']['chemical_pass_rate']*100:.1f}%")
    print(f"Statistical Stability Pass Rate: {results['summary']['statistical_pass_rate']*100:.1f}%")
    
    print("\n" + "="*60)
