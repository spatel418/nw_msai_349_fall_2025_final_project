"""
MMLI Dataset VAE-KDE Augmentation Experiment
=============================================

This experiment generates varying numbers of synthetic samples (20 to 30K)
and measures:
1. Prediction accuracy (MAE, RMSE, R²) for baseline vs augmented models
2. Stability of synthetic molecules (physical, chemical, statistical)
3. Accuracy breakdown by stable vs unstable samples

Usage:
    python run_experiment.py --data_path mmli.csv --target T80

Requirements:
    pip install numpy pandas scikit-learn torch xgboost scipy
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# Local imports
from vae_kde_augmenter import VAEKDEAugmenter, create_augmenter_for_mmli
from stability_validator import MolecularStabilityValidator, print_validation_summary
from simple_regression import train_baseline, train_with_synthetic


# Sample sizes to test
SAMPLE_SIZES = [20, 60, 100, 500, 1000, 3000, 6000, 10000, 20000, 30000]


def load_data(data_path, target_column):
    """
    Load MMLI dataset
    
    Parameters:
    -----------
    data_path : str
        Path to mmli.csv
    target_column : str
        Name of target column (typically 'T80')
        
    Returns:
    --------
    df : pandas DataFrame
        Full dataset
    X : pandas DataFrame
        Features only
    y : pandas Series
        Target values
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    
    # Filter to numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_column not in numeric_cols:
        raise ValueError(f"Target column '{target_column}' not found in numeric columns")
    
    # Remove target from features
    feature_cols = [c for c in numeric_cols if c != target_column]
    
    X = df[feature_cols].fillna(0)
    y = df[target_column].fillna(df[target_column].mean())
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target range: {y.min():.2f} to {y.max():.2f}")
    
    return df, X, y


def run_single_experiment(X, y, target_column, n_synthetic, output_dir, model_type='randomforest', k_folds=5):
    """
    Run experiment with specific number of synthetic samples
    
    Parameters:
    -----------
    X : pandas DataFrame
        Original features
    y : pandas Series
        Original targets
    target_column : str
        Target column name
    n_synthetic : int
        Number of synthetic samples to generate
    output_dir : str
        Directory to save results
    model_type : str
        Type of regression model
    k_folds : int
        Number of CV folds
        
    Returns:
    --------
    results : dict
        Comprehensive experiment results
    """
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {n_synthetic} Synthetic Samples")
    print(f"{'#'*70}")
    
    experiment_dir = os.path.join(output_dir, f'n_synthetic_{n_synthetic}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 1. Generate synthetic samples using VAE-KDE
    print("\n1. Generating synthetic samples...")
    augmenter = create_augmenter_for_mmli(len(X))
    X_aug, y_aug, is_synthetic = augmenter.augment_dataset(X, y, n_synthetic)
    
    # 2. Validate stability of synthetic samples
    print("\n2. Validating synthetic sample stability...")
    
    # Prepare full original data for validator
    original_data = X.copy()
    original_data[target_column] = y
    
    validator = MolecularStabilityValidator(original_data, target_column)
    
    # Get only synthetic samples for validation
    X_synthetic = X_aug.iloc[~is_synthetic == False].iloc[len(X):].reset_index(drop=True)
    y_synthetic = y_aug.iloc[~is_synthetic == False].iloc[len(y):].reset_index(drop=True)
    
    # Handle case where we're getting synthetic samples from is_synthetic mask
    synthetic_mask = is_synthetic
    X_synthetic = X_aug[synthetic_mask].reset_index(drop=True)
    y_synthetic = y_aug[synthetic_mask].reset_index(drop=True)
    
    validation_results = validator.validate_all(X_synthetic, y_synthetic)
    print_validation_summary(validation_results)
    
    # Save validation results
    validation_summary = {
        'n_synthetic': n_synthetic,
        'n_stable': validation_results['n_stable'],
        'n_unstable': validation_results['n_unstable'],
        'stability_rate': validation_results['stability_rate'],
        'physical_pass_rate': validation_results['summary']['physical_pass_rate'],
        'chemical_pass_rate': validation_results['summary']['chemical_pass_rate'],
        'statistical_pass_rate': validation_results['summary']['statistical_pass_rate']
    }
    
    with open(os.path.join(experiment_dir, 'validation_results.json'), 'w') as f:
        json.dump(validation_summary, f, indent=2)
    
    # 3. Train with ALL synthetic samples
    print("\n3. Training with ALL synthetic samples...")
    results_all = train_with_synthetic(
        X_original=X,
        y_original=y,
        X_augmented=X_aug,
        y_augmented=y_aug,
        is_synthetic=is_synthetic,
        target_column=target_column,
        model_type=model_type,
        k_folds=k_folds,
        output_dir=os.path.join(experiment_dir, 'all_synthetic')
    )
    
    # 4. Train with STABLE synthetic samples only
    print("\n4. Training with STABLE synthetic samples only...")
    
    # Get stable samples
    X_stable, y_stable = validator.get_stable_samples(X_synthetic, y_synthetic, validation_results)
    
    if len(X_stable) > 0:
        # Create augmented dataset with only stable synthetic samples
        X_aug_stable = pd.concat([X.reset_index(drop=True), X_stable], ignore_index=True)
        y_aug_stable = pd.concat([y.reset_index(drop=True), y_stable], ignore_index=True)
        is_synthetic_stable = np.concatenate([
            np.zeros(len(X), dtype=bool),
            np.ones(len(X_stable), dtype=bool)
        ])
        
        results_stable = train_with_synthetic(
            X_original=X,
            y_original=y,
            X_augmented=X_aug_stable,
            y_augmented=y_aug_stable,
            is_synthetic=is_synthetic_stable,
            target_column=target_column,
            model_type=model_type,
            k_folds=k_folds,
            output_dir=os.path.join(experiment_dir, 'stable_synthetic')
        )
    else:
        print("  No stable samples available - skipping stable-only training")
        results_stable = None
    
    # 5. Train with UNSTABLE synthetic samples only (for comparison)
    print("\n5. Training with UNSTABLE synthetic samples only...")
    
    unstable_indices = validation_results['unstable_indices']
    if len(unstable_indices) > 0:
        X_unstable = X_synthetic.iloc[unstable_indices].reset_index(drop=True)
        y_unstable = y_synthetic.iloc[unstable_indices].reset_index(drop=True)
        
        X_aug_unstable = pd.concat([X.reset_index(drop=True), X_unstable], ignore_index=True)
        y_aug_unstable = pd.concat([y.reset_index(drop=True), y_unstable], ignore_index=True)
        is_synthetic_unstable = np.concatenate([
            np.zeros(len(X), dtype=bool),
            np.ones(len(X_unstable), dtype=bool)
        ])
        
        results_unstable = train_with_synthetic(
            X_original=X,
            y_original=y,
            X_augmented=X_aug_unstable,
            y_augmented=y_aug_unstable,
            is_synthetic=is_synthetic_unstable,
            target_column=target_column,
            model_type=model_type,
            k_folds=k_folds,
            output_dir=os.path.join(experiment_dir, 'unstable_synthetic')
        )
    else:
        print("  No unstable samples - skipping unstable-only training")
        results_unstable = None
    
    # Compile all results
    experiment_results = {
        'n_synthetic_requested': n_synthetic,
        'n_synthetic_generated': int(is_synthetic.sum()),
        'n_original': len(X),
        'validation': validation_summary,
        'accuracy_all_synthetic': {
            'mean_mae': results_all['aggregate_metrics']['mean_mae'],
            'std_mae': results_all['aggregate_metrics']['std_mae'],
            'mean_rmse': results_all['aggregate_metrics']['mean_rmse'],
            'std_rmse': results_all['aggregate_metrics']['std_rmse'],
            'mean_r2': results_all['aggregate_metrics']['mean_r2'],
            'std_r2': results_all['aggregate_metrics']['std_r2']
        },
        'accuracy_stable_only': {
            'n_stable_used': len(X_stable) if results_stable else 0,
            'mean_mae': results_stable['aggregate_metrics']['mean_mae'] if results_stable else None,
            'std_mae': results_stable['aggregate_metrics']['std_mae'] if results_stable else None,
            'mean_rmse': results_stable['aggregate_metrics']['mean_rmse'] if results_stable else None,
            'std_rmse': results_stable['aggregate_metrics']['std_rmse'] if results_stable else None,
            'mean_r2': results_stable['aggregate_metrics']['mean_r2'] if results_stable else None,
            'std_r2': results_stable['aggregate_metrics']['std_r2'] if results_stable else None
        },
        'accuracy_unstable_only': {
            'n_unstable_used': len(unstable_indices) if results_unstable else 0,
            'mean_mae': results_unstable['aggregate_metrics']['mean_mae'] if results_unstable else None,
            'std_mae': results_unstable['aggregate_metrics']['std_mae'] if results_unstable else None,
            'mean_rmse': results_unstable['aggregate_metrics']['mean_rmse'] if results_unstable else None,
            'std_rmse': results_unstable['aggregate_metrics']['std_rmse'] if results_unstable else None,
            'mean_r2': results_unstable['aggregate_metrics']['mean_r2'] if results_unstable else None,
            'std_r2': results_unstable['aggregate_metrics']['std_r2'] if results_unstable else None
        }
    }
    
    # Save experiment results
    with open(os.path.join(experiment_dir, 'experiment_results.json'), 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    return experiment_results


def run_full_experiment(data_path, target_column, output_dir, model_type='randomforest', 
                        k_folds=5, sample_sizes=None, n_workers=None):
    """
    Run complete experiment across all sample sizes with optional parallelism
    
    Parameters:
    -----------
    data_path : str
        Path to MMLI CSV file
    target_column : str
        Target column name
    output_dir : str
        Base output directory
    model_type : str
        Type of regression model
    k_folds : int
        Number of CV folds
    sample_sizes : list, optional
        List of synthetic sample sizes to test
    n_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
        
    Returns:
    --------
    all_results : dict
        Complete results for all experiments
    """
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(output_dir, f'mmli_experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    print("="*70)
    print("MMLI VAE-KDE DATA AUGMENTATION EXPERIMENT")
    print("="*70)
    print(f"Output directory: {experiment_dir}")
    print(f"Sample sizes to test: {sample_sizes}")
    print(f"Model type: {model_type}")
    print(f"K-folds: {k_folds}")
    print(f"Parallel workers: {n_workers}")
    
    # Load data
    df, X, y = load_data(data_path, target_column)
    
    # Run baseline first (sequential)
    print("\n" + "="*70)
    print("BASELINE (No Augmentation)")
    print("="*70)
    
    baseline_results = train_baseline(
        X=X,
        y=y,
        target_column=target_column,
        model_type=model_type,
        k_folds=k_folds,
        output_dir=os.path.join(experiment_dir, 'baseline')
    )
    
    # Save baseline results
    baseline_summary = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'mean_mae': baseline_results['aggregate_metrics']['mean_mae'],
        'std_mae': baseline_results['aggregate_metrics']['std_mae'],
        'mean_rmse': baseline_results['aggregate_metrics']['mean_rmse'],
        'std_rmse': baseline_results['aggregate_metrics']['std_rmse'],
        'mean_r2': baseline_results['aggregate_metrics']['mean_r2'],
        'std_r2': baseline_results['aggregate_metrics']['std_r2']
    }
    
    with open(os.path.join(experiment_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_summary, f, indent=2)
    
    # Run experiments for each sample size
    all_results = {
        'baseline': baseline_summary,
        'experiments': {}
    }
    
    # Run experiments in parallel if n_workers > 1
    if n_workers > 1 and len(sample_sizes) > 1:
        print(f"\n{'='*70}")
        print(f"Running {len(sample_sizes)} experiments in parallel with {n_workers} workers")
        print("="*70)
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_n = {}
            for n_synthetic in sample_sizes:
                future = executor.submit(
                    run_single_experiment,
                    X, y, target_column, n_synthetic,
                    experiment_dir, model_type, k_folds
                )
                future_to_n[future] = n_synthetic
            
            # Collect results as they complete
            for future in as_completed(future_to_n):
                n_synthetic = future_to_n[future]
                try:
                    results = future.result()
                    all_results['experiments'][n_synthetic] = results
                    print(f"✓ Completed experiment with {n_synthetic} synthetic samples")
                except Exception as e:
                    print(f"✗ Error with {n_synthetic} samples: {e}")
                    all_results['experiments'][n_synthetic] = {'error': str(e)}
    else:
        # Sequential execution
        for n_synthetic in sample_sizes:
            try:
                results = run_single_experiment(
                    X=X,
                    y=y,
                    target_column=target_column,
                    n_synthetic=n_synthetic,
                    output_dir=experiment_dir,
                    model_type=model_type,
                    k_folds=k_folds
                )
                all_results['experiments'][n_synthetic] = results
            except Exception as e:
                print(f"\nError with {n_synthetic} samples: {e}")
                all_results['experiments'][n_synthetic] = {'error': str(e)}
    
    # Generate summary table
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print(f"\n{'N Synthetic':>12} | {'Stability':>10} | {'MAE (All)':>12} | {'MAE (Stable)':>12} | {'R² (All)':>10} | {'R² (Stable)':>10}")
    print("-"*85)
    
    print(f"{'Baseline':>12} | {'N/A':>10} | {baseline_summary['mean_mae']:>12.4f} | {'N/A':>12} | {baseline_summary['mean_r2']:>10.4f} | {'N/A':>10}")
    
    for n_synthetic in sorted(sample_sizes):
        if n_synthetic in all_results['experiments'] and 'error' not in all_results['experiments'][n_synthetic]:
            exp = all_results['experiments'][n_synthetic]
            stability = f"{exp['validation']['stability_rate']*100:.1f}%"
            mae_all = f"{exp['accuracy_all_synthetic']['mean_mae']:.4f}"
            r2_all = f"{exp['accuracy_all_synthetic']['mean_r2']:.4f}"
            
            if exp['accuracy_stable_only']['mean_mae'] is not None:
                mae_stable = f"{exp['accuracy_stable_only']['mean_mae']:.4f}"
                r2_stable = f"{exp['accuracy_stable_only']['mean_r2']:.4f}"
            else:
                mae_stable = "N/A"
                r2_stable = "N/A"
            
            print(f"{n_synthetic:>12} | {stability:>10} | {mae_all:>12} | {mae_stable:>12} | {r2_all:>10} | {r2_stable:>10}")
    
    # Save final summary
    with open(os.path.join(experiment_dir, 'full_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {experiment_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='MMLI VAE-KDE Data Augmentation Experiment')
    parser.add_argument('--data_path', type=str, required=True, help='Path to MMLI CSV file')
    parser.add_argument('--target', type=str, default='T80', help='Target column name')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model', type=str, default='randomforest', 
                        choices=['linear', 'bayesian', 'elasticnet', 'xgboost', 'randomforest', 'svm'],
                        help='Model type')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--sample_sizes', type=str, default=None, 
                        help='Comma-separated list of sample sizes (e.g., "20,100,500")')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run experiments sequentially (no parallelism)')
    
    args = parser.parse_args()
    
    # Parse sample sizes if provided
    if args.sample_sizes:
        sample_sizes = [int(x.strip()) for x in args.sample_sizes.split(',')]
    else:
        sample_sizes = SAMPLE_SIZES
    
    # Determine workers
    n_workers = 1 if args.sequential else args.workers
    
    # Run experiment
    run_full_experiment(
        data_path=args.data_path,
        target_column=args.target,
        output_dir=args.output_dir,
        model_type=args.model,
        k_folds=args.k_folds,
        sample_sizes=sample_sizes,
        n_workers=n_workers
    )


if __name__ == '__main__':
    main()
