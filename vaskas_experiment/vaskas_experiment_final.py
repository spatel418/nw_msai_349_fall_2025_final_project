
import sys
# sys.path.append('/mnt/project')

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from vaskas_augmentation import augment_vaskas_with_validation, compare_distributions
from regression import smoter_augmentation
import matplotlib.pyplot as plt

def create_experiment_directory(base_dir='vaskas_experiments'):
    """Create directory for experiment results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / f'experiment_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def load_and_prepare_vaskas_data(filepath, target_col, test_size=0.2, random_state=42):
    """
    Load Vaskas dataset and split into train/test
    """
    print("\n" + "="*80)
    print("LOADING VASKAS DATASET")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Remove non-numeric columns if any
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"\nRemoving non-numeric columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("\n⚠️  Found missing values, dropping rows with NaN...")
        df = df.dropna()
        print(f"Dataset size after dropping NaN: {len(df)}")
    
    # Split into features and target
    if target_col not in df.columns:
        print(f"\n❌ Error: Target column '{target_col}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return None, None, None, None
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    print(f"\nTarget '{target_col}' statistics:")
    print(f"Train - Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}, Range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Test  - Mean: {y_test.mean():.3f}, Std: {y_test.std():.3f}, Range: [{y_test.min():.3f}, {y_test.max():.3f}]")
    
    return X_train, X_test, y_train, y_test

def create_stratified_bins(y_train, num_bins=20):
    """Create stratified bins for regional sampling"""
    bin_edges = np.percentile(y_train.values, np.linspace(0, 100, num_bins + 1))
    
    # Ensure unique bin edges
    if len(np.unique(bin_edges)) < len(bin_edges):
        print("⚠️  Duplicate bin edges detected, using equal-width bins instead")
        bin_edges = np.linspace(np.min(y_train.values), np.max(y_train.values), num_bins + 1)
    
    region_assignments = np.digitize(y_train.values, bin_edges[1:-1])
    
    return bin_edges, region_assignments

def train_and_evaluate_model(X_train, y_train, X_test, y_test, method_name):
    """
    Train Random Forest and evaluate
    """
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {method_name}")
    print(f"{'='*80}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (using reasonable defaults for 1948 samples)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\nTraining on {len(X_train)} samples...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on train set
    y_train_pred = model.predict(X_train_scaled)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results = {
        'method': method_name,
        'n_train': len(X_train),
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    
    print(f"\nTrain Metrics:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R²:  {train_r2:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²:  {test_r2:.4f}")
    
    return model, results, scaler

def run_baseline_experiment(X_train, y_train, X_test, y_test):
    """
    Experiment 1: No data augmentation (baseline)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE (No Augmentation)")
    print("="*80)
    
    model, results, scaler = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, 
        method_name="Baseline (No Augmentation)"
    )
    
    return results

def run_vaekde_experiment(X_train, y_train, X_test, y_test, target_col, exp_dir,
                          num_bins=20, min_samples_per_region=50, validate=True):
    """
    Experiment 2: VAE-KDE with chemical validation
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: VAE-KDE WITH VALIDATION")
    print("="*80)
    
    # Create bins
    bin_edges, region_assignments = create_stratified_bins(y_train, num_bins=num_bins)
    
    # Prepare data
    df_train = X_train.copy()
    df_train[target_col] = y_train.values
    
    # Augment with validation
    vaekde_dir = exp_dir / 'vaekde_augmentation'
    vaekde_dir.mkdir(exist_ok=True)
    
    df_augmented, validation_stats = augment_vaskas_with_validation(
        df_train=df_train,
        target_column=target_col,
        bin_edges=bin_edges,
        num_bins=num_bins,
        balance_strategy='increase',
        augmented_bucket_size=min_samples_per_region,
        region_assignments=region_assignments,
        save_dir=str(vaekde_dir),
        validate=validate,
        strict_validation=False
    )
    
    # Compare distributions
    compare_distributions(df_train, df_augmented, target_col, save_dir=str(vaekde_dir))
    
    # Separate features and target
    # Remove target_region column if it exists (it's just metadata, not a feature)
    columns_to_drop = [target_col]
    if 'target_region' in df_augmented.columns:
        columns_to_drop.append('target_region')
    
    X_train_aug = df_augmented.drop(columns=columns_to_drop)
    y_train_aug = df_augmented[target_col]
    
    # Train and evaluate
    model, results, scaler = train_and_evaluate_model(
        X_train_aug, y_train_aug, X_test, y_test,
        method_name="VAE-KDE (with validation)"
    )
    
    # Add validation stats to results
    results['validation_stats'] = validation_stats
    
    return results

def run_smote_experiment(X_train, y_train, X_test, y_test, target_col):
    """
    Experiment 3: SMOTE augmentation
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: SMOTE")
    print("="*80)
    
    # Prepare data
    df_train = X_train.copy()
    df_train[target_col] = y_train.values
    
    # Apply SMOTE with error handling
    print("\nApplying SMOTE augmentation...")
    try:
        df_augmented = smoter_augmentation(df_train, target_col)
    except Exception as e:
        print(f"\n⚠️  SMOTE augmentation failed with error: {str(e)}")
        print("This is a known issue with the smogn library on some datasets.")
        print("Skipping SMOTE experiment and continuing with available results...\n")
        return None
    
    # Separate features and target
    # Remove target_region column if it exists (it's just metadata, not a feature)
    columns_to_drop = [target_col]
    if 'target_region' in df_augmented.columns:
        columns_to_drop.append('target_region')
    
    X_train_aug = df_augmented.drop(columns=columns_to_drop)
    y_train_aug = df_augmented[target_col]
    
    print(f"After SMOTE: {len(df_augmented)} samples (added {len(df_augmented) - len(df_train)})")
    
    # Train and evaluate
    model, results, scaler = train_and_evaluate_model(
        X_train_aug, y_train_aug, X_test, y_test,
        method_name="SMOTE"
    )
    
    return results

def compare_results(all_results, exp_dir):
    """
    Compare all experimental results
    """
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df[['method', 'n_train', 'train_r2', 'test_r2', 
                                   'train_mae', 'test_mae', 'train_mse', 'test_mse']]
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_df.to_csv(exp_dir / 'results_comparison.csv', index=False)
    print(f"\n✓ Saved comparison table to {exp_dir / 'results_comparison.csv'}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    methods = comparison_df['method'].values
    x_pos = np.arange(len(methods))
    
    # Define colors flexibly
    colors = ['blue', 'red', 'green'][:len(methods)]
    
    # R² comparison
    axes[0].bar(x_pos, comparison_df['test_r2'], alpha=0.7, color=colors)
    axes[0].set_ylabel('Test R²')
    axes[0].set_title('R² Comparison (Higher is Better)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=comparison_df['test_r2'].iloc[0], color='blue', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].legend()
    
    # MAE comparison
    axes[1].bar(x_pos, comparison_df['test_mae'], alpha=0.7, color=colors)
    axes[1].set_ylabel('Test MAE')
    axes[1].set_title('MAE Comparison (Lower is Better)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=comparison_df['test_mae'].iloc[0], color='blue', linestyle='--', alpha=0.5, label='Baseline')
    axes[1].legend()
    
    # Training set size
    axes[2].bar(x_pos, comparison_df['n_train'], alpha=0.7, color=colors)
    axes[2].set_ylabel('Training Samples')
    axes[2].set_title('Training Set Size')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(methods, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'results_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {exp_dir / 'results_comparison.png'}")
    plt.close()
    
    # Determine winner
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    
    best_r2_idx = comparison_df['test_r2'].idxmax()
    best_method = comparison_df.loc[best_r2_idx, 'method']
    best_r2 = comparison_df.loc[best_r2_idx, 'test_r2']
    baseline_r2 = comparison_df.loc[0, 'test_r2']
    
    print(f"\n✓ Best method by Test R²: {best_method} (R² = {best_r2:.4f})")
    
    if best_r2_idx == 0:
        print("\n  Baseline performed best - data augmentation did NOT help!")
        print("  Possible reasons:")
        print("    - Dataset already well-balanced")
        print("    - Synthetic data introduced noise")
        print("    - Dataset too small/large for augmentation to matter")
    else:
        improvement = ((best_r2 - baseline_r2) / abs(baseline_r2)) * 100
        print(f"\n  {best_method} improved over baseline by {improvement:.1f}%!")
        print("  Data augmentation was SUCCESSFUL!")
    
    return comparison_df

def main(vaskas_path, target_col, num_bins=20, min_samples_per_region=50):
    """
    Main experiment pipeline
    """
    print("\n" + "="*80)
    print("VASKAS DATASET EXPERIMENT")
    print("VAE-KDE Data Augmentation Validation")
    print("="*80)
    
    # Create experiment directory
    exp_dir = create_experiment_directory()
    print(f"\nExperiment directory: {exp_dir}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_vaskas_data(
        vaskas_path, target_col, test_size=0.2, random_state=42
    )
    
    if X_train is None:
        return
    
    # Save configuration
    config = {
        'vaskas_path': vaskas_path,
        'target_col': target_col,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'num_features': X_train.shape[1],
        'num_bins': num_bins,
        'min_samples_per_region': min_samples_per_region,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run all experiments
    all_results = []
    
    # Experiment 1: Baseline
    baseline_results = run_baseline_experiment(X_train, y_train, X_test, y_test)
    all_results.append(baseline_results)
    
    # Experiment 2: VAE-KDE
    vaekde_results = run_vaekde_experiment(
        X_train, y_train, X_test, y_test, target_col, exp_dir,
        num_bins=num_bins, min_samples_per_region=min_samples_per_region, validate=True
    )
    all_results.append(vaekde_results)
    
    # Experiment 3: SMOTE
    smote_results = run_smote_experiment(X_train, y_train, X_test, y_test, target_col)
    if smote_results is not None:
        all_results.append(smote_results)
    else:
        print("\n⚠️  SMOTE experiment was skipped due to errors.")
        print("Continuing with Baseline vs VAE-KDE comparison only.\n")
    
    # Compare results
    comparison_df = compare_results(all_results, exp_dir)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {exp_dir}")
    print("\nGenerated files:")
    print(f"  - config.json: Experiment configuration")
    print(f"  - results_comparison.csv: Numerical results")
    print(f"  - results_comparison.png: Visual comparison")
    print(f"  - vaekde_augmentation/: VAE-KDE outputs and validation report")
    
    return exp_dir, comparison_df

if __name__ == "__main__":
    VASKAS_PATH = "vaskas.csv"
    TARGET_COL = "barrier" 
    
    # Augmentation parameters
    NUM_BINS = 20  # Number of regions for stratified sampling
    MIN_SAMPLES_PER_REGION = 100  # Target samples per region
    
    # Run experiment
    exp_dir, results = main(
        vaskas_path=VASKAS_PATH,
        target_col=TARGET_COL,
        num_bins=NUM_BINS,
        min_samples_per_region=MIN_SAMPLES_PER_REGION
    )
    
    print("\n✓ Done!")
