"""
MMLI Sample Size Scaling Experiment
====================================
This script runs experiments to generate varying amounts of synthetic samples
using the VAE-KDE approach, then evaluates model performance and synthetic
sample stability.

Sample sizes tested: 20, 60, 100, 500, 1000, 3000, 6000, 10000, 20000, 30000

Usage:
    python run_mmli_experiment.py

Requirements:
    - mmli.csv in the same directory
    - All dependent files in the same directory
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Import local modules
from vae_kde_augmenter import VAEKDEAugmenter, data_augment_vae_kde
from stability_validator import MolecularStabilityValidator
from regression_evaluator import RegressionEvaluator
from experiment_utils import (
    setup_experiment_directory,
    save_experiment_results,
    generate_summary_report,
    plot_results
)

# Configuration
SAMPLE_SIZES = [20, 60, 100, 500, 1000, 3000, 6000, 10000, 20000, 30000]
TARGET_COLUMN = "T80"
RANDOM_STATE = 13
TEST_SIZE = 0.3
NUM_REGIONS = 10
MODELS_TO_TEST = ["randomforest", "xgboost", "linear", "svm"]


def load_mmli_data(filepath="mmli.csv"):
    """Load and prepare MMLI dataset"""
    print(f"Loading MMLI dataset from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"MMLI dataset not found at {filepath}. "
            "Please ensure mmli.csv is in the current directory."
        )
    
    df = pd.read_csv(filepath)
    
    # Remove non-feature columns if present
    columns_to_drop = []
    for col in ["Batch_ID", "Smiles", "SMILES", "smiles", "Name", "name", "ID", "id"]:
        if col in df.columns:
            columns_to_drop.append(col)
    
    # Store SMILES for stability validation if present
    smiles_col = None
    for col in ["Smiles", "SMILES", "smiles"]:
        if col in df.columns:
            smiles_col = df[col].copy()
            break
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Target range: {df[TARGET_COLUMN].min():.2f} to {df[TARGET_COLUMN].max():.2f}")
    
    return df, smiles_col


def run_single_experiment(df_train, n_synthetic, experiment_dir, validator):
    """
    Run a single experiment with specified number of synthetic samples.
    """
    print(f"\n{'='*60}")
    print(f"Generating {n_synthetic} synthetic samples...")
    print(f"{'='*60}")
    
    start_time = time.time()
    results = {
        "n_synthetic_requested": n_synthetic,
        "n_synthetic_generated": 0,
        "n_stable": 0,
        "n_unstable": 0,
        "stability_rate": 0.0,
        "generation_time": 0.0,
        "metrics_all": {},
        "metrics_stable": {},
        "metrics_unstable": {},
        "stability_details": {}
    }
    
    # Separate features and target
    X = df_train.drop(columns=[TARGET_COLUMN])
    y = df_train[TARGET_COLUMN]
    
    # Calculate region edges for VAE-KDE
    target_values = y.values
    region_edges = np.percentile(target_values, np.linspace(0, 100, NUM_REGIONS + 1))
    if len(np.unique(region_edges)) < len(region_edges):
        region_edges = np.linspace(np.min(target_values), np.max(target_values), NUM_REGIONS + 1)
    region_assignments = np.digitize(target_values, region_edges[1:-1])
    
    # Calculate samples per region to achieve target
    n_original = len(df_train)
    total_target = n_original + n_synthetic
    samples_per_region = max(1, total_target // NUM_REGIONS)
    
    try:
        # Generate synthetic samples
        df_augmented = data_augment_vae_kde(
            df_train=df_train.copy(),
            target_column=TARGET_COLUMN,
            bin_edges=region_edges,
            num_bins=NUM_REGIONS,
            balance_strategy='increase',
            augmented_bucket_size=samples_per_region,
            region_assignments=region_assignments
        )
        
        generation_time = time.time() - start_time
        results["generation_time"] = generation_time
        
        # Identify synthetic samples
        n_generated = len(df_augmented) - n_original
        results["n_synthetic_generated"] = n_generated
        
        print(f"Generated {n_generated} synthetic samples in {generation_time:.2f}s")
        
        # Extract synthetic samples for validation
        synthetic_df = df_augmented.iloc[n_original:].copy()
        
        # Validate stability of synthetic samples
        print("\nValidating synthetic sample stability...")
        stability_results = validator.validate_batch(synthetic_df)
        
        results["n_stable"] = stability_results["n_stable"]
        results["n_unstable"] = stability_results["n_unstable"]
        results["stability_rate"] = stability_results["stability_rate"]
        results["stability_details"] = stability_results["details"]
        
        # Split augmented data by stability
        stable_mask = stability_results["stable_mask"]
        
        # Prepare datasets for evaluation
        df_all = df_augmented.copy()
        
        # Only stable synthetic + original
        df_stable = pd.concat([
            df_train,
            synthetic_df[stable_mask]
        ], ignore_index=True)
        
        # Only unstable synthetic + original
        df_unstable = pd.concat([
            df_train,
            synthetic_df[~stable_mask]
        ], ignore_index=True) if (~stable_mask).any() else df_train.copy()
        
        # Save datasets
        df_all.to_csv(os.path.join(experiment_dir, "augmented_all.csv"), index=False)
        df_stable.to_csv(os.path.join(experiment_dir, "augmented_stable.csv"), index=False)
        df_unstable.to_csv(os.path.join(experiment_dir, "augmented_unstable.csv"), index=False)
        
        # Evaluate models
        print("\nEvaluating models...")
        evaluator = RegressionEvaluator(
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Evaluate on all synthetic data
        print("  - Evaluating on all synthetic data...")
        results["metrics_all"] = evaluator.evaluate(df_all, MODELS_TO_TEST)
        
        # Evaluate on stable synthetic data only
        if results["n_stable"] > 0:
            print("  - Evaluating on stable synthetic data...")
            results["metrics_stable"] = evaluator.evaluate(df_stable, MODELS_TO_TEST)
        
        # Evaluate on unstable synthetic data only
        if results["n_unstable"] > 0:
            print("  - Evaluating on unstable synthetic data...")
            results["metrics_unstable"] = evaluator.evaluate(df_unstable, MODELS_TO_TEST)
        
    except Exception as e:
        print(f"Error during experiment: {str(e)}")
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results


def run_baseline_experiment(df_train, experiment_dir):
    """Run baseline experiment without augmentation"""
    print(f"\n{'='*60}")
    print("Running baseline experiment (no augmentation)...")
    print(f"{'='*60}")
    
    evaluator = RegressionEvaluator(
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    results = {
        "n_samples": len(df_train),
        "metrics": evaluator.evaluate(df_train, MODELS_TO_TEST)
    }
    
    return results


def main():
    """Main experiment runner"""
    print("="*60)
    print("MMLI Sample Size Scaling Experiment")
    print("="*60)
    
    # Setup
    base_dir = setup_experiment_directory("mmli_experiment_results")
    print(f"\nResults will be saved to: {base_dir}")
    
    # Load data
    df, smiles = load_mmli_data()
    
    # Initialize stability validator
    validator = MolecularStabilityValidator(
        feature_columns=df.drop(columns=[TARGET_COLUMN]).columns.tolist(),
        target_column=TARGET_COLUMN
    )
    
    # Fit validator on original data
    validator.fit(df)
    
    # Store all results
    all_results = {
        "config": {
            "sample_sizes": SAMPLE_SIZES,
            "target_column": TARGET_COLUMN,
            "test_size": TEST_SIZE,
            "num_regions": NUM_REGIONS,
            "models": MODELS_TO_TEST,
            "random_state": RANDOM_STATE
        },
        "baseline": None,
        "experiments": {}
    }
    
    # Run baseline
    baseline_dir = os.path.join(base_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    all_results["baseline"] = run_baseline_experiment(df, baseline_dir)
    
    # Save baseline results
    with open(os.path.join(baseline_dir, "results.json"), 'w') as f:
        json.dump(all_results["baseline"], f, indent=2, default=str)
    
    # Run experiments for each sample size
    for n_synthetic in SAMPLE_SIZES:
        exp_dir = os.path.join(base_dir, f"synthetic_{n_synthetic}")
        os.makedirs(exp_dir, exist_ok=True)
        
        exp_results = run_single_experiment(df, n_synthetic, exp_dir, validator)
        all_results["experiments"][n_synthetic] = exp_results
        
        # Save individual experiment results
        with open(os.path.join(exp_dir, "results.json"), 'w') as f:
            json.dump(exp_results, f, indent=2, default=str)
        
        print(f"\nExperiment {n_synthetic} complete:")
        print(f"  - Generated: {exp_results['n_synthetic_generated']}")
        print(f"  - Stable: {exp_results['n_stable']} ({exp_results['stability_rate']*100:.1f}%)")
    
    # Save all results
    save_experiment_results(all_results, base_dir)
    
    # Generate summary report
    generate_summary_report(all_results, base_dir)
    
    # Generate plots
    plot_results(all_results, base_dir)
    
    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"Results saved to: {base_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
