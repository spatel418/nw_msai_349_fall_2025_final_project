"""
VAE-KDE Data Augmentation with Chemical Validation
Modified version that includes chemical validity checks for Vaskas dataset
"""

import sys
sys.path.append('/mnt/project')

from data_augmentation_upsampling_vae_kde import DataAugmentationVAE, data_augment_vae
from chemical_validator import validate_synthetic_data
import numpy as np
import pandas as pd
import os

def augment_vaskas_with_validation(
    df_train,
    target_column,
    bin_edges,
    num_bins=20,
    balance_strategy='increase',
    augmented_bucket_size=None,
    region_assignments=None,
    save_dir=None,
    validate=True,
    strict_validation=False
):
    """
    Augment Vaskas dataset using VAE-KDE with chemical validation
    
    Parameters:
    -----------
    df_train : pandas DataFrame
        Training data with features and target
    target_column : str
        Name of the target column
    bin_edges : numpy array
        Region boundaries for stratified sampling
    num_bins : int
        Number of regions
    balance_strategy : str
        'equal', 'proportional', or 'increase'
    augmented_bucket_size : int
        Minimum samples per region for 'increase' strategy
    region_assignments : numpy array
        Pre-computed region assignments
    save_dir : str
        Directory to save outputs
    validate : bool
        Whether to apply chemical validation
    strict_validation : bool
        Use stricter validation criteria
        
    Returns:
    --------
    df_augmented : pandas DataFrame
        Augmented dataset with valid synthetic samples
    validation_stats : dict
        Statistics about validation process
    """
    
    print("\n" + "="*80)
    print("VAE-KDE DATA AUGMENTATION WITH CHEMICAL VALIDATION")
    print("="*80)
    
    # Step 1: Generate synthetic data using VAE-KDE
    print("\nStep 1: Generating synthetic data with VAE-KDE...")
    df_augmented_raw = data_augment_vae(
        df_train=df_train,
        target_column=target_column,
        bin_edges=bin_edges,
        num_bins=num_bins,
        balance_strategy=balance_strategy,
        augmented_bucket_size=augmented_bucket_size,
        region_assignments=region_assignments,
        save_dir=save_dir
    )
    
    # Separate original and synthetic data
    n_original = len(df_train)
    df_original = df_augmented_raw.iloc[:n_original].copy()
    df_synthetic_raw = df_augmented_raw.iloc[n_original:].copy()
    
    n_synthetic_generated = len(df_synthetic_raw)
    print(f"\nGenerated {n_synthetic_generated} synthetic samples")
    
    validation_stats = {
        'n_generated': n_synthetic_generated,
        'n_valid': n_synthetic_generated,  # Will update if validation applied
        'n_invalid': 0,
        'validity_rate': 1.0,
        'validation_applied': validate
    }
    
    # Step 2: Apply chemical validation if requested
    if validate and n_synthetic_generated > 0:
        print("\nStep 2: Applying chemical validation...")
        
        # Get feature names (excluding target)
        feature_cols = [col for col in df_synthetic_raw.columns if col != target_column]
        
        # Extract synthetic features
        X_synthetic_raw = df_synthetic_raw[feature_cols].values
        y_synthetic_raw = df_synthetic_raw[target_column].values
        
        # Validate
        X_synthetic_valid, valid_mask, val_stats = validate_synthetic_data(
            X_synthetic_raw,
            feature_names=feature_cols,
            strict=strict_validation,
            verbose=True
        )
        
        # Keep only valid synthetic samples
        df_synthetic_valid = df_synthetic_raw[valid_mask].copy()
        y_synthetic_valid = y_synthetic_raw[valid_mask]
        
        # Update stats
        validation_stats['n_valid'] = len(df_synthetic_valid)
        validation_stats['n_invalid'] = validation_stats['n_generated'] - validation_stats['n_valid']
        validation_stats['validity_rate'] = val_stats['validity_rate']
        validation_stats['validation_report'] = val_stats['report']
        
        print(f"\n✓ Validation complete:")
        print(f"  Generated: {validation_stats['n_generated']}")
        print(f"  Valid: {validation_stats['n_valid']} ({100*validation_stats['validity_rate']:.1f}%)")
        print(f"  Invalid: {validation_stats['n_invalid']} ({100*(1-validation_stats['validity_rate']):.1f}%)")
        
        # Combine original + valid synthetic
        df_augmented = pd.concat([df_original, df_synthetic_valid], ignore_index=True)
    else:
        print("\nStep 2: Skipping validation (disabled)")
        df_augmented = df_augmented_raw
    
    # Save validation report
    if save_dir and validate:
        report_path = os.path.join(save_dir, 'validation_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(validation_stats['validation_report']))
        print(f"\nSaved validation report to {report_path}")
    
    print(f"\nFinal augmented dataset size: {len(df_augmented)} samples")
    print(f"  Original: {len(df_original)}")
    print(f"  Synthetic (valid): {len(df_augmented) - len(df_original)}")
    
    return df_augmented, validation_stats


def compare_distributions(df_original, df_augmented, target_col, save_dir=None):
    """
    Compare distributions of original vs augmented data
    """
    import matplotlib.pyplot as plt
    
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON")
    print("="*80)
    
    n_original = len(df_original)
    df_synthetic = df_augmented.iloc[n_original:].copy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original distribution
    axes[0].hist(df_original[target_col], bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Original Data\n(n={len(df_original)})')
    axes[0].grid(True, alpha=0.3)
    
    # Synthetic distribution
    if len(df_synthetic) > 0:
        axes[1].hist(df_synthetic[target_col], bins=50, edgecolor='black', alpha=0.7, color='red')
        axes[1].set_xlabel(target_col)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Synthetic Data (Valid)\n(n={len(df_synthetic)})')
        axes[1].grid(True, alpha=0.3)
    
    # Combined distribution
    axes[2].hist(df_original[target_col], bins=50, edgecolor='black', alpha=0.5, color='blue', label='Original')
    if len(df_synthetic) > 0:
        axes[2].hist(df_synthetic[target_col], bins=50, edgecolor='black', alpha=0.5, color='red', label='Synthetic')
    axes[2].set_xlabel(target_col)
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Combined Distribution\n(n={len(df_augmented)})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plot_path = os.path.join(save_dir, 'distribution_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved distribution comparison to {plot_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    print(f"\nTarget '{target_col}' Statistics:")
    print(f"\nOriginal Data:")
    print(df_original[target_col].describe())
    
    if len(df_synthetic) > 0:
        print(f"\nSynthetic Data:")
        print(df_synthetic[target_col].describe())
    
    print(f"\nCombined Data:")
    print(df_augmented[target_col].describe())


if __name__ == "__main__":
    print("VAE-KDE Augmentation with Chemical Validation")
    print("="*80)
    print("\nThis module provides validated data augmentation for molecular datasets")
    print("\nKey features:")
    print("  - VAE-KDE based synthetic data generation")
    print("  - Chemical validity checking")
    print("  - Distribution comparison and visualization")
    print("\nUse this in vaskas_experiment.py to run full experiments")
