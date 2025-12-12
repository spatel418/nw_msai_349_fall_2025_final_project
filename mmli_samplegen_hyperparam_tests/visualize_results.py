"""
Visualization Script for MMLI Experiment Results
================================================

Creates plots showing how accuracy changes with number of synthetic samples.

Usage:
    python visualize_results.py --results_dir ./results/mmli_experiment_YYYYMMDD_HHMMSS
"""

import os
import json
import argparse
import numpy as np

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_results(results_dir):
    """Load experiment results from JSON file"""
    results_path = os.path.join(results_dir, 'full_results.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def create_accuracy_plot(results, output_path):
    """Create plot showing accuracy vs number of synthetic samples"""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot creation - matplotlib not available")
        return
    
    baseline = results['baseline']
    experiments = results['experiments']
    
    # Prepare data
    sample_sizes = sorted([int(k) for k in experiments.keys() if 'error' not in experiments[str(k)]])
    
    # MAE data
    mae_all = [experiments[str(n)]['accuracy_all_synthetic']['mean_mae'] for n in sample_sizes]
    mae_stable = []
    for n in sample_sizes:
        val = experiments[str(n)]['accuracy_stable_only']['mean_mae']
        mae_stable.append(val if val is not None else np.nan)
    
    # R² data
    r2_all = [experiments[str(n)]['accuracy_all_synthetic']['mean_r2'] for n in sample_sizes]
    r2_stable = []
    for n in sample_sizes:
        val = experiments[str(n)]['accuracy_stable_only']['mean_r2']
        r2_stable.append(val if val is not None else np.nan)
    
    # Stability rates
    stability_rates = [experiments[str(n)]['validation']['stability_rate'] * 100 for n in sample_sizes]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: MAE vs Sample Size
    ax1 = axes[0, 0]
    ax1.axhline(y=baseline['mean_mae'], color='gray', linestyle='--', label='Baseline', linewidth=2)
    ax1.plot(sample_sizes, mae_all, 'b-o', label='All Synthetic', linewidth=2, markersize=8)
    ax1.plot(sample_sizes, mae_stable, 'g-s', label='Stable Only', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_title('MAE vs Synthetic Sample Size', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² vs Sample Size
    ax2 = axes[0, 1]
    ax2.axhline(y=baseline['mean_r2'], color='gray', linestyle='--', label='Baseline', linewidth=2)
    ax2.plot(sample_sizes, r2_all, 'b-o', label='All Synthetic', linewidth=2, markersize=8)
    ax2.plot(sample_sizes, r2_stable, 'g-s', label='Stable Only', linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² vs Synthetic Sample Size', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stability Rate vs Sample Size
    ax3 = axes[1, 0]
    ax3.bar(range(len(sample_sizes)), stability_rates, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(sample_sizes)))
    ax3.set_xticklabels([str(n) for n in sample_sizes], rotation=45)
    ax3.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax3.set_ylabel('Stability Rate (%)', fontsize=12)
    ax3.set_title('Synthetic Sample Stability Rate', fontsize=14)
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Improvement over Baseline
    ax4 = axes[1, 1]
    baseline_r2 = baseline['mean_r2']
    improvement_all = [(r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0 for r2 in r2_all]
    improvement_stable = [(r2 - baseline_r2) / abs(baseline_r2) * 100 if (baseline_r2 != 0 and not np.isnan(r2)) else np.nan for r2 in r2_stable]
    
    x = np.arange(len(sample_sizes))
    width = 0.35
    ax4.bar(x - width/2, improvement_all, width, label='All Synthetic', color='steelblue', alpha=0.7)
    ax4.bar(x + width/2, improvement_stable, width, label='Stable Only', color='forestgreen', alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(n) for n in sample_sizes], rotation=45)
    ax4.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax4.set_ylabel('R² Improvement over Baseline (%)', fontsize=12)
    ax4.set_title('Performance Improvement over Baseline', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def create_stability_breakdown_plot(results, output_path):
    """Create plot showing stability breakdown by type"""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot creation - matplotlib not available")
        return
    
    experiments = results['experiments']
    sample_sizes = sorted([int(k) for k in experiments.keys() if 'error' not in experiments[str(k)]])
    
    physical_rates = [experiments[str(n)]['validation']['physical_pass_rate'] * 100 for n in sample_sizes]
    chemical_rates = [experiments[str(n)]['validation']['chemical_pass_rate'] * 100 for n in sample_sizes]
    statistical_rates = [experiments[str(n)]['validation']['statistical_pass_rate'] * 100 for n in sample_sizes]
    overall_rates = [experiments[str(n)]['validation']['stability_rate'] * 100 for n in sample_sizes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sample_sizes))
    width = 0.2
    
    ax.bar(x - 1.5*width, physical_rates, width, label='Physical', color='#FF6B6B', alpha=0.8)
    ax.bar(x - 0.5*width, chemical_rates, width, label='Chemical', color='#4ECDC4', alpha=0.8)
    ax.bar(x + 0.5*width, statistical_rates, width, label='Statistical', color='#45B7D1', alpha=0.8)
    ax.bar(x + 1.5*width, overall_rates, width, label='Overall', color='#2C3E50', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sample_sizes], rotation=45)
    ax.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Stability Validation Breakdown', fontsize=14)
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Stability plot saved to: {output_path}")


def print_summary_table(results):
    """Print formatted summary table"""
    baseline = results['baseline']
    experiments = results['experiments']
    
    sample_sizes = sorted([int(k) for k in experiments.keys() if 'error' not in experiments.get(str(k), {})])
    
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Config':<15} | {'N Samples':>10} | {'Stability':>10} | {'MAE':>12} | {'RMSE':>12} | {'R²':>10}")
    print("-"*85)
    
    # Baseline
    print(f"{'Baseline':<15} | {baseline['n_samples']:>10} | {'N/A':>10} | {baseline['mean_mae']:>12.4f} | {baseline['mean_rmse']:>12.4f} | {baseline['mean_r2']:>10.4f}")
    print("-"*85)
    
    # Experiments
    for n in sample_sizes:
        exp = experiments[str(n)]
        
        # All synthetic
        print(f"{'All ' + str(n):<15} | {n:>10} | {exp['validation']['stability_rate']*100:>9.1f}% | "
              f"{exp['accuracy_all_synthetic']['mean_mae']:>12.4f} | "
              f"{exp['accuracy_all_synthetic']['mean_rmse']:>12.4f} | "
              f"{exp['accuracy_all_synthetic']['mean_r2']:>10.4f}")
        
        # Stable only
        if exp['accuracy_stable_only']['mean_mae'] is not None:
            n_stable = exp['accuracy_stable_only']['n_stable_used']
            print(f"{'Stable ' + str(n):<15} | {n_stable:>10} | {'100.0%':>10} | "
                  f"{exp['accuracy_stable_only']['mean_mae']:>12.4f} | "
                  f"{exp['accuracy_stable_only']['mean_rmse']:>12.4f} | "
                  f"{exp['accuracy_stable_only']['mean_r2']:>10.4f}")
        
        print("-"*85)
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize MMLI experiment results')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to results directory')
    parser.add_argument('--output_prefix', type=str, default='mmli_results', help='Output filename prefix')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    # Print summary table
    print_summary_table(results)
    
    # Create plots
    if MATPLOTLIB_AVAILABLE:
        accuracy_plot_path = os.path.join(args.results_dir, f'{args.output_prefix}_accuracy.png')
        stability_plot_path = os.path.join(args.results_dir, f'{args.output_prefix}_stability.png')
        
        create_accuracy_plot(results, accuracy_plot_path)
        create_stability_breakdown_plot(results, stability_plot_path)
    else:
        print("\nNote: Install matplotlib to generate plots:")
        print("  pip install matplotlib")


if __name__ == '__main__':
    main()
