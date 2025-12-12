"""
MMLI Experiment Results Visualization
=====================================
Generates comprehensive plots to analyze VAE-KDE data augmentation results.

Usage:
    python plot_results.py --results_path results/mmli_experiment_XXXXXX/full_results.json
    
Or specify the directory:
    python plot_results.py --results_dir results/mmli_experiment_XXXXXX/
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_results(results_path=None, results_dir=None):
    """Load experiment results from JSON file"""
    if results_path:
        with open(results_path, 'r') as f:
            return json.load(f)
    elif results_dir:
        full_results_path = os.path.join(results_dir, 'full_results.json')
        with open(full_results_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Must provide either results_path or results_dir")


def extract_metrics(results):
    """Extract metrics into organized arrays for plotting"""
    baseline = results['baseline']
    experiments = results['experiments']
    
    # Sort sample sizes
    sample_sizes = sorted([int(k) for k in experiments.keys()])
    
    metrics = {
        'sample_sizes': sample_sizes,
        'baseline_mae': baseline['mean_mae'],
        'baseline_r2': baseline['mean_r2'],
        'baseline_rmse': baseline['mean_rmse'],
        
        # All synthetic
        'mae_all': [],
        'mae_all_std': [],
        'r2_all': [],
        'r2_all_std': [],
        'rmse_all': [],
        'rmse_all_std': [],
        
        # Stable only
        'mae_stable': [],
        'mae_stable_std': [],
        'r2_stable': [],
        'r2_stable_std': [],
        'rmse_stable': [],
        'rmse_stable_std': [],
        
        # Unstable only
        'mae_unstable': [],
        'r2_unstable': [],
        
        # Stability metrics
        'stability_rate': [],
        'physical_pass_rate': [],
        'chemical_pass_rate': [],
        'statistical_pass_rate': [],
        'n_stable': [],
        'n_unstable': [],
    }
    
    for n in sample_sizes:
        exp = experiments[str(n)]
        
        if 'error' in exp:
            # Fill with NaN for failed experiments
            for key in metrics:
                if key not in ['sample_sizes', 'baseline_mae', 'baseline_r2', 'baseline_rmse']:
                    if isinstance(metrics[key], list):
                        metrics[key].append(np.nan)
            continue
        
        # All synthetic metrics
        metrics['mae_all'].append(exp['accuracy_all_synthetic']['mean_mae'])
        metrics['mae_all_std'].append(exp['accuracy_all_synthetic']['std_mae'])
        metrics['r2_all'].append(exp['accuracy_all_synthetic']['mean_r2'])
        metrics['r2_all_std'].append(exp['accuracy_all_synthetic']['std_r2'])
        metrics['rmse_all'].append(exp['accuracy_all_synthetic']['mean_rmse'])
        metrics['rmse_all_std'].append(exp['accuracy_all_synthetic']['std_rmse'])
        
        # Stable only metrics
        stable = exp['accuracy_stable_only']
        metrics['mae_stable'].append(stable['mean_mae'] if stable['mean_mae'] else np.nan)
        metrics['mae_stable_std'].append(stable['std_mae'] if stable['std_mae'] else np.nan)
        metrics['r2_stable'].append(stable['mean_r2'] if stable['mean_r2'] else np.nan)
        metrics['r2_stable_std'].append(stable['std_r2'] if stable['std_r2'] else np.nan)
        metrics['rmse_stable'].append(stable['mean_rmse'] if stable['mean_rmse'] else np.nan)
        metrics['rmse_stable_std'].append(stable['std_rmse'] if stable['std_rmse'] else np.nan)
        
        # Unstable only metrics
        unstable = exp['accuracy_unstable_only']
        metrics['mae_unstable'].append(unstable['mean_mae'] if unstable['mean_mae'] else np.nan)
        metrics['r2_unstable'].append(unstable['mean_r2'] if unstable['mean_r2'] else np.nan)
        
        # Stability metrics
        val = exp['validation']
        metrics['stability_rate'].append(val['stability_rate'])
        metrics['physical_pass_rate'].append(val['physical_pass_rate'])
        metrics['chemical_pass_rate'].append(val['chemical_pass_rate'])
        metrics['statistical_pass_rate'].append(val['statistical_pass_rate'])
        metrics['n_stable'].append(val['n_stable'])
        metrics['n_unstable'].append(val['n_unstable'])
    
    # Convert lists to numpy arrays
    for key in metrics:
        if isinstance(metrics[key], list):
            metrics[key] = np.array(metrics[key])
    
    return metrics


def plot_mae_comparison(metrics, output_dir):
    """Plot MAE comparison: Baseline vs All Synthetic vs Stable Only"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sample_sizes = metrics['sample_sizes']
    x = np.arange(len(sample_sizes))
    
    # Baseline line
    ax.axhline(y=metrics['baseline_mae'], color='black', linestyle='--', 
               linewidth=2, label=f"Baseline (MAE={metrics['baseline_mae']:.4f})")
    
    # All synthetic
    ax.errorbar(x, metrics['mae_all'], yerr=metrics['mae_all_std'], 
                marker='o', markersize=8, linewidth=2, capsize=5,
                color='#2196F3', label='All Synthetic')
    
    # Stable only
    ax.errorbar(x, metrics['mae_stable'], yerr=metrics['mae_stable_std'],
                marker='s', markersize=8, linewidth=2, capsize=5,
                color='#4CAF50', label='Stable Only')
    
    ax.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title('MAE vs Number of Synthetic Samples', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotations
    best_all_idx = np.nanargmin(metrics['mae_all'])
    best_stable_idx = np.nanargmin(metrics['mae_stable'])
    
    improvement_all = (metrics['baseline_mae'] - metrics['mae_all'][best_all_idx]) / metrics['baseline_mae'] * 100
    improvement_stable = (metrics['baseline_mae'] - metrics['mae_stable'][best_stable_idx]) / metrics['baseline_mae'] * 100
    
    textstr = f"Best All Synthetic: {metrics['mae_all'][best_all_idx]:.4f} ({improvement_all:+.1f}%)\n"
    textstr += f"Best Stable Only: {metrics['mae_stable'][best_stable_idx]:.4f} ({improvement_stable:+.1f}%)"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: mae_comparison.png")


def plot_r2_comparison(metrics, output_dir):
    """Plot R² comparison: Baseline vs All Synthetic vs Stable Only"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sample_sizes = metrics['sample_sizes']
    x = np.arange(len(sample_sizes))
    
    # Baseline line
    ax.axhline(y=metrics['baseline_r2'], color='black', linestyle='--', 
               linewidth=2, label=f"Baseline (R²={metrics['baseline_r2']:.4f})")
    
    # All synthetic
    ax.errorbar(x, metrics['r2_all'], yerr=metrics['r2_all_std'], 
                marker='o', markersize=8, linewidth=2, capsize=5,
                color='#2196F3', label='All Synthetic')
    
    # Stable only
    ax.errorbar(x, metrics['r2_stable'], yerr=metrics['r2_stable_std'],
                marker='s', markersize=8, linewidth=2, capsize=5,
                color='#4CAF50', label='Stable Only')
    
    ax.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score vs Number of Synthetic Samples', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotations
    best_all_idx = np.nanargmax(metrics['r2_all'])
    best_stable_idx = np.nanargmax(metrics['r2_stable'])
    
    improvement_all = (metrics['r2_all'][best_all_idx] - metrics['baseline_r2']) / abs(metrics['baseline_r2']) * 100
    improvement_stable = (metrics['r2_stable'][best_stable_idx] - metrics['baseline_r2']) / abs(metrics['baseline_r2']) * 100
    
    textstr = f"Best All Synthetic: {metrics['r2_all'][best_all_idx]:.4f} ({improvement_all:+.1f}%)\n"
    textstr += f"Best Stable Only: {metrics['r2_stable'][best_stable_idx]:.4f} ({improvement_stable:+.1f}%)"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: r2_comparison.png")


def plot_stability_rates(metrics, output_dir):
    """Plot stability rates across sample sizes"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sample_sizes = metrics['sample_sizes']
    x = np.arange(len(sample_sizes))
    width = 0.2
    
    # Plot each stability type
    ax.bar(x - 1.5*width, metrics['physical_pass_rate'] * 100, width, 
           label='Physical', color='#FF9800', alpha=0.8)
    ax.bar(x - 0.5*width, metrics['chemical_pass_rate'] * 100, width,
           label='Chemical', color='#9C27B0', alpha=0.8)
    ax.bar(x + 0.5*width, metrics['statistical_pass_rate'] * 100, width,
           label='Statistical', color='#00BCD4', alpha=0.8)
    ax.bar(x + 1.5*width, metrics['stability_rate'] * 100, width,
           label='Overall', color='#4CAF50', alpha=0.8)
    
    ax.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Stability Validation Pass Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add average stability annotation
    avg_stability = np.nanmean(metrics['stability_rate']) * 100
    ax.axhline(y=avg_stability, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(len(sample_sizes)-1, avg_stability + 2, f'Avg: {avg_stability:.1f}%', 
            fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: stability_rates.png")


def plot_stable_vs_unstable_count(metrics, output_dir):
    """Plot number of stable vs unstable samples"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sample_sizes = metrics['sample_sizes']
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    ax.bar(x - width/2, metrics['n_stable'], width, label='Stable', color='#4CAF50', alpha=0.8)
    ax.bar(x + width/2, metrics['n_unstable'], width, label='Unstable', color='#F44336', alpha=0.8)
    
    ax.set_xlabel('Number of Synthetic Samples Requested', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Stable vs Unstable Sample Counts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (stable, unstable) in enumerate(zip(metrics['n_stable'], metrics['n_unstable'])):
        total = stable + unstable
        if total > 0:
            pct = stable / total * 100
            ax.text(i, stable + unstable/2, f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stable_unstable_counts.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: stable_unstable_counts.png")


def plot_improvement_over_baseline(metrics, output_dir):
    """Plot percentage improvement over baseline"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sample_sizes = metrics['sample_sizes']
    x = np.arange(len(sample_sizes))
    
    # MAE improvement (negative is better, so we flip)
    mae_improvement_all = (metrics['baseline_mae'] - metrics['mae_all']) / metrics['baseline_mae'] * 100
    mae_improvement_stable = (metrics['baseline_mae'] - metrics['mae_stable']) / metrics['baseline_mae'] * 100
    
    ax1.bar(x - 0.2, mae_improvement_all, 0.4, label='All Synthetic', color='#2196F3', alpha=0.8)
    ax1.bar(x + 0.2, mae_improvement_stable, 0.4, label='Stable Only', color='#4CAF50', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax1.set_ylabel('MAE Improvement (%)', fontsize=12)
    ax1.set_title('MAE Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # R² improvement
    r2_improvement_all = (metrics['r2_all'] - metrics['baseline_r2']) / abs(metrics['baseline_r2']) * 100
    r2_improvement_stable = (metrics['r2_stable'] - metrics['baseline_r2']) / abs(metrics['baseline_r2']) * 100
    
    ax2.bar(x - 0.2, r2_improvement_all, 0.4, label='All Synthetic', color='#2196F3', alpha=0.8)
    ax2.bar(x + 0.2, r2_improvement_stable, 0.4, label='Stable Only', color='#4CAF50', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax2.set_ylabel('R² Improvement (%)', fontsize=12)
    ax2.set_title('R² Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_over_baseline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: improvement_over_baseline.png")


def plot_stable_vs_unstable_performance(metrics, output_dir):
    """Compare performance when training with stable vs unstable samples"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sample_sizes = metrics['sample_sizes']
    x = np.arange(len(sample_sizes))
    
    # MAE comparison
    ax1.plot(x, metrics['mae_stable'], 'o-', markersize=8, linewidth=2,
             color='#4CAF50', label='Stable Only')
    ax1.plot(x, metrics['mae_unstable'], 's-', markersize=8, linewidth=2,
             color='#F44336', label='Unstable Only')
    ax1.axhline(y=metrics['baseline_mae'], color='black', linestyle='--', 
                linewidth=2, label='Baseline')
    
    ax1.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('MAE: Stable vs Unstable Samples', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # R² comparison
    ax2.plot(x, metrics['r2_stable'], 'o-', markersize=8, linewidth=2,
             color='#4CAF50', label='Stable Only')
    ax2.plot(x, metrics['r2_unstable'], 's-', markersize=8, linewidth=2,
             color='#F44336', label='Unstable Only')
    ax2.axhline(y=metrics['baseline_r2'], color='black', linestyle='--', 
                linewidth=2, label='Baseline')
    
    ax2.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R²: Stable vs Unstable Samples', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stable_vs_unstable_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: stable_vs_unstable_performance.png")


def plot_comprehensive_summary(metrics, output_dir):
    """Create a comprehensive 2x2 summary plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    sample_sizes = metrics['sample_sizes']
    x = np.arange(len(sample_sizes))
    
    # Plot 1: MAE with error bars
    ax1 = axes[0, 0]
    ax1.axhline(y=metrics['baseline_mae'], color='black', linestyle='--', linewidth=2, label='Baseline')
    ax1.errorbar(x, metrics['mae_all'], yerr=metrics['mae_all_std'], 
                 marker='o', markersize=6, linewidth=2, capsize=4,
                 color='#2196F3', label='All Synthetic')
    ax1.errorbar(x, metrics['mae_stable'], yerr=metrics['mae_stable_std'],
                 marker='s', markersize=6, linewidth=2, capsize=4,
                 color='#4CAF50', label='Stable Only')
    ax1.set_xlabel('N Synthetic Samples')
    ax1.set_ylabel('MAE')
    ax1.set_title('Mean Absolute Error', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² with error bars
    ax2 = axes[0, 1]
    ax2.axhline(y=metrics['baseline_r2'], color='black', linestyle='--', linewidth=2, label='Baseline')
    ax2.errorbar(x, metrics['r2_all'], yerr=metrics['r2_all_std'], 
                 marker='o', markersize=6, linewidth=2, capsize=4,
                 color='#2196F3', label='All Synthetic')
    ax2.errorbar(x, metrics['r2_stable'], yerr=metrics['r2_stable_std'],
                 marker='s', markersize=6, linewidth=2, capsize=4,
                 color='#4CAF50', label='Stable Only')
    ax2.set_xlabel('N Synthetic Samples')
    ax2.set_ylabel('R²')
    ax2.set_title('R² Score', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stability rates
    ax3 = axes[1, 0]
    ax3.plot(x, metrics['stability_rate'] * 100, 'o-', markersize=8, linewidth=2,
             color='#4CAF50', label='Overall')
    ax3.plot(x, metrics['physical_pass_rate'] * 100, 's--', markersize=6, linewidth=1.5,
             color='#FF9800', alpha=0.7, label='Physical')
    ax3.plot(x, metrics['chemical_pass_rate'] * 100, '^--', markersize=6, linewidth=1.5,
             color='#9C27B0', alpha=0.7, label='Chemical')
    ax3.plot(x, metrics['statistical_pass_rate'] * 100, 'd--', markersize=6, linewidth=1.5,
             color='#00BCD4', alpha=0.7, label='Statistical')
    ax3.set_xlabel('N Synthetic Samples')
    ax3.set_ylabel('Pass Rate (%)')
    ax3.set_title('Stability Validation Rates', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax3.legend(loc='best', fontsize=9)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Improvement over baseline
    ax4 = axes[1, 1]
    mae_improvement = (metrics['baseline_mae'] - metrics['mae_all']) / metrics['baseline_mae'] * 100
    r2_improvement = (metrics['r2_all'] - metrics['baseline_r2']) / abs(metrics['baseline_r2']) * 100
    
    ax4.bar(x - 0.2, mae_improvement, 0.4, label='MAE Improvement', color='#2196F3', alpha=0.8)
    ax4.bar(x + 0.2, r2_improvement, 0.4, label='R² Improvement', color='#4CAF50', alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('N Synthetic Samples')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Improvement Over Baseline', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(s) for s in sample_sizes], rotation=45)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MMLI VAE-KDE Data Augmentation Results Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: comprehensive_summary.png")


def plot_log_scale_performance(metrics, output_dir):
    """Plot performance metrics with log scale x-axis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sample_sizes = np.array(metrics['sample_sizes'])
    
    # MAE on log scale
    ax1.axhline(y=metrics['baseline_mae'], color='black', linestyle='--', linewidth=2, label='Baseline')
    ax1.semilogx(sample_sizes, metrics['mae_all'], 'o-', markersize=8, linewidth=2,
                 color='#2196F3', label='All Synthetic')
    ax1.semilogx(sample_sizes, metrics['mae_stable'], 's-', markersize=8, linewidth=2,
                 color='#4CAF50', label='Stable Only')
    ax1.set_xlabel('Number of Synthetic Samples (log scale)', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('MAE vs Sample Size (Log Scale)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # R² on log scale
    ax2.axhline(y=metrics['baseline_r2'], color='black', linestyle='--', linewidth=2, label='Baseline')
    ax2.semilogx(sample_sizes, metrics['r2_all'], 'o-', markersize=8, linewidth=2,
                 color='#2196F3', label='All Synthetic')
    ax2.semilogx(sample_sizes, metrics['r2_stable'], 's-', markersize=8, linewidth=2,
                 color='#4CAF50', label='Stable Only')
    ax2.set_xlabel('Number of Synthetic Samples (log scale)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² vs Sample Size (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_log_scale.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: performance_log_scale.png")


def generate_summary_table(metrics, output_dir):
    """Generate and save a summary table as text and CSV"""
    sample_sizes = metrics['sample_sizes']
    
    # Create summary table
    lines = []
    lines.append("=" * 120)
    lines.append("MMLI VAE-KDE DATA AUGMENTATION EXPERIMENT SUMMARY")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Baseline MAE: {metrics['baseline_mae']:.4f}")
    lines.append(f"Baseline R²:  {metrics['baseline_r2']:.4f}")
    lines.append(f"Baseline RMSE: {metrics['baseline_rmse']:.4f}")
    lines.append("")
    lines.append("-" * 120)
    header = f"{'N Synthetic':>12} | {'Stability':>10} | {'MAE (All)':>12} | {'MAE (Stable)':>12} | {'R² (All)':>10} | {'R² (Stable)':>10} | {'Δ MAE %':>10} | {'Δ R² %':>10}"
    lines.append(header)
    lines.append("-" * 120)
    
    for i, n in enumerate(sample_sizes):
        stability = f"{metrics['stability_rate'][i]*100:.1f}%"
        mae_all = f"{metrics['mae_all'][i]:.4f}" if not np.isnan(metrics['mae_all'][i]) else "N/A"
        mae_stable = f"{metrics['mae_stable'][i]:.4f}" if not np.isnan(metrics['mae_stable'][i]) else "N/A"
        r2_all = f"{metrics['r2_all'][i]:.4f}" if not np.isnan(metrics['r2_all'][i]) else "N/A"
        r2_stable = f"{metrics['r2_stable'][i]:.4f}" if not np.isnan(metrics['r2_stable'][i]) else "N/A"
        
        # Calculate improvements
        if not np.isnan(metrics['mae_all'][i]):
            mae_improvement = (metrics['baseline_mae'] - metrics['mae_all'][i]) / metrics['baseline_mae'] * 100
            mae_imp_str = f"{mae_improvement:+.1f}%"
        else:
            mae_imp_str = "N/A"
            
        if not np.isnan(metrics['r2_all'][i]):
            r2_improvement = (metrics['r2_all'][i] - metrics['baseline_r2']) / abs(metrics['baseline_r2']) * 100
            r2_imp_str = f"{r2_improvement:+.1f}%"
        else:
            r2_imp_str = "N/A"
        
        line = f"{n:>12} | {stability:>10} | {mae_all:>12} | {mae_stable:>12} | {r2_all:>10} | {r2_stable:>10} | {mae_imp_str:>10} | {r2_imp_str:>10}"
        lines.append(line)
    
    lines.append("-" * 120)
    lines.append("")
    
    # Find best performing configuration
    best_mae_idx = np.nanargmin(metrics['mae_all'])
    best_r2_idx = np.nanargmax(metrics['r2_all'])
    
    lines.append("BEST CONFIGURATIONS:")
    lines.append(f"  Lowest MAE: {sample_sizes[best_mae_idx]} synthetic samples (MAE={metrics['mae_all'][best_mae_idx]:.4f})")
    lines.append(f"  Highest R²: {sample_sizes[best_r2_idx]} synthetic samples (R²={metrics['r2_all'][best_r2_idx]:.4f})")
    lines.append("")
    lines.append("=" * 120)
    
    # Save to text file
    summary_text = "\n".join(lines)
    with open(os.path.join(output_dir, 'summary_table.txt'), 'w') as f:
        f.write(summary_text)
    print(f"Saved: summary_table.txt")
    
    # Save to CSV
    import csv
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N_Synthetic', 'Stability_Rate', 'MAE_All', 'MAE_Stable', 'MAE_Unstable',
                         'R2_All', 'R2_Stable', 'R2_Unstable', 'RMSE_All', 'RMSE_Stable',
                         'Physical_Pass', 'Chemical_Pass', 'Statistical_Pass'])
        for i, n in enumerate(sample_sizes):
            writer.writerow([
                n,
                metrics['stability_rate'][i],
                metrics['mae_all'][i],
                metrics['mae_stable'][i],
                metrics['mae_unstable'][i],
                metrics['r2_all'][i],
                metrics['r2_stable'][i],
                metrics['r2_unstable'][i],
                metrics['rmse_all'][i],
                metrics['rmse_stable'][i],
                metrics['physical_pass_rate'][i],
                metrics['chemical_pass_rate'][i],
                metrics['statistical_pass_rate'][i]
            ])
    print(f"Saved: summary_table.csv")
    
    # Print to console
    print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser(description='Generate plots for MMLI experiment results')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to full_results.json file')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to experiment results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as results)')
    
    args = parser.parse_args()
    
    # Determine paths
    if args.results_path:
        results_path = args.results_path
        default_output = os.path.dirname(results_path)
    elif args.results_dir:
        results_path = None
        default_output = args.results_dir
    else:
        # Try to find results in current directory
        import glob
        matches = glob.glob('results/mmli_experiment_*/full_results.json')
        if matches:
            results_path = sorted(matches)[-1]  # Most recent
            default_output = os.path.dirname(results_path)
            print(f"Found results: {results_path}")
        else:
            print("Error: No results found. Please specify --results_path or --results_dir")
            return
    
    output_dir = args.output_dir if args.output_dir else default_output
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots subdirectory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Loading results...")
    results = load_results(results_path=results_path, results_dir=args.results_dir)
    
    print(f"Extracting metrics...")
    metrics = extract_metrics(results)
    
    print(f"\nGenerating plots in: {plots_dir}")
    print("-" * 50)
    
    # Generate all plots
    plot_mae_comparison(metrics, plots_dir)
    plot_r2_comparison(metrics, plots_dir)
    plot_stability_rates(metrics, plots_dir)
    plot_stable_vs_unstable_count(metrics, plots_dir)
    plot_improvement_over_baseline(metrics, plots_dir)
    plot_stable_vs_unstable_performance(metrics, plots_dir)
    plot_comprehensive_summary(metrics, plots_dir)
    plot_log_scale_performance(metrics, plots_dir)
    
    # Generate summary table
    generate_summary_table(metrics, plots_dir)
    
    print("-" * 50)
    print(f"\nAll plots saved to: {plots_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(plots_dir)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
