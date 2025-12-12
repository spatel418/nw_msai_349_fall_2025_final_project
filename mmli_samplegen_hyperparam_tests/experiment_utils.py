"""
Experiment Utilities
====================
Helper functions for running and managing experiments.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be skipped.")


def setup_experiment_directory(base_name="experiment"):
    """
    Create a timestamped experiment directory.
    
    Parameters:
    -----------
    base_name : str
        Base name for the experiment directory
    
    Returns:
    --------
    str : Path to the created directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{base_name}_{timestamp}"
    
    os.makedirs(dir_name, exist_ok=True)
    
    return dir_name


def save_experiment_results(results, output_dir):
    """
    Save experiment results to JSON.
    
    Parameters:
    -----------
    results : dict
        Experiment results
    output_dir : str
        Directory to save results
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results_converted = convert_numpy(results)
    
    output_path = os.path.join(output_dir, "all_results.json")
    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")


def generate_summary_report(results, output_dir):
    """
    Generate a summary report of the experiment.
    
    Parameters:
    -----------
    results : dict
        Experiment results
    output_dir : str
        Directory to save report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MMLI Sample Size Scaling Experiment - Summary Report")
    lines.append("=" * 70)
    lines.append("")
    
    # Configuration
    lines.append("CONFIGURATION")
    lines.append("-" * 40)
    config = results.get('config', {})
    for key, value in config.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    
    # Baseline results
    lines.append("BASELINE RESULTS (No Augmentation)")
    lines.append("-" * 40)
    baseline = results.get('baseline', {})
    if baseline:
        lines.append(f"  Number of samples: {baseline.get('n_samples', 'N/A')}")
        metrics = baseline.get('metrics', {})
        for model_name, model_metrics in metrics.items():
            if 'error' not in model_metrics:
                test_metrics = model_metrics.get('test', {})
                lines.append(f"  {model_name}:")
                lines.append(f"    Test R²: {test_metrics.get('r2', 'N/A'):.4f}")
                lines.append(f"    Test MAE: {test_metrics.get('mae', 'N/A'):.4f}")
    lines.append("")
    
    # Experiment results
    lines.append("AUGMENTATION EXPERIMENT RESULTS")
    lines.append("-" * 40)
    
    experiments = results.get('experiments', {})
    
    # Create summary table
    summary_data = []
    
    for n_synthetic, exp_results in sorted(experiments.items(), key=lambda x: int(x[0])):
        if 'error' in exp_results:
            continue
        
        row = {
            'N Synthetic Requested': n_synthetic,
            'N Generated': exp_results.get('n_synthetic_generated', 'N/A'),
            'N Stable': exp_results.get('n_stable', 'N/A'),
            'Stability Rate': f"{exp_results.get('stability_rate', 0)*100:.1f}%",
            'Gen Time (s)': f"{exp_results.get('generation_time', 0):.1f}",
        }
        
        # Best model results on all synthetic
        metrics_all = exp_results.get('metrics_all', {})
        best_r2 = -np.inf
        best_model = 'N/A'
        for model_name, model_metrics in metrics_all.items():
            if 'error' not in model_metrics:
                test_r2 = model_metrics.get('test', {}).get('r2', -np.inf)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model_name
        
        row['Best Model (All)'] = best_model
        row['Best R² (All)'] = f"{best_r2:.4f}" if best_r2 > -np.inf else 'N/A'
        
        # Best model results on stable synthetic
        metrics_stable = exp_results.get('metrics_stable', {})
        best_r2_stable = -np.inf
        for model_name, model_metrics in metrics_stable.items():
            if 'error' not in model_metrics:
                test_r2 = model_metrics.get('test', {}).get('r2', -np.inf)
                if test_r2 > best_r2_stable:
                    best_r2_stable = test_r2
        
        row['Best R² (Stable)'] = f"{best_r2_stable:.4f}" if best_r2_stable > -np.inf else 'N/A'
        
        summary_data.append(row)
    
    # Print summary table
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        lines.append(df_summary.to_string(index=False))
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    # Save report
    report_text = "\n".join(lines)
    
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print("\n" + report_text)
    
    # Save summary as CSV
    if summary_data:
        df_summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


def plot_results(results, output_dir):
    """
    Generate plots of experiment results.
    
    Parameters:
    -----------
    results : dict
        Experiment results
    output_dir : str
        Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plots - matplotlib not available")
        return
    
    experiments = results.get('experiments', {})
    baseline = results.get('baseline', {})
    
    # Prepare data
    sample_sizes = []
    r2_all = []
    r2_stable = []
    stability_rates = []
    gen_times = []
    
    for n_synthetic, exp_results in sorted(experiments.items(), key=lambda x: int(x[0])):
        if 'error' in exp_results:
            continue
        
        sample_sizes.append(int(n_synthetic))
        stability_rates.append(exp_results.get('stability_rate', 0) * 100)
        gen_times.append(exp_results.get('generation_time', 0))
        
        # Get best R² from all models
        metrics_all = exp_results.get('metrics_all', {})
        best_r2 = -np.inf
        for model_metrics in metrics_all.values():
            if isinstance(model_metrics, dict) and 'error' not in model_metrics:
                test_r2 = model_metrics.get('test', {}).get('r2', -np.inf)
                if test_r2 > best_r2:
                    best_r2 = test_r2
        r2_all.append(best_r2 if best_r2 > -np.inf else np.nan)
        
        # Get best R² from stable
        metrics_stable = exp_results.get('metrics_stable', {})
        best_r2_stable = -np.inf
        for model_metrics in metrics_stable.values():
            if isinstance(model_metrics, dict) and 'error' not in model_metrics:
                test_r2 = model_metrics.get('test', {}).get('r2', -np.inf)
                if test_r2 > best_r2_stable:
                    best_r2_stable = test_r2
        r2_stable.append(best_r2_stable if best_r2_stable > -np.inf else np.nan)
    
    # Get baseline R²
    baseline_r2 = -np.inf
    baseline_metrics = baseline.get('metrics', {})
    for model_metrics in baseline_metrics.values():
        if isinstance(model_metrics, dict) and 'error' not in model_metrics:
            test_r2 = model_metrics.get('test', {}).get('r2', -np.inf)
            if test_r2 > baseline_r2:
                baseline_r2 = test_r2
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: R² vs Sample Size
    ax1 = axes[0, 0]
    ax1.semilogx(sample_sizes, r2_all, 'o-', label='All Synthetic', linewidth=2, markersize=8)
    ax1.semilogx(sample_sizes, r2_stable, 's--', label='Stable Only', linewidth=2, markersize=8)
    if baseline_r2 > -np.inf:
        ax1.axhline(y=baseline_r2, color='r', linestyle=':', label=f'Baseline (R²={baseline_r2:.3f})', linewidth=2)
    ax1.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax1.set_ylabel('Test R²', fontsize=12)
    ax1.set_title('Model Performance vs Synthetic Sample Size', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stability Rate vs Sample Size
    ax2 = axes[0, 1]
    ax2.semilogx(sample_sizes, stability_rates, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax2.set_ylabel('Stability Rate (%)', fontsize=12)
    ax2.set_title('Synthetic Sample Stability vs Sample Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Generation Time vs Sample Size
    ax3 = axes[1, 0]
    ax3.loglog(sample_sizes, gen_times, 'o-', color='orange', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax3.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax3.set_title('Generation Time vs Sample Size', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: R² Improvement over Baseline
    ax4 = axes[1, 1]
    if baseline_r2 > -np.inf:
        r2_improvement_all = [r - baseline_r2 if not np.isnan(r) else np.nan for r in r2_all]
        r2_improvement_stable = [r - baseline_r2 if not np.isnan(r) else np.nan for r in r2_stable]
        ax4.semilogx(sample_sizes, r2_improvement_all, 'o-', label='All Synthetic', linewidth=2, markersize=8)
        ax4.semilogx(sample_sizes, r2_improvement_stable, 's--', label='Stable Only', linewidth=2, markersize=8)
        ax4.axhline(y=0, color='r', linestyle=':', linewidth=2)
        ax4.set_xlabel('Number of Synthetic Samples', fontsize=12)
        ax4.set_ylabel('R² Improvement over Baseline', fontsize=12)
        ax4.set_title('R² Improvement vs Sample Size', fontsize=14)
        ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, "experiment_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {plot_path}")


def generate_model_comparison_plot(results, output_dir):
    """Generate a plot comparing different models across sample sizes."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    experiments = results.get('experiments', {})
    models = results.get('config', {}).get('models', [])
    
    if not models or not experiments:
        return
    
    # Collect data per model
    model_data = {m: {'sizes': [], 'r2': []} for m in models}
    
    for n_synthetic, exp_results in sorted(experiments.items(), key=lambda x: int(x[0])):
        if 'error' in exp_results:
            continue
        
        metrics_all = exp_results.get('metrics_all', {})
        
        for model_name in models:
            if model_name in metrics_all and 'error' not in metrics_all[model_name]:
                test_r2 = metrics_all[model_name].get('test', {}).get('r2', np.nan)
                model_data[model_name]['sizes'].append(int(n_synthetic))
                model_data[model_name]['r2'].append(test_r2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for model_name, color in zip(models, colors):
        data = model_data[model_name]
        if data['sizes']:
            ax.semilogx(data['sizes'], data['r2'], 'o-', 
                       label=model_name, color=color, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Synthetic Samples', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Model Comparison Across Sample Sizes', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to {plot_path}")
