"""
Vaskas Dataset Explorer
Analyzes the Vaskas dataset to understand its structure, features, and distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_vaskas_dataset(filepath):
    """Load and perform initial inspection of Vaskas dataset"""
    df = pd.read_csv(filepath)
    print("="*80)
    print("VASKAS DATASET OVERVIEW")
    print("="*80)
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    return df

def analyze_columns(df):
    """Analyze column types and content"""
    print("\n" + "="*80)
    print("COLUMN ANALYSIS")
    print("="*80)
    
    print("\nAll columns:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    print("\nColumn data types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values!")
    
    return df.columns.tolist()

def identify_target(df):
    """Try to identify the target variable"""
    print("\n" + "="*80)
    print("TARGET VARIABLE IDENTIFICATION")
    print("="*80)
    
    # Common target names
    possible_targets = ['yield', 'activity', 'conversion', 'selectivity', 
                       'rate', 'barrier', 'energy', 'activation']
    
    found_targets = []
    for col in df.columns:
        if any(target in col.lower() for target in possible_targets):
            found_targets.append(col)
    
    if found_targets:
        print(f"\nPossible target columns: {found_targets}")
        for target in found_targets:
            print(f"\n{target} statistics:")
            print(df[target].describe())
    else:
        print("\nNo obvious target column found. Please specify manually.")
        print("Here are numeric columns that could be targets:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            print(f"  - {col}")
    
    return found_targets

def analyze_target_distribution(df, target_col):
    """Analyze the distribution of the target variable"""
    if target_col not in df.columns:
        print(f"Error: {target_col} not found in dataset")
        return
    
    print("\n" + "="*80)
    print(f"TARGET DISTRIBUTION ANALYSIS: {target_col}")
    print("="*80)
    
    target = df[target_col]
    
    print("\nBasic statistics:")
    print(target.describe())
    
    print(f"\nSkewness: {target.skew():.3f}")
    print(f"Kurtosis: {target.kurtosis():.3f}")
    
    # Check for gaps in distribution
    sorted_vals = np.sort(target.values)
    gaps = np.diff(sorted_vals)
    large_gaps = gaps > np.percentile(gaps, 95)
    
    if large_gaps.sum() > 0:
        print(f"\nFound {large_gaps.sum()} large gaps in target distribution")
        print("This might indicate undersampled regions")
    
    return target

def visualize_distributions(df, target_col, save_dir=None):
    """Create visualizations of the dataset"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # Target distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram
    axes[0, 0].hist(df[target_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel(target_col)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of {target_col}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(df[target_col])
    axes[0, 1].set_ylabel(target_col)
    axes[0, 1].set_title(f'Box Plot of {target_col}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_target = np.sort(df[target_col])
    cumulative = np.arange(1, len(sorted_target) + 1) / len(sorted_target)
    axes[1, 0].plot(sorted_target, cumulative)
    axes[1, 0].set_xlabel(target_col)
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title(f'Cumulative Distribution of {target_col}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df[target_col], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'Q-Q Plot of {target_col}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved target distribution plot to {save_dir / 'target_distribution.png'}")
    else:
        plt.show()
    
    plt.close()
    
    # Correlation heatmap for top features
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        # Calculate correlation with target
        target_corr = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        top_features = target_corr.head(11).index.tolist()  # Top 10 + target
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric_df[top_features].corr(), annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f'Correlation Heatmap: Top 10 Features Correlated with {target_col}')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"Saved correlation heatmap to {save_dir / 'correlation_heatmap.png'}")
        else:
            plt.show()
        
        plt.close()
        
        print("\nTop 10 features correlated with target:")
        for i, (feat, corr) in enumerate(target_corr.head(11).items(), 1):
            if feat != target_col:
                print(f"{i-1}. {feat}: {corr:.3f}")

def identify_molecular_descriptors(columns):
    """Identify types of molecular descriptors present"""
    print("\n" + "="*80)
    print("MOLECULAR DESCRIPTOR TYPES")
    print("="*80)
    
    descriptor_types = {
        'Basic Properties': ['mass', 'weight', 'atoms', 'bonds'],
        'Electronic Properties': ['homo', 'lumo', 'charge', 'dipole', 'gap'],
        'Topological': ['ring', 'path', 'index', 'connectivity'],
        'Geometric': ['distance', 'angle', 'volume', 'area', 'radius'],
        'Quantum': ['orbital', 'energy', 'dos', 'tdos', 'sdos'],
        'Chemical': ['donor', 'acceptor', 'aromatic', 'hetero']
    }
    
    found_types = {dtype: [] for dtype in descriptor_types}
    
    for col in columns:
        col_lower = col.lower()
        for dtype, keywords in descriptor_types.items():
            if any(keyword in col_lower for keyword in keywords):
                found_types[dtype].append(col)
    
    for dtype, features in found_types.items():
        if features:
            print(f"\n{dtype} ({len(features)} features):")
            for feat in features[:5]:  # Show first 5
                print(f"  - {feat}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")
    
    return found_types

def generate_summary_report(df, target_col, save_dir=None):
    """Generate a summary report"""
    report = []
    report.append("="*80)
    report.append("VASKAS DATASET SUMMARY REPORT")
    report.append("="*80)
    report.append(f"\nDataset Size: {len(df)} samples")
    report.append(f"Number of Features: {len(df.columns) - 1}")  # Excluding target
    report.append(f"Target Variable: {target_col}")
    report.append(f"\nTarget Statistics:")
    report.append(f"  Mean: {df[target_col].mean():.3f}")
    report.append(f"  Std: {df[target_col].std():.3f}")
    report.append(f"  Min: {df[target_col].min():.3f}")
    report.append(f"  Max: {df[target_col].max():.3f}")
    report.append(f"  Skewness: {df[target_col].skew():.3f}")
    
    # Check if distribution is uniform or skewed
    if abs(df[target_col].skew()) > 1:
        report.append(f"\n⚠️  Target distribution is HIGHLY SKEWED")
        report.append(f"   This is a good use case for data augmentation!")
    elif abs(df[target_col].skew()) > 0.5:
        report.append(f"\n⚠️  Target distribution is MODERATELY SKEWED")
        report.append(f"   Data augmentation might help")
    else:
        report.append(f"\n✓  Target distribution is relatively UNIFORM")
        report.append(f"   Data augmentation may not be necessary")
    
    report_text = '\n'.join(report)
    print("\n" + report_text)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        with open(save_dir / 'dataset_summary.txt', 'w') as f:
            f.write(report_text)
        print(f"\nSaved summary report to {save_dir / 'dataset_summary.txt'}")

def main(vaskas_path, target_col=None, output_dir='vaskas_analysis'):
    """Main analysis pipeline"""
    
    # Load dataset
    df = load_vaskas_dataset(vaskas_path)
    
    # Analyze columns
    columns = analyze_columns(df)
    
    # Identify target if not provided
    if target_col is None:
        possible_targets = identify_target(df)
        if possible_targets:
            target_col = possible_targets[0]
            print(f"\nUsing '{target_col}' as target variable")
        else:
            print("\nPlease specify target column manually")
            return
    
    # Analyze target distribution
    analyze_target_distribution(df, target_col)
    
    # Identify descriptor types
    identify_molecular_descriptors(columns)
    
    # Create visualizations
    visualize_distributions(df, target_col, save_dir=output_dir)
    
    # Generate summary
    generate_summary_report(df, target_col, save_dir=output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return df, target_col

if __name__ == "__main__":
    # Update this path to your Vaskas dataset
    VASKAS_PATH = "vaskas.csv"
    
    # If you know the target column name, specify it here
    # Otherwise set to None and the script will try to find it
    TARGET_COL = None  # e.g., 'yield', 'activation_energy', etc.
    
    df, target = main(VASKAS_PATH, target_col=TARGET_COL)
