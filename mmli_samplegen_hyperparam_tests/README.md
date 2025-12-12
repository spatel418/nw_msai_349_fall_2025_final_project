# MMLI VAE-KDE Data Augmentation Experiment

This experiment tests how different amounts of synthetic data (generated using VAE-KDE) affect prediction accuracy on the MMLI dataset.

## Overview

The experiment:
1. Generates varying numbers of synthetic samples (20, 60, 100, 500, 1K, 3K, 6K, 10K, 20K, 30K)
2. Validates synthetic samples for molecular stability (physical, chemical, statistical)
3. Trains regression models and measures accuracy on:
   - Baseline (no augmentation)
   - All synthetic samples
   - Only stable synthetic samples
   - Only unstable synthetic samples

## Files

- `run_experiment.py` - Main experiment runner
- `vae_kde_augmenter.py` - VAE-KDE synthetic sample generation
- `stability_validator.py` - Molecular stability validation
- `simple_regression.py` - Regression framework for training/evaluation
- `requirements.txt` - Required Python packages

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Place your `mmli.csv` file in the same folder, then run:

```bash
python run_experiment.py --data_path mmli.csv --target T80
```

### Full Options

```bash
python run_experiment.py \
    --data_path mmli.csv \
    --target T80 \
    --output_dir ./results \
    --model randomforest \
    --k_folds 5 \
    --sample_sizes "20,60,100,500,1000,3000,6000,10000,20000,30000" \
    --workers 4
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | (required) | Path to MMLI CSV file |
| `--target` | `T80` | Target column name |
| `--output_dir` | `./results` | Output directory for results |
| `--model` | `randomforest` | Model type: `linear`, `bayesian`, `elasticnet`, `xgboost`, `randomforest`, `svm` |
| `--k_folds` | `5` | Number of cross-validation folds |
| `--sample_sizes` | All sizes | Comma-separated list of synthetic sample sizes |
| `--workers` | CPU count - 1 | Number of parallel workers |
| `--sequential` | False | Run sequentially (no parallelism) |

### Quick Test (Smaller Sample Sizes)

For a quick test with fewer samples:

```bash
python run_experiment.py --data_path mmli.csv --target T80 --sample_sizes "20,60,100"
```

## Performance Optimization

### Apple M1/M2 GPU Support (MPS)

The VAE training automatically uses Apple's Metal Performance Shaders (MPS) when available on M1/M2 Macs. No additional configuration needed - PyTorch will detect and use the GPU automatically.

To verify MPS is being used:
```bash
python test_dependencies.py
# Look for: MPS (Apple M1/M2) available: True
```

### Parallelism

By default, experiments for different sample sizes run in parallel using multiple CPU cores:

```bash
# Use 4 parallel workers
python run_experiment.py --data_path mmli.csv --target T80 --workers 4

# Use all available cores minus 1 (default)
python run_experiment.py --data_path mmli.csv --target T80

# Run sequentially (useful for debugging)
python run_experiment.py --data_path mmli.csv --target T80 --sequential
```

**Note**: Each parallel worker will use the GPU for VAE training, so memory usage scales with number of workers. On M1 Macs with limited unified memory, you may want to reduce workers:

```bash
# Conservative for 8GB M1
python run_experiment.py --data_path mmli.csv --target T80 --workers 2

# For 16GB+ M1
python run_experiment.py --data_path mmli.csv --target T80 --workers 4
```

## Output Structure

```
results/
└── mmli_experiment_YYYYMMDD_HHMMSS/
    ├── baseline/
    │   └── fold_1/, fold_2/, ...
    ├── baseline_results.json
    ├── n_synthetic_20/
    │   ├── validation_results.json
    │   ├── experiment_results.json
    │   ├── all_synthetic/
    │   ├── stable_synthetic/
    │   └── unstable_synthetic/
    ├── n_synthetic_60/
    │   └── ...
    └── full_results.json
```

## Metrics

### Prediction Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of Determination

### Stability Metrics
- **Physical Stability**: Features within observed physical ranges (±10% tolerance)
- **Chemical Stability**: Correlation patterns maintained between features
- **Statistical Stability**: Samples follow learned distribution (ensemble outlier detection)

## Expected Results

The experiment will produce a summary table like:

```
N Synthetic |  Stability |   MAE (All) | MAE (Stable) |   R² (All) | R² (Stable)
----------------------------------------------------------------------------------
   Baseline |        N/A |      X.XXXX |          N/A |     X.XXXX |         N/A
         20 |      95.0% |      X.XXXX |       X.XXXX |     X.XXXX |      X.XXXX
         60 |      92.5% |      X.XXXX |       X.XXXX |     X.XXXX |      X.XXXX
        100 |      90.0% |      X.XXXX |       X.XXXX |     X.XXXX |      X.XXXX
        ...
```

## Notes

1. **MMLI Dataset**: This experiment is designed for the MMLI (Molecular Machine Learning Initiative) dataset with ~43 samples. The target is typically `T80`.

2. **VAE Training**: The VAE is configured for very small datasets with appropriate regularization (high dropout, low beta).

3. **Stability Validation**: Uses ensemble of outlier detection methods (LOF, Isolation Forest, Elliptic Envelope) for robust validation.

4. **Cross-Validation**: Evaluation is always performed on original (organic) samples only, but training can include synthetic samples.

## Approach Details

This implementation follows the same VAE-KDE approach used for the Vaskas dataset:

1. **VAE Architecture**: 2-layer encoder/decoder with dropout regularization
2. **Latent Space Interpolation**: Generates new samples by interpolating between nearest neighbors in latent space
3. **KDE-Based Sampling**: Uses Kernel Density Estimation for balanced sampling across the target distribution
4. **Stability Validation**: Multi-dimensional validation checking physical ranges, chemical correlations, and statistical distribution
