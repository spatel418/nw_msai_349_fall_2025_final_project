"""
Quick Test Script for MMLI Experiment
=====================================

Run this first to verify all dependencies are working correctly.
Creates a small synthetic dataset for testing.

Usage:
    python test_dependencies.py
"""

import sys
import traceback


def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"  ✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        # Check for Apple M1/M2 MPS backend
        if hasattr(torch.backends, 'mps'):
            print(f"    MPS (Apple M1/M2) available: {torch.backends.mps.is_available()}")
            if torch.backends.mps.is_available():
                print(f"    MPS built: {torch.backends.mps.is_built()}")
        else:
            print(f"    MPS (Apple M1/M2): Not supported in this PyTorch version")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import xgboost as xgb
        print(f"  ✓ xgboost {xgb.__version__}")
    except ImportError as e:
        print(f"  ✗ xgboost: {e}")
        return False
    
    try:
        import scipy
        print(f"  ✓ scipy {scipy.__version__}")
    except ImportError as e:
        print(f"  ✗ scipy: {e}")
        return False
    
    return True


def test_local_modules():
    """Test local module imports"""
    print("\nTesting local modules...")
    
    try:
        from vae_kde_augmenter import VAEKDEAugmenter, create_augmenter_for_mmli
        print("  ✓ vae_kde_augmenter")
    except Exception as e:
        print(f"  ✗ vae_kde_augmenter: {e}")
        traceback.print_exc()
        return False
    
    try:
        from stability_validator import MolecularStabilityValidator
        print("  ✓ stability_validator")
    except Exception as e:
        print(f"  ✗ stability_validator: {e}")
        traceback.print_exc()
        return False
    
    try:
        from simple_regression import train_baseline, train_with_synthetic
        print("  ✓ simple_regression")
    except Exception as e:
        print(f"  ✗ simple_regression: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_vae_generation():
    """Test VAE sample generation with synthetic data"""
    print("\nTesting VAE sample generation...")
    
    import numpy as np
    import pandas as pd
    from vae_kde_augmenter import create_augmenter_for_mmli
    
    # Create synthetic MMLI-like data (43 samples, 10 features)
    np.random.seed(13)
    n_samples = 43
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features) * 10 + 50,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.rand(n_samples) * 100, name='T80')
    
    print(f"  Created test data: {n_samples} samples, {n_features} features")
    
    # Generate 20 synthetic samples
    augmenter = create_augmenter_for_mmli(n_samples)
    X_aug, y_aug, is_synthetic = augmenter.augment_dataset(X, y, n_synthetic_samples=20)
    
    n_generated = is_synthetic.sum()
    print(f"  Generated {n_generated} synthetic samples")
    
    if n_generated == 20:
        print("  ✓ VAE generation working correctly")
        return True
    else:
        print(f"  ✗ Expected 20 samples, got {n_generated}")
        return False


def test_stability_validation():
    """Test stability validation"""
    print("\nTesting stability validation...")
    
    import numpy as np
    import pandas as pd
    from stability_validator import MolecularStabilityValidator
    
    # Create test data
    np.random.seed(13)
    n_samples = 43
    n_features = 10
    
    original_data = pd.DataFrame(
        np.random.randn(n_samples, n_features) * 10 + 50,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    original_data['T80'] = np.random.rand(n_samples) * 100
    
    # Create synthetic samples (some in range, some out)
    X_synthetic = pd.DataFrame(
        np.random.randn(10, n_features) * 10 + 50,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_synthetic = pd.Series(np.random.rand(10) * 100, name='T80')
    
    validator = MolecularStabilityValidator(original_data, 'T80')
    results = validator.validate_all(X_synthetic, y_synthetic)
    
    print(f"  Validation completed:")
    print(f"    - Physical pass rate: {results['summary']['physical_pass_rate']*100:.1f}%")
    print(f"    - Chemical pass rate: {results['summary']['chemical_pass_rate']*100:.1f}%")
    print(f"    - Statistical pass rate: {results['summary']['statistical_pass_rate']*100:.1f}%")
    print(f"    - Overall stability: {results['stability_rate']*100:.1f}%")
    
    print("  ✓ Stability validation working correctly")
    return True


def test_regression():
    """Test regression framework"""
    print("\nTesting regression framework...")
    
    import numpy as np
    import pandas as pd
    from simple_regression import train_baseline
    
    # Create test data
    np.random.seed(13)
    n_samples = 43
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    # Create correlated target
    y = pd.Series(X['feature_0'] * 2 + X['feature_1'] + np.random.randn(n_samples) * 0.1, name='T80')
    
    results = train_baseline(X, y, target_column='T80', model_type='linear', k_folds=3)
    
    r2 = results['aggregate_metrics']['mean_r2']
    print(f"  Baseline R²: {r2:.4f}")
    
    if r2 > 0.5:  # Should be high given the data generation
        print("  ✓ Regression framework working correctly")
        return True
    else:
        print(f"  ✗ Unexpected R² value: {r2}")
        return False


def main():
    print("="*60)
    print("MMLI Experiment - Dependency and Functionality Test")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages:")
        print("   pip install -r requirements.txt")
        all_passed = False
    
    # Test local modules
    if not test_local_modules():
        print("\n❌ Local module tests failed. Check file paths.")
        all_passed = False
    
    # Test VAE generation (only if previous tests passed)
    if all_passed:
        try:
            if not test_vae_generation():
                all_passed = False
        except Exception as e:
            print(f"\n❌ VAE generation test failed: {e}")
            traceback.print_exc()
            all_passed = False
    
    # Test stability validation
    if all_passed:
        try:
            if not test_stability_validation():
                all_passed = False
        except Exception as e:
            print(f"\n❌ Stability validation test failed: {e}")
            traceback.print_exc()
            all_passed = False
    
    # Test regression
    if all_passed:
        try:
            if not test_regression():
                all_passed = False
        except Exception as e:
            print(f"\n❌ Regression test failed: {e}")
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Ready to run experiment.")
        print("\nNext steps:")
        print("  1. Place your mmli.csv file in this directory")
        print("  2. Run: python run_experiment.py --data_path mmli.csv --target T80")
    else:
        print("❌ Some tests failed. Please fix issues before running experiment.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
