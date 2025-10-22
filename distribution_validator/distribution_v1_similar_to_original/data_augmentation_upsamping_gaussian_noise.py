import numpy as np
import pandas as pd

def add_gaussian_noise(df, target_column, n_samples_multiplier=2, noise_std_X=0.1, noise_std_y=0.1,):
    """
    Generate new samples by adding Gaussian noise to randomly selected original samples.
    
    Parameters:
    -----------
    X : array-like, shape (n_original, n_features)
        Original features
    y : array-like, shape (n_original,)
        Original targets
    n_samples_multiplier : float
        multiplier to the length of X to increase size of the dataset
    noise_std_X : float, default=0.1
        Standard deviation of noise to add to features
    noise_std_y : float, default=0.05
        Standard deviation of noise to add to targets
        
    Returns:
    --------
    X_new : array, shape (n_samples, n_features)
        New feature samples
    y_new : array, shape (n_samples,)
        New target samples
    """
    try:
        if n_samples_multiplier < 1:
            print("Multiplier should be greater than 1")
            return
        np.random.seed(13)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Randomly select base samples
        indices = np.random.choice(len(df), size=n_samples_multiplier * df.shape[0], replace=True)
        df_augmented = X.iloc[indices].copy()
        for col in X.columns:
            df_augmented[col] += np.random.normal(0, noise_std_X, n_samples_multiplier * X.shape[0])

        # Add noise to target
        y_augmented = y.iloc[indices] + np.random.normal(0, noise_std_y, n_samples_multiplier * X.shape[0])
        df_augmented[target_column] = y_augmented.values
        
        return df_augmented
    except Exception as e:
        print(repr(e))