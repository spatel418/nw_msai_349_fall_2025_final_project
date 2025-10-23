import smogn
import pandas as pd

def smoter_augmentation(df, target_col, k=5, pert=0.02, samp_method='balance'):
    """
    Apply SMOTER using the smogn library.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe
    target_col : str
        Name of the target column
    k : int, default=5
        Number of nearest neighbors
    pert : float, default=0.02
        Perturbation/noise level
    samp_method : str, default='balance'
        Sampling method ('balance', 'extreme', or 'normal')
        
    Returns:
    --------
    df_new : pandas.DataFrame
        Augmented dataframe
    """
    # Apply SMOGN (which includes SMOTER)
    df_augmented = smogn.smoter(
        data=df,
        y=target_col,
        k=k,
        pert=pert,
        samp_method=samp_method
    )
    
    return df_augmented