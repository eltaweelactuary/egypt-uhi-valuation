"""
Actuarial Loss Estimation - Utility Functions
Workers' Compensation Insurance Claims Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_data(data_type: str = 'train') -> pd.DataFrame:
    """
    Load train or test data from the data/raw directory.
    
    Args:
        data_type: 'train' or 'test'
    
    Returns:
        DataFrame with the loaded data
    """
    root = get_project_root()
    file_path = root / 'data' / 'raw' / f'{data_type}.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)


def save_submission(predictions: np.ndarray, filename: str = 'submission.csv'):
    """
    Save predictions to submission file.
    
    Args:
        predictions: Array of predicted values
        filename: Output filename
    """
    root = get_project_root()
    output_path = root / 'submissions' / filename
    
    # Load test data to get ClaimNumber
    test_df = load_data('test')
    
    submission = pd.DataFrame({
        'ClaimNumber': test_df['ClaimNumber'],
        'UltimateIncurredClaimCost': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    
    return submission


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        verbose: Print memory reduction info
    
    Returns:
        DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
