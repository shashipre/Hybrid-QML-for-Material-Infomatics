import numpy as np
from typing import Dict

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics: MAE, RMSE, R².
    
    Args:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): Ground truth values.
        
    Returns:
        Dict[str, float]: Dictionary containing MAE, RMSE, and R2 scores.
    """
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}
