import os
import logging
import numpy as np
import torch
from typing import Dict, Any

from src.data.dataset import MatBenchExfoliationDataset
from src.models.hybrid_model import HybridQuantumGNN
from src.models.classical_baseline import ClassicalGNN
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

class MatBenchRunner:
    """
    Runner for executing 5-fold cross-validation on the MatBench exfoliation dataset.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the runner.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary for data, model, and training.
        """
        self.config = config
        seed = self.config.get('training', {}).get('seed', 42)
        set_seed(seed)
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete 5-fold cross-validation evaluation.
        
        Returns:
            Dict[str, Any]: Dictionary containing individual fold metrics, and overall mean and std MAE.
        """
        logger.info("Starting MatBench 5-fold evaluation")
        dataset = MatBenchExfoliationDataset(self.config)
        
        results = []
        for fold in range(5):
            logger.info(f"--- Running Fold {fold} ---")
            fold_results = self._run_fold(dataset, fold)
            results.append(fold_results)
            
        maes = [r['mae'] for r in results if r and 'mae' in r]
        mean_mae = np.mean(maes) if maes else 0.0
        std_mae = np.std(maes) if maes else 0.0
        
        logger.info(f"MatBench Evaluation Complete. Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        
        return {
            'folds': results,
            'mean_mae': float(mean_mae),
            'std_mae': float(std_mae)
        }
        
    def run_single_fold(self, fold_index: int) -> Dict[str, Any]:
        """
        Run evaluation on a single fold, useful for debugging.
        
        Args:
            fold_index (int): Index of the fold to evaluate (0-4).
            
        Returns:
            Dict[str, Any]: Metrics for the evaluated fold.
        """
        logger.info(f"Starting single fold evaluation for Fold {fold_index}")
        dataset = MatBenchExfoliationDataset(self.config)
        return self._run_fold(dataset, fold_index)
        
    def _run_fold(self, dataset: MatBenchExfoliationDataset, fold_index: int) -> Dict[str, Any]:
        """
        Internal method to train and evaluate on a specific fold.
        
        Args:
            dataset (MatBenchExfoliationDataset): The complete dataset handling MatBench.
            fold_index (int): The fold to run.
            
        Returns:
            Dict[str, Any]: Metrics for the fold.
        """
        # Load train and test data loaders for the specific fold
        train_loader, test_loader = dataset.load_fold(fold_index)
        
        model_type = self.config.get('model_type', 'hybrid')
        # For hybrid model: quantum layer needs CPU, so keep model on CPU
        # For classical model: can use GPU
        if model_type == 'classical':
            requested_device = self.config.get('device', {}).get('gnn', 'cpu')
            device = requested_device if (requested_device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        else:
            device = 'cpu'  # Hybrid model manages GPU/CPU internally
        
        if model_type == 'hybrid':
            model = HybridQuantumGNN(self.config)
        elif model_type == 'classical':
            model = ClassicalGNN(self.config)
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' in config. Use 'hybrid' or 'classical'.")
            
        model = model.to(device)
        
        trainer = Trainer(model, self.config, device=device)
        
        target_mean = getattr(dataset, 'target_mean', 0.0)
        target_std = getattr(dataset, 'target_std', 1.0)
        
        logger.info(f"Training model on fold {fold_index}...")
        train_results = trainer.train(
            train_loader, 
            val_loader=test_loader, 
            target_mean=target_mean, 
            target_std=target_std
        )
        
        best_metrics = train_results.get('best_metrics', {})
        if not best_metrics:
            # If best_metrics isn't tracked properly or no validation loader was provided
            best_metrics = trainer.evaluate(test_loader, target_mean, target_std)
            
        logger.info(f"Fold {fold_index} final metrics: {best_metrics}")
        
        # NOTE: Recording predictions with MatBench's benchmark task instance
        # typically happens here, e.g., task.record(fold_index, predictions).
        # We assume `dataset` manages MatBench recording internally or metrics are returned.
        
        return best_metrics
