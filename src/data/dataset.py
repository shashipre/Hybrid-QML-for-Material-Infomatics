import logging
import torch
import numpy as np
from typing import Tuple, Dict, Any, List
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from src.data.crystal_graph import CrystalGraphConverter

try:
    from matbench.bench import MatbenchBenchmark
except ImportError:
    MatbenchBenchmark = None
    logging.warning("Matbench not installed. Please install it using `pip install matbench`.")

logger = logging.getLogger(__name__)

class MatBenchExfoliationDataset:
    """
    Dataset wrapper for MatBench exfoliation energy task.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (dict): Configuration dict containing batch_size, cutoff_radius, etc.
        """
        if MatbenchBenchmark is None:
            raise ImportError("matbench is required to use this dataset. Run `pip install matbench`.")
            
        self.config = config
        data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
        self.batch_size = data_cfg.get("batch_size", config.get("batch_size", 16))
        
        cutoff_radius = data_cfg.get("cutoff_radius", config.get("cutoff_radius", 8.0))
        max_neighbors = data_cfg.get("max_neighbors", config.get("max_neighbors", 12))
        
        self.converter = CrystalGraphConverter(
            cutoff_radius=cutoff_radius, 
            max_neighbors=max_neighbors
        )
        
        task_name = data_cfg.get("task_name", "matbench_jdft2d")
        logger.info(f"Initializing MatbenchBenchmark for {task_name}...")
        self.mb = MatbenchBenchmark(subset=[task_name], autoload=False)
        self.task = list(self.mb.tasks)[0]
        self.task.load()
        
        # State variables for normalization
        self.target_mean = 0.0
        self.target_std = 1.0
        
        # Cache for converted PyG Data objects
        self.cache: Dict[str, List[Data]] = {}

    def get_num_folds(self) -> int:
        """Returns the number of folds in the dataset."""
        return len(self.task.folds) # Usually 5
        
    def _convert_and_cache(self, cache_key: str, inputs, outputs, standardize: bool = False):
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        data_list = []
        for struct, target in zip(inputs, outputs):
            data = self.converter.convert(struct, target)
            data_list.append(data)
            
        if standardize:
            targets = torch.tensor([data.y.item() for data in data_list])
            self.target_mean = targets.mean().item()
            self.target_std = targets.std().item()
            
            if self.target_std < 1e-6:
                self.target_std = 1.0
                
            for data in data_list:
                data.y = (data.y - self.target_mean) / self.target_std
        else:
            # Apply existing standardization if it's test data
            for data in data_list:
                data.y = (data.y - self.target_mean) / self.target_std
                
        self.cache[cache_key] = data_list
        return data_list

    def load_fold(self, fold_index: int) -> Tuple[DataLoader, DataLoader]:
        """
        Loads the train and test DataLoaders for a specific fold.
        
        Args:
            fold_index (int): The fold to load.
            
        Returns:
            Tuple[DataLoader, DataLoader]: Train loader and Test loader.
        """
        logger.info(f"Loading fold {fold_index}...")
        
        train_inputs, train_outputs = self.task.get_train_and_val_data(fold_index)
        test_inputs, test_outputs = self.task.get_test_data(fold_index, include_target=True)
        
        train_cache_key = f"train_fold_{fold_index}"
        test_cache_key = f"test_fold_{fold_index}"
        
        train_data = self._convert_and_cache(train_cache_key, train_inputs, train_outputs, standardize=True)
        test_data = self._convert_and_cache(test_cache_key, test_inputs, test_outputs, standardize=False)
        
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False
        )
        
        return train_loader, test_loader
        
    def record_results(self, fold_index: int, predictions: np.ndarray):
        """
        Records the predictions back to the Matbench task for evaluation.
        
        Args:
            fold_index (int): The fold index.
            predictions (np.ndarray): The predicted targets (must be un-standardized).
        """
        self.task.record(fold_index, predictions)
        logger.info(f"Recorded results for fold {fold_index}.")
