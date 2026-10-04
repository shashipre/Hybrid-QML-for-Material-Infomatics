import os
import copy
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from src.training.metrics import compute_metrics

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for training and evaluating models.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize Trainer.
        
        Args:
            model (nn.Module): The PyTorch model to train.
            config (Dict[str, Any]): Configuration dictionary.
            device (str): Device to run training on (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.config = config
        self.device = device
        
        train_config = config.get('training', {})
        lr = train_config.get('lr', 0.001)
        weight_decay = train_config.get('weight_decay', 1e-5)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        lr_patience = train_config.get('lr_patience', 10)
        lr_factor = train_config.get('lr_factor', 0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=lr_patience, factor=lr_factor
        )
        
        self.clip_val = train_config.get('grad_clip', 1.0)
        self.patience = train_config.get('patience', 20)
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        self.train_history = []
        self.val_history = []
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train the model for a single epoch.
        
        Args:
            train_loader (DataLoader): The training data loader.
            
        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch)
            
            target = batch.y.float().to(out.device)
            
            loss = self.criterion(out, target)
            loss.backward()
            
            if self.clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
                
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(1, num_batches)
        
    def evaluate(self, test_loader: torch.utils.data.DataLoader, mean: float, std: float) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            test_loader (DataLoader): The evaluation data loader.
            mean (float): Mean for un-standardizing predictions and targets.
            std (float): Standard deviation for un-standardizing.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                target = batch.y.float().to(out.device)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
        import numpy as np
        if not all_preds:
            return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
            
        preds_np = np.concatenate(all_preds, axis=0).flatten()
        targets_np = np.concatenate(all_targets, axis=0).flatten()
        
        preds_real = preds_np * std + mean
        targets_real = targets_np * std + mean
        
        return compute_metrics(preds_real, targets_real)
        
    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: Optional[torch.utils.data.DataLoader] = None, 
              target_mean: float = 0.0, target_std: float = 1.0) -> Dict[str, Any]:
        """
        Full training loop with early stopping and learning rate scheduling.
        
        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader, optional): Validation data loader.
            target_mean (float): Mean for evaluation un-standardization.
            target_std (float): Std for evaluation un-standardization.
            
        Returns:
            Dict[str, Any]: Dictionary with training history and best metrics.
        """
        epochs = self.config.get('training', {}).get('epochs', 100)
        best_metrics = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.train_history.append(train_loss)
            
            val_mae = float('inf')
            val_loss = float('inf')
            
            if val_loader is not None:
                metrics = self.evaluate(val_loader, target_mean, target_std)
                self.val_history.append(metrics)
                val_mae = metrics['mae']
                val_loss = val_mae  # Use MAE as surrogate for validation loss
                
                self.scheduler.step(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.patience_counter = 0
                    best_metrics = metrics
                else:
                    self.patience_counter += 1
            
            if epoch % 10 == 0 or epoch == 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.6f}")
                
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metrics': best_metrics
        }
        
    def save_checkpoint(self, path: str):
        """Save the model's state dictionary to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load the model's state dictionary from the specified path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Checkpoint loaded from {path}")
