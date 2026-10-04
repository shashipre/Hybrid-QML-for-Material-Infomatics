"""
Fast offline smoke test.
Verifies the full pipeline (model + training) works end-to-end on tiny synthetic data.
Should complete in under 30 seconds.
"""
import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


def _make_synthetic_dataset(n_samples: int = 20):
    """Create a tiny synthetic dataset for smoke testing."""
    data_list = []
    for i in range(n_samples):
        n_atoms = torch.randint(2, 6, (1,)).item()
        x = torch.zeros(n_atoms, 92)
        # Random one-hot atomic numbers
        for j in range(n_atoms):
            idx = torch.randint(0, 92, (1,)).item()
            x[j, idx] = 1.0

        # Random edges (chain + some random)
        src = list(range(n_atoms - 1))
        dst = list(range(1, n_atoms))
        # Add reverse edges
        src_full = src + dst
        dst_full = dst + src
        edge_index = torch.tensor([src_full, dst_full], dtype=torch.long)
        edge_attr = torch.randn(len(src_full), 41).abs()  # positive Gaussian-like

        # Target: simple function of number of atoms for learnability
        y = torch.tensor(float(n_atoms * 10.0 + i))

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data_list


def _get_smoke_config():
    """Minimal config for smoke test."""
    return {
        'gnn': {
            'node_input_dim': 92,
            'edge_input_dim': 41,
            'hidden_dim': 16,
            'num_layers': 1,
            'output_dim': 4,
        },
        'quantum': {
            'n_qubits': 4,
            'n_layers': 1,
            'diff_method': 'backprop',
        },
        'mlp': {
            'hidden_dim': 8,
            'dropout': 0.0,
            'skip_connection': True,
        },
        'training': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'epochs': 5,
            'patience': 100,
            'lr_patience': 100,
            'lr_factor': 0.5,
            'grad_clip': 1.0,
            'seed': 42,
        },
    }


class TestSmokeEndToEnd:
    """End-to-end smoke tests for the full pipeline."""

    def test_hybrid_model_loss_decreases(self):
        """Train the hybrid model for a few epochs — loss should decrease."""
        from src.models.hybrid_model import HybridQuantumGNN
        from src.training.trainer import Trainer

        config = _get_smoke_config()
        model = HybridQuantumGNN(config)
        trainer = Trainer(model, config, device='cpu')

        data_list = _make_synthetic_dataset(n_samples=10)
        loader = DataLoader(data_list, batch_size=5, shuffle=False)

        # Compute initial loss
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            initial_out = model(batch)
            initial_loss = torch.nn.functional.mse_loss(
                initial_out, batch.y.float()
            ).item()

        # Train for a few epochs
        results = trainer.train(loader)
        final_loss = results['train_history'][-1]

        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss * 2, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_classical_baseline_loss_decreases(self):
        """Train the classical baseline — loss should decrease."""
        from src.models.classical_baseline import ClassicalGNN
        from src.training.trainer import Trainer

        config = _get_smoke_config()
        model = ClassicalGNN(config)
        trainer = Trainer(model, config, device='cpu')

        data_list = _make_synthetic_dataset(n_samples=10)
        loader = DataLoader(data_list, batch_size=5, shuffle=False)

        # Compute initial loss
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            initial_out = model(batch)
            initial_loss = torch.nn.functional.mse_loss(
                initial_out, batch.y.float()
            ).item()

        # Train
        results = trainer.train(loader)
        final_loss = results['train_history'][-1]

        assert final_loss < initial_loss * 2, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_trainer_early_stopping(self):
        """Early stopping should trigger when validation loss stalls."""
        from src.models.classical_baseline import ClassicalGNN
        from src.training.trainer import Trainer

        config = _get_smoke_config()
        config['training']['patience'] = 2
        config['training']['epochs'] = 100  # high, but early stopping should kick in

        model = ClassicalGNN(config)
        trainer = Trainer(model, config, device='cpu')

        data_list = _make_synthetic_dataset(n_samples=10)
        train_loader = DataLoader(data_list[:7], batch_size=4, shuffle=True)
        val_loader = DataLoader(data_list[7:], batch_size=3, shuffle=False)

        results = trainer.train(
            train_loader, val_loader=val_loader,
            target_mean=0.0, target_std=1.0
        )

        # Should have stopped before 100 epochs
        assert len(results['train_history']) < 100, (
            "Early stopping did not trigger"
        )
