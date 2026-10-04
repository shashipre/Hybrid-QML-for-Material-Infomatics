"""Tests for the full hybrid quantum GNN model."""
import pytest
import torch
from torch_geometric.data import Data, Batch


def _make_dummy_batch(batch_size: int = 2, num_nodes_per_graph: int = 3):
    """Create a dummy PyG Batch for testing."""
    data_list = []
    for _ in range(batch_size):
        n = num_nodes_per_graph
        x = torch.randn(n, 92)  # 92-dim one-hot-like node features
        # Simple chain graph: 0-1, 1-2, ..., plus reverse edges
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.randn(len(src), 41)  # 41-dim Gaussian expansion
        y = torch.tensor(100.0)  # dummy target
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(data_list)


def _get_config():
    """Get a small config for testing."""
    return {
        'gnn': {
            'node_input_dim': 92,
            'edge_input_dim': 41,
            'hidden_dim': 16,
            'num_layers': 2,
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
    }


class TestHybridQuantumGNN:
    """Tests for the hybrid quantum-classical GNN."""

    @pytest.fixture
    def model(self):
        from src.models.hybrid_model import HybridQuantumGNN
        return HybridQuantumGNN(_get_config())

    def test_forward_shape(self, model):
        """Output should be (batch_size,)."""
        batch = _make_dummy_batch(batch_size=2)
        out = model(batch)
        assert out.shape == (2,)

    def test_backward_pass(self, model):
        """Loss backward should produce non-zero gradients on all parameter groups."""
        batch = _make_dummy_batch(batch_size=2)
        out = model(batch)
        loss = torch.nn.functional.mse_loss(out, torch.zeros(2))
        loss.backward()

        # Check GNN parameters have gradients
        for name, param in model.gnn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"GNN param {name} has no gradient"

        # Check quantum layer weights have gradients
        assert model.quantum.weights.grad is not None, "Quantum weights have no gradient"

        # Check MLP parameters have gradients
        for name, param in model.mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"MLP param {name} has no gradient"

    def test_skip_connection_disabled(self):
        """Model should work with skip_connection=False."""
        from src.models.hybrid_model import HybridQuantumGNN

        config = _get_config()
        config['mlp']['skip_connection'] = False
        model = HybridQuantumGNN(config)

        batch = _make_dummy_batch(batch_size=2)
        out = model(batch)
        assert out.shape == (2,)


class TestClassicalGNN:
    """Tests for the classical baseline GNN."""

    @pytest.fixture
    def model(self):
        from src.models.classical_baseline import ClassicalGNN
        return ClassicalGNN(_get_config())

    def test_forward_shape(self, model):
        """Output should be (batch_size,)."""
        batch = _make_dummy_batch(batch_size=2)
        out = model(batch)
        assert out.shape == (2,)

    def test_backward_pass(self, model):
        """Loss backward should produce gradients."""
        batch = _make_dummy_batch(batch_size=2)
        out = model(batch)
        loss = torch.nn.functional.mse_loss(out, torch.zeros(2))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Param {name} has no gradient"
