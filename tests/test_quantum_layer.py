"""Tests for the PennyLane quantum circuit layer."""
import pytest
import torch


class TestQuantumCircuitLayer:
    """Tests for QuantumCircuitLayer."""

    @pytest.fixture
    def layer(self):
        from src.models.quantum_layer import QuantumCircuitLayer
        return QuantumCircuitLayer(n_qubits=4, n_layers=2, diff_method='backprop')

    def test_output_shape(self, layer):
        """Output shape should be (batch_size, n_qubits)."""
        x = torch.randn(3, 4)  # batch_size=3, n_qubits=4
        out = layer(x)
        assert out.shape == (3, 4)

    def test_output_range(self, layer):
        """PauliZ expectations should be in [-1, 1]."""
        x = torch.randn(2, 4)
        out = layer(x)
        assert (out >= -1.0 - 1e-6).all(), f"Min value: {out.min()}"
        assert (out <= 1.0 + 1e-6).all(), f"Max value: {out.max()}"

    def test_gradient_flow_to_weights(self, layer):
        """Gradients should flow to the circuit's trainable weights."""
        x = torch.randn(2, 4)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weights.grad is not None
        assert not torch.all(layer.weights.grad == 0), "Gradients are all zero"

    def test_gradient_flow_to_input(self, layer):
        """Gradients should flow back to the input tensor."""
        x = torch.randn(2, 4, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0), "Input gradients are all zero"

    def test_single_sample(self, layer):
        """Should handle a single-sample batch."""
        x = torch.randn(1, 4)
        out = layer(x)
        assert out.shape == (1, 4)

    def test_deterministic(self, layer):
        """Same input should produce same output (no stochastic measurement)."""
        x = torch.randn(2, 4)
        out1 = layer(x)
        out2 = layer(x)
        torch.testing.assert_close(out1, out2)
