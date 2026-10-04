import torch
import torch.nn as nn
from src.models.gnn import GNNBackbone

class ClassicalGNN(nn.Module):
    """
    Classical baseline model that replaces the quantum layer with a standard
    feed-forward neural network of comparable parameter count.
    """
    def __init__(self, config: dict):
        super().__init__()
        gnn_cfg = config.get('gnn', {})
        mlp_cfg = config.get('mlp', {})
        q_cfg = config.get('quantum', {})
        
        n_qubits = q_cfg.get('n_qubits', 8)
        self.skip_connection = mlp_cfg.get('skip_connection', False)
        
        # 1. GNN Backbone
        self.gnn = GNNBackbone(
            node_input_dim=gnn_cfg.get('node_input_dim', 92),
            edge_input_dim=gnn_cfg.get('edge_input_dim', 41),
            hidden_dim=gnn_cfg.get('hidden_dim', 64),
            num_layers=gnn_cfg.get('num_layers', 3),
            output_dim=n_qubits
        )
        
        # 2. Classical Equivalent of Quantum Layer
        # The quantum layer has n_layers * n_qubits * 3 parameters
        # For a fair comparison, this classical layer should have a roughly similar capacity
        self.classical_surrogate = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.ReLU(),
            nn.Linear(n_qubits, n_qubits)
        )
        
        # 3. MLP Head
        input_dim = n_qubits * 2 if self.skip_connection else n_qubits
        mlp_hidden = mlp_cfg.get('hidden_dim', 32)
        dropout = mlp_cfg.get('dropout', 0.1)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Batch object containing .x, .edge_index, .edge_attr, .batch
            
        Returns:
            Predicted exfoliation energy, shape (batch_size,)
        """
        # GNN forward
        gnn_out = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Surrogate classical layer
        surrogate_out = self.classical_surrogate(gnn_out)
        
        # Combine representations if skip connection is enabled
        if self.skip_connection:
            combined = torch.cat([gnn_out, surrogate_out], dim=-1)
        else:
            combined = surrogate_out
            
        # MLP forward
        output = self.mlp(combined)
        
        # Return shape (batch_size,)
        return output.squeeze(-1)
