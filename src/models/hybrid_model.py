import torch
import torch.nn as nn
from src.models.gnn import GNNBackbone
from src.models.quantum_layer import QuantumCircuitLayer

class HybridQuantumGNN(nn.Module):
    """
    Hybrid Quantum-Classical Graph Neural Network.
    Combines a CGCNN backbone, a PennyLane VQC, and an MLP head.
    """
    def __init__(self, config: dict):
        super().__init__()
        gnn_cfg = config.get('gnn', {})
        q_cfg = config.get('quantum', {})
        mlp_cfg = config.get('mlp', {})
        
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
        
        # 2. Quantum Layer
        self.quantum = QuantumCircuitLayer(
            n_qubits=n_qubits,
            n_layers=q_cfg.get('n_layers', 3),
            diff_method=q_cfg.get('diff_method', 'backprop')
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
        
        # Move to CPU for quantum layer if needed (keeping gradients)
        q_out = self.quantum(gnn_out)
        
        # Combine representations if skip connection is enabled
        if self.skip_connection:
            combined = torch.cat([gnn_out, q_out], dim=-1)
        else:
            combined = q_out
            
        # MLP forward
        combined = combined.to(next(self.mlp.parameters()).device)
        output = self.mlp(combined)
        
        # Return shape (batch_size,)
        return output.squeeze(-1)
