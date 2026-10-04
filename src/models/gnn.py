import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool

class GNNBackbone(nn.Module):
    """
    CGCNN-style Graph Neural Network backbone.
    Extracts features from crystal graphs and compresses them to a latent vector
    compatible with the number of qubits in the quantum layer.
    """
    def __init__(
        self,
        node_input_dim: int = 92,
        edge_input_dim: int = 41,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 8
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Initial node feature projection
        self.embedding = nn.Linear(node_input_dim, hidden_dim)

        # CGConv layers
        self.convs = nn.ModuleList([
            CGConv(channels=hidden_dim, dim=edge_input_dim, batch_norm=True)
            for _ in range(num_layers)
        ])

        # Final projection to output_dim (qubit count)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features, shape (num_nodes, node_input_dim)
            edge_index: Edge indices, shape (2, num_edges)
            edge_attr: Edge features, shape (num_edges, edge_input_dim)
            batch: Batch vector, shape (num_nodes) indicating graph assignment
            
        Returns:
            Latent graph representations, shape (batch_size, output_dim)
        """
        x = self.embedding(x)

        for conv in self.convs:
            # CGConv expects (x, edge_index, edge_attr)
            out = conv(x, edge_index, edge_attr)
            out = self.relu(out)
            x = out

        # Global pooling (graph-level readout)
        x = global_mean_pool(x, batch)

        # Compress to output_dim
        x = self.fc_out(x)

        return x
