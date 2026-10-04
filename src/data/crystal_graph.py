import logging
import torch
from torch_geometric.data import Data
from typing import Optional

from src.data.featurizers import AtomFeaturizer, BondFeaturizer

logger = logging.getLogger(__name__)

class CrystalGraphConverter:
    """
    Converts a pymatgen Structure into a PyTorch Geometric Data object.
    """
    def __init__(self, cutoff_radius: float = 8.0, max_neighbors: int = 12):
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer(cutoff_radius=cutoff_radius)
        
    def convert(self, structure, target: Optional[float] = None) -> Data:
        """
        Converts the structure to a PyG Data object.
        
        Args:
            structure (pymatgen.core.Structure): The input crystal structure.
            target (float, optional): The target property value.
            
        Returns:
            torch_geometric.data.Data: PyG Data object containing x, edge_index, edge_attr, y.
        """
        # Node features
        x = self.atom_featurizer(structure)
        n_atoms = len(structure)
        
        # Find neighbors
        # get_all_neighbors returns a list of lists of neighbors for each site
        all_neighbors = structure.get_all_neighbors(self.cutoff_radius)
        
        src_indices = []
        dst_indices = []
        distances = []
        
        for i, neighbors in enumerate(all_neighbors):
            # Sort neighbors by distance
            neighbors = sorted(neighbors, key=lambda n: n.nn_distance)
            
            # Keep at most max_neighbors
            neighbors = neighbors[:self.max_neighbors]
            
            for n in neighbors:
                j = n.index
                distance = n.nn_distance
                
                # Add edge i -> j
                src_indices.append(i)
                dst_indices.append(j)
                distances.append(distance)
                
                # Also add edge j -> i (undirected graph)
                src_indices.append(j)
                dst_indices.append(i)
                distances.append(distance)
        
        if len(src_indices) == 0:
            logger.warning(f"No neighbors found within cutoff {self.cutoff_radius} for structure. Adding self-loops.")
            src_indices = list(range(n_atoms))
            dst_indices = list(range(n_atoms))
            distances = [0.0] * n_atoms
            
        # Convert to tensors
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        distances_tensor = torch.tensor(distances, dtype=torch.float32)
        
        # Remove duplicate edges to avoid double counting from adding j->i manually
        edges = set()
        unique_src = []
        unique_dst = []
        unique_dist = []
        for s, d, dist in zip(src_indices, dst_indices, distances):
            if (s, d) not in edges:
                edges.add((s, d))
                unique_src.append(s)
                unique_dst.append(d)
                unique_dist.append(dist)
                
        edge_index = torch.tensor([unique_src, unique_dst], dtype=torch.long)
        unique_dist_tensor = torch.tensor(unique_dist, dtype=torch.float32)
        
        # Edge features
        edge_attr = self.bond_featurizer(unique_dist_tensor)
        
        # Target
        y = torch.tensor([target], dtype=torch.float32) if target is not None else None
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
