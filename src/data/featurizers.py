import torch
import numpy as np
from typing import List, Union

def one_hot_encoding(value: int, num_classes: int) -> List[int]:
    """
    Returns a one-hot encoded vector for the given value.
    
    Args:
        value (int): The value to encode (0-indexed).
        num_classes (int): Total number of classes.
        
    Returns:
        List[int]: One-hot encoded list of length `num_classes`.
    """
    encoding = [0] * num_classes
    if 0 <= value < num_classes:
        encoding[value] = 1
    return encoding

class AtomFeaturizer:
    """
    Featurizes a pymatgen structure's atoms into a node feature matrix.
    Uses one-hot encoding of the atomic number.
    """
    def __init__(self, num_elements: int = 92):
        self.num_elements = num_elements
        
    def __call__(self, structure) -> torch.Tensor:
        """
        Args:
            structure (pymatgen.core.Structure): The crystal structure.
            
        Returns:
            torch.Tensor: Node feature matrix of shape (n_atoms, num_elements).
        """
        features = []
        for site in structure:
            # pymatgen Element Z is 1-indexed. Z=1 is H.
            # We map Z=1 to index 0.
            z = site.specie.Z - 1
            features.append(one_hot_encoding(z, self.num_elements))
            
        return torch.tensor(features, dtype=torch.float32)

    def featurize(self, structure) -> torch.Tensor:
        """Alias for __call__."""
        return self(structure)

class BondFeaturizer:
    """
    Featurizes distances between atom pairs using Gaussian distance expansion.
    """
    def __init__(self, cutoff_radius: float = 8.0, num_gaussians: int = 41):
        self.cutoff_radius = cutoff_radius
        self.num_gaussians = num_gaussians
        
        # Initialize centers and the coefficient for the Gaussian
        self.centers = torch.linspace(0, cutoff_radius, num_gaussians)
        step = self.centers[1] - self.centers[0]
        self.coeff = 0.5 / (step ** 2)

    def expand(self, distances: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Alias for __call__."""
        return self(distances)
        
    def __call__(self, distances: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            distances: 1D array/tensor of edge distances.
            
        Returns:
            torch.Tensor: Edge feature matrix of shape (n_edges, num_gaussians).
        """
        if not isinstance(distances, torch.Tensor):
            distances = torch.tensor(distances, dtype=torch.float32)
            
        # distances: (n_edges, 1)
        distances = distances.view(-1, 1)
        # centers: (1, num_gaussians)
        centers = self.centers.view(1, -1)
        
        # Gaussian expansion
        expansion = torch.exp(-self.coeff * (distances - centers) ** 2)
        return expansion
