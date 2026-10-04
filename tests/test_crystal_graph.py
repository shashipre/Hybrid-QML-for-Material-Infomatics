"""Tests for crystal graph construction from pymatgen structures."""
import pytest
import torch
import numpy as np


def _make_dummy_structure():
    """Create a simple pymatgen Structure for testing."""
    from pymatgen.core import Structure, Lattice

    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    return structure


class TestAtomFeaturizer:
    """Tests for AtomFeaturizer."""

    def test_output_shape(self):
        from src.data.featurizers import AtomFeaturizer

        structure = _make_dummy_structure()
        featurizer = AtomFeaturizer()
        features = featurizer.featurize(structure)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (len(structure), 92)

    def test_one_hot_encoding(self):
        from src.data.featurizers import AtomFeaturizer

        structure = _make_dummy_structure()
        featurizer = AtomFeaturizer()
        features = featurizer.featurize(structure)

        # Silicon is element 14 → index 13 in 0-indexed one-hot
        for i in range(len(structure)):
            assert features[i, 13] == 1.0
            assert features[i].sum() == 1.0  # exactly one-hot


class TestBondFeaturizer:
    """Tests for BondFeaturizer (Gaussian distance expansion)."""

    def test_output_shape(self):
        from src.data.featurizers import BondFeaturizer

        featurizer = BondFeaturizer(cutoff_radius=8.0, num_gaussians=41)
        distances = torch.tensor([1.0, 2.5, 4.0, 7.9])
        features = featurizer.expand(distances)

        assert features.shape == (4, 41)

    def test_values_positive(self):
        from src.data.featurizers import BondFeaturizer

        featurizer = BondFeaturizer(cutoff_radius=8.0, num_gaussians=41)
        distances = torch.tensor([1.0, 5.0])
        features = featurizer.expand(distances)

        # Gaussian expansion should produce positive values
        assert (features >= 0).all()
        assert (features <= 1).all()


class TestCrystalGraphConverter:
    """Tests for CrystalGraphConverter."""

    def test_basic_conversion(self):
        from src.data.crystal_graph import CrystalGraphConverter

        structure = _make_dummy_structure()
        converter = CrystalGraphConverter(cutoff_radius=8.0, max_neighbors=12)
        data = converter.convert(structure, target=42.0)

        # Check required fields
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "y")

        # Check shapes
        assert data.x.shape[0] == 2  # 2 atoms
        assert data.x.shape[1] == 92  # one-hot
        assert data.edge_index.shape[0] == 2  # (2, num_edges)
        assert data.edge_attr.shape[1] == 41  # Gaussian bins
        assert data.y.item() == pytest.approx(42.0)

    def test_edges_are_undirected(self):
        from src.data.crystal_graph import CrystalGraphConverter

        structure = _make_dummy_structure()
        converter = CrystalGraphConverter(cutoff_radius=8.0, max_neighbors=12)
        data = converter.convert(structure, target=0.0)

        # Number of edges should be even (undirected = both directions)
        assert data.edge_index.shape[1] % 2 == 0

    def test_no_target(self):
        from src.data.crystal_graph import CrystalGraphConverter

        structure = _make_dummy_structure()
        converter = CrystalGraphConverter()
        data = converter.convert(structure, target=None)

        assert data.y is None or data.y.item() == 0.0
