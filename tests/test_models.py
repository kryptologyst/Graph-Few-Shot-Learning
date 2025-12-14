"""Tests for Graph Few-Shot Learning."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import GCNEncoder, GATEncoder, PrototypicalNetwork, create_model
from src.data import FewShotSampler, GraphDataset, create_synthetic_dataset, get_device, set_seed
from src.utils import load_config, validate_config


class TestGCNEncoder:
    """Test GCN encoder."""
    
    def test_gcn_encoder_creation(self):
        """Test GCN encoder creation."""
        encoder = GCNEncoder(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_layers=2,
        )
        
        assert encoder.num_layers == 2
        assert len(encoder.convs) == 2
        assert encoder.convs[0].in_channels == 64
        assert encoder.convs[0].out_channels == 32
        assert encoder.convs[1].in_channels == 32
        assert encoder.convs[1].out_channels == 16
    
    def test_gcn_encoder_forward(self):
        """Test GCN encoder forward pass."""
        encoder = GCNEncoder(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_layers=2,
        )
        
        # Create dummy data
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass
        output = encoder(x, edge_index)
        
        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGATEncoder:
    """Test GAT encoder."""
    
    def test_gat_encoder_creation(self):
        """Test GAT encoder creation."""
        encoder = GATEncoder(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_layers=2,
            num_heads=4,
        )
        
        assert encoder.num_layers == 2
        assert len(encoder.convs) == 2
        assert encoder.convs[0].heads == 4
        assert encoder.convs[1].heads == 1  # Last layer has 1 head
    
    def test_gat_encoder_forward(self):
        """Test GAT encoder forward pass."""
        encoder = GATEncoder(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            num_layers=2,
            num_heads=4,
        )
        
        # Create dummy data
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass
        output = encoder(x, edge_index)
        
        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestPrototypicalNetwork:
    """Test Prototypical Network."""
    
    def test_prototypical_network_creation(self):
        """Test Prototypical Network creation."""
        encoder = GCNEncoder(64, 32, 16)
        model = PrototypicalNetwork(encoder)
        
        assert model.encoder == encoder
        assert model.distance_metric == "euclidean"
    
    def test_prototypical_network_forward(self):
        """Test Prototypical Network forward pass."""
        encoder = GCNEncoder(64, 32, 16)
        model = PrototypicalNetwork(encoder)
        
        # Create dummy data
        x = torch.randn(100, 64)
        edge_index = torch.randint(0, 100, (2, 200))
        support_idx = torch.tensor([0, 1, 2, 3, 4])
        query_idx = torch.tensor([5, 6, 7, 8, 9])
        support_labels = torch.tensor([0, 0, 1, 1, 2])
        query_labels = torch.tensor([0, 1, 1, 2, 2])
        
        # Forward pass
        loss, acc, metrics = model(
            x, edge_index, support_idx, query_idx, support_labels, query_labels
        )
        
        assert isinstance(loss, torch.Tensor)
        assert isinstance(acc, float)
        assert isinstance(metrics, dict)
        assert 0 <= acc <= 1
        assert loss.item() >= 0


class TestFewShotSampler:
    """Test Few-Shot Sampler."""
    
    def test_sampler_creation(self):
        """Test sampler creation."""
        sampler = FewShotSampler(
            num_ways=5,
            num_support=1,
            num_query=5,
        )
        
        assert sampler.num_ways == 5
        assert sampler.num_support == 1
        assert sampler.num_query == 5
    
    def test_sample_episode(self):
        """Test episode sampling."""
        sampler = FewShotSampler(
            num_ways=3,
            num_support=2,
            num_query=3,
        )
        
        # Create dummy labels
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        
        # Sample episode
        support_idx, query_idx, support_labels, query_labels = sampler.sample_episode(
            labels, split="train"
        )
        
        assert len(support_idx) == 6  # 3 ways * 2 support
        assert len(query_idx) == 9   # 3 ways * 3 query
        assert len(support_labels) == 6
        assert len(query_labels) == 9
        
        # Check that all support and query indices are unique
        all_indices = torch.cat([support_idx, query_idx])
        assert len(torch.unique(all_indices)) == len(all_indices)


class TestSyntheticDataset:
    """Test synthetic dataset creation."""
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        data = create_synthetic_dataset(
            num_nodes=100,
            num_features=32,
            num_classes=5,
            edge_prob=0.1,
            random_seed=42,
        )
        
        assert data.num_nodes == 100
        assert data.x.shape == (100, 32)
        assert data.y.shape == (100,)
        assert data.edge_index.shape[0] == 2
        assert data.train_mask.shape == (100,)
        assert data.val_mask.shape == (100,)
        assert data.test_mask.shape == (100,)
        
        # Check that masks are mutually exclusive
        train_val_overlap = torch.logical_and(data.train_mask, data.val_mask).sum()
        train_test_overlap = torch.logical_and(data.train_mask, data.test_mask).sum()
        val_test_overlap = torch.logical_and(data.val_mask, data.test_mask).sum()
        
        assert train_val_overlap == 0
        assert train_test_overlap == 0
        assert val_test_overlap == 0


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Check that random numbers are deterministic
        torch.manual_seed(42)
        np.random.seed(42)
        
        rand1 = torch.randn(10)
        rand2 = np.random.randn(10)
        
        set_seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        rand3 = torch.randn(10)
        rand4 = np.random.randn(10)
        
        assert torch.allclose(rand1, rand3)
        assert np.allclose(rand2, rand4)


class TestConfig:
    """Test configuration handling."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            "model": {"type": "gcn"},
            "data": {"dataset": "Cora"},
            "training": {"num_epochs": 100},
        }
        
        # Should not raise exception
        validate_config(valid_config)
        
        # Invalid config - missing model type
        invalid_config = {
            "model": {},
            "data": {"dataset": "Cora"},
            "training": {"num_epochs": 100},
        }
        
        with pytest.raises(ValueError):
            validate_config(invalid_config)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training."""
        # Create synthetic dataset
        data = create_synthetic_dataset(
            num_nodes=50,
            num_features=16,
            num_classes=3,
            edge_prob=0.2,
            random_seed=42,
        )
        
        # Create dataset wrapper
        dataset = GraphDataset("synthetic", "data")
        dataset.data = data
        dataset.num_features = 16
        dataset.num_classes = 3
        
        # Create model
        model = create_model(
            model_type="gcn",
            in_channels=16,
            hidden_channels=8,
            out_channels=4,
        )
        
        # Test forward pass
        support_idx, query_idx, support_labels, query_labels = dataset.sample_episode(
            split="train",
            num_ways=3,
            num_support=1,
            num_query=2,
        )
        
        device = get_device()
        model = model.to(device)
        
        loss, acc, metrics = model(
            data.x.to(device),
            data.edge_index.to(device),
            support_idx.to(device),
            query_idx.to(device),
            support_labels.to(device),
            query_labels.to(device),
        )
        
        assert isinstance(loss, torch.Tensor)
        assert isinstance(acc, float)
        assert isinstance(metrics, dict)
        assert 0 <= acc <= 1
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])
