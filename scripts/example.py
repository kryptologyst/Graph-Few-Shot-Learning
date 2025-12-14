#!/usr/bin/env python3
"""Simple example script demonstrating Graph Few-Shot Learning."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from src.models import create_model
from src.data import GraphDataset, create_synthetic_dataset, get_device, set_seed
from src.train import Trainer
from src.eval import Evaluator


def main():
    """Run a simple example of graph few-shot learning."""
    print("Graph Few-Shot Learning Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    data = create_synthetic_dataset(
        num_nodes=200,
        num_features=32,
        num_classes=5,
        edge_prob=0.05,
        random_seed=42,
    )
    
    # Create dataset wrapper
    dataset = GraphDataset("synthetic", "data")
    dataset.data = data
    dataset.num_features = 32
    dataset.num_classes = 5
    
    print(f"Dataset created:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {dataset.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    
    # Create model
    print("\nCreating GCN model...")
    model = create_model(
        model_type="gcn",
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=8,
        num_layers=2,
        dropout=0.3,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Training configuration
    config = {
        "model": {"type": "gcn"},
        "data": {"dataset": "synthetic"},
        "training": {
            "num_epochs": 20,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "patience": 5,
            "checkpoint_dir": "checkpoints",
        },
        "few_shot": {
            "num_ways": 3,
            "num_support": 1,
            "num_query": 3,
            "episodes_per_epoch": 50,
            "val_episodes": 20,
        },
        "device": {"seed": 42, "deterministic": True},
    }
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, dataset, device, config)
    
    # Train the model
    print("\nTraining model...")
    history = trainer.train(config["training"]["num_epochs"])
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluator = Evaluator(model, dataset, device)
    
    results = evaluator.evaluate_episodes(
        num_episodes=50,
        split="test",
        num_ways=3,
        num_support=1,
        num_query=3,
    )
    
    # Print results
    print("\nResults:")
    print(f"  Test Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"  Test Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
    
    # Test different numbers of support shots
    print("\nTesting different support shots...")
    shot_results = evaluator.evaluate_different_shots(
        num_ways=3,
        shots=[1, 2, 3, 5],
        num_query=3,
        num_episodes=20,
        split="test",
    )
    
    print("Shot Results:")
    for shot, result in shot_results.items():
        print(f"  {shot}-shot: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
