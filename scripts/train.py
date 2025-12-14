#!/usr/bin/env python3
"""Main training script for Graph Few-Shot Learning."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import wandb
from omegaconf import OmegaConf

from src.models import create_model
from src.data import GraphDataset, create_synthetic_dataset, get_device, set_seed
from src.train import Trainer
from src.eval import Evaluator
from src.utils import (
    load_config,
    create_directories,
    setup_logging,
    print_config,
    validate_config,
    plot_training_history,
)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Graph Few-Shot Learning Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate the model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint to load for evaluation",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Create directories
    directories = create_directories(".")
    
    # Setup logging
    setup_logging(directories["logs"], config["logging"]["log_level"])
    
    # Set random seed
    if config["device"]["deterministic"]:
        set_seed(config["device"]["seed"])
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Print configuration
    print("Configuration:")
    print_config(config)
    print()
    
    # Load dataset
    print("Loading dataset...")
    if config["data"]["dataset"] == "synthetic":
        data = create_synthetic_dataset(
            num_nodes=config["data"]["synthetic"]["num_nodes"],
            num_features=config["data"]["synthetic"]["num_features"],
            num_classes=config["data"]["synthetic"]["num_classes"],
            edge_prob=config["data"]["synthetic"]["edge_prob"],
            random_seed=config["device"]["seed"],
        )
        dataset = GraphDataset(
            dataset_name="synthetic",
            data_dir=config["data"]["data_dir"],
            normalize_features=config["data"]["normalize_features"],
            make_undirected=config["data"]["make_undirected"],
        )
        dataset.data = data
        dataset.num_features = config["data"]["synthetic"]["num_features"]
        dataset.num_classes = config["data"]["synthetic"]["num_classes"]
    else:
        dataset = GraphDataset(
            dataset_name=config["data"]["dataset"],
            data_dir=config["data"]["data_dir"],
            normalize_features=config["data"]["normalize_features"],
            make_undirected=config["data"]["make_undirected"],
        )
    
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Nodes: {dataset.data.num_nodes}")
    print(f"Edges: {dataset.data.edge_index.size(1)}")
    print(f"Features: {dataset.num_features}")
    print(f"Classes: {dataset.num_classes}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=config["model"]["type"],
        in_channels=dataset.num_features,
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=config["model"]["out_channels"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        use_batch_norm=config["model"]["use_batch_norm"],
        use_residual=config["model"].get("use_residual", False),
        num_heads=config["model"].get("num_heads", 4),
    )
    
    # Set distance metric
    model.distance_metric = config["model"]["distance_metric"]
    
    print(f"Model: {config['model']['type'].upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # Initialize trainer
    trainer = Trainer(model, dataset, device, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Initialize wandb if enabled
    if config["logging"]["use_wandb"]:
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            config=config,
            name=f"{config['model']['type']}_{config['data']['dataset']}",
        )
    
    if args.eval_only:
        # Evaluation only
        print("Running evaluation...")
        evaluator = Evaluator(model, dataset, device)
        
        # Basic evaluation
        results = evaluator.evaluate_episodes(
            num_episodes=config["evaluation"]["num_episodes"],
            split="test",
            num_ways=config["few_shot"]["num_ways"],
            num_support=config["few_shot"]["num_support"],
            num_query=config["few_shot"]["num_query"],
        )
        
        # Shot evaluation
        shot_results = evaluator.evaluate_different_shots(
            num_ways=config["few_shot"]["num_ways"],
            shots=config["evaluation"]["shots_to_evaluate"],
            num_query=config["few_shot"]["num_query"],
            num_episodes=50,
            split="test",
        )
        
        # Way evaluation
        way_results = evaluator.evaluate_different_ways(
            ways=config["evaluation"]["ways_to_evaluate"],
            num_support=config["few_shot"]["num_support"],
            num_query=config["few_shot"]["num_query"],
            num_episodes=50,
            split="test",
        )
        
        # Generate plots
        if config["evaluation"]["save_plots"]:
            evaluator.plot_shot_accuracy(shot_results, "assets/plots/shot_accuracy.png")
            evaluator.plot_way_accuracy(way_results, "assets/plots/way_accuracy.png")
            evaluator.plot_embeddings("test", "assets/plots/embeddings.png")
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"Mean Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
        
        print("\nShot Results:")
        for shot, result in shot_results.items():
            print(f"  {shot}-shot: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
        
        print("\nWay Results:")
        for way, result in way_results.items():
            print(f"  {way}-way: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    else:
        # Training
        print("Starting training...")
        history = trainer.train(config["training"]["num_epochs"])
        
        # Plot training history
        plot_training_history(history, "assets/plots/training_history.png")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        evaluator = Evaluator(model, dataset, device)
        
        results = evaluator.evaluate_episodes(
            num_episodes=config["evaluation"]["num_episodes"],
            split="test",
            num_ways=config["few_shot"]["num_ways"],
            num_support=config["few_shot"]["num_support"],
            num_query=config["few_shot"]["num_query"],
        )
        
        print(f"\nFinal Test Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        
        # Generate comprehensive evaluation
        shot_results = evaluator.evaluate_different_shots(
            num_ways=config["few_shot"]["num_ways"],
            shots=config["evaluation"]["shots_to_evaluate"],
            num_query=config["few_shot"]["num_query"],
            num_episodes=50,
            split="test",
        )
        
        way_results = evaluator.evaluate_different_ways(
            ways=config["evaluation"]["ways_to_evaluate"],
            num_support=config["few_shot"]["num_support"],
            num_query=config["few_shot"]["num_query"],
            num_episodes=50,
            split="test",
        )
        
        # Generate plots
        if config["evaluation"]["save_plots"]:
            evaluator.plot_shot_accuracy(shot_results, "assets/plots/shot_accuracy.png")
            evaluator.plot_way_accuracy(way_results, "assets/plots/way_accuracy.png")
            evaluator.plot_embeddings("test", "assets/plots/embeddings.png")
        
        # Save results
        final_results = {
            "test_accuracy": results["mean_accuracy"],
            "test_std": results["std_accuracy"],
            "shot_results": shot_results,
            "way_results": way_results,
            "config": config,
        }
        
        import json
        with open("assets/results.json", "w") as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Final test accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    
    # Close wandb if enabled
    if config["logging"]["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
