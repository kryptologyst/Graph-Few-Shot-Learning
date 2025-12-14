"""Evaluation utilities for graph few-shot learning."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd

from ..models import PrototypicalNetwork
from ..data import GraphDataset


class Evaluator:
    """Evaluator for graph few-shot learning models."""
    
    def __init__(
        self,
        model: PrototypicalNetwork,
        dataset: GraphDataset,
        device: torch.device,
    ) -> None:
        """Initialize the evaluator.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_episodes(
        self,
        num_episodes: int = 100,
        split: str = "test",
        num_ways: int = 5,
        num_support: int = 1,
        num_query: int = 5,
    ) -> Dict[str, List[float]]:
        """Evaluate the model on multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            split: Which split to evaluate on
            num_ways: Number of classes per episode
            num_support: Number of support examples per class
            num_query: Number of query examples per class
            
        Returns:
            Dictionary of evaluation metrics
        """
        accuracies = []
        losses = []
        episode_metrics = []
        
        print(f"Evaluating on {num_episodes} episodes...")
        
        with torch.no_grad():
            progress_bar = tqdm(range(num_episodes), desc="Evaluation")
            
            for episode in progress_bar:
                try:
                    # Sample episode
                    support_idx, query_idx, support_labels, query_labels = self.dataset.sample_episode(
                        split=split,
                        num_ways=num_ways,
                        num_support=num_support,
                        num_query=num_query,
                    )
                    
                    # Move to device
                    support_idx = support_idx.to(self.device)
                    query_idx = query_idx.to(self.device)
                    support_labels = support_labels.to(self.device)
                    query_labels = query_labels.to(self.device)
                    
                    # Forward pass
                    loss, acc, metrics = self.model(
                        self.dataset.data.x.to(self.device),
                        self.dataset.data.edge_index.to(self.device),
                        support_idx,
                        query_idx,
                        support_labels,
                        query_labels,
                    )
                    
                    # Store metrics
                    accuracies.append(acc)
                    losses.append(loss.item())
                    episode_metrics.append(metrics)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "acc": f"{acc:.4f}",
                        "loss": f"{loss.item():.4f}",
                    })
                
                except ValueError as e:
                    # Skip episodes that don't have enough examples
                    print(f"Skipping episode {episode}: {e}")
                    continue
        
        # Compute summary statistics
        results = {
            "accuracies": accuracies,
            "losses": losses,
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "mean_loss": np.mean(losses),
            "std_loss": np.std(losses),
            "episode_metrics": episode_metrics,
        }
        
        print(f"Evaluation Results:")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"  Mean Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
        
        return results
    
    def evaluate_different_shots(
        self,
        num_ways: int = 5,
        shots: List[int] = [1, 2, 3, 5, 10],
        num_query: int = 5,
        num_episodes: int = 50,
        split: str = "test",
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate the model with different numbers of support shots.
        
        Args:
            num_ways: Number of classes per episode
            shots: List of support shots to evaluate
            num_query: Number of query examples per class
            num_episodes: Number of episodes per shot
            split: Which split to evaluate on
            
        Returns:
            Dictionary mapping shots to evaluation results
        """
        results = {}
        
        for shot in shots:
            print(f"Evaluating {shot}-shot learning...")
            
            shot_results = self.evaluate_episodes(
                num_episodes=num_episodes,
                split=split,
                num_ways=num_ways,
                num_support=shot,
                num_query=num_query,
            )
            
            results[shot] = {
                "mean_accuracy": shot_results["mean_accuracy"],
                "std_accuracy": shot_results["std_accuracy"],
                "mean_loss": shot_results["mean_loss"],
                "std_loss": shot_results["std_loss"],
            }
        
        return results
    
    def evaluate_different_ways(
        self,
        ways: List[int] = [2, 3, 5, 7, 10],
        num_support: int = 1,
        num_query: int = 5,
        num_episodes: int = 50,
        split: str = "test",
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate the model with different numbers of classes.
        
        Args:
            ways: List of number of classes to evaluate
            num_support: Number of support examples per class
            num_query: Number of query examples per class
            num_episodes: Number of episodes per way
            split: Which split to evaluate on
            
        Returns:
            Dictionary mapping ways to evaluation results
        """
        results = {}
        
        for way in ways:
            print(f"Evaluating {way}-way learning...")
            
            way_results = self.evaluate_episodes(
                num_episodes=num_episodes,
                split=split,
                num_ways=way,
                num_support=num_support,
                num_query=num_query,
            )
            
            results[way] = {
                "mean_accuracy": way_results["mean_accuracy"],
                "std_accuracy": way_results["std_accuracy"],
                "mean_loss": way_results["mean_loss"],
                "std_loss": way_results["std_loss"],
            }
        
        return results
    
    def get_embeddings(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray]:
        """Get node embeddings for visualization.
        
        Args:
            split: Which split to get embeddings for
            
        Returns:
            Tuple of (embeddings, labels)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get embeddings for all nodes
            embeddings = self.model.encoder(
                self.dataset.data.x.to(self.device),
                self.dataset.data.edge_index.to(self.device),
            )
            
            # Get labels for the specified split
            if split == "train":
                mask = self.dataset.data.train_mask
            elif split == "val":
                mask = self.dataset.data.val_mask
            elif split == "test":
                mask = self.dataset.data.test_mask
            else:
                mask = torch.ones(len(self.dataset.data.y), dtype=torch.bool)
            
            embeddings = embeddings[mask].cpu().numpy()
            labels = self.dataset.data.y[mask].cpu().numpy()
        
        return embeddings, labels
    
    def plot_embeddings(
        self,
        split: str = "test",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot t-SNE visualization of node embeddings.
        
        Args:
            split: Which split to visualize
            save_path: Path to save the plot
        """
        embeddings, labels = self.get_embeddings(split)
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.7,
        )
        plt.colorbar(scatter)
        plt.title(f"t-SNE Visualization of Node Embeddings ({split} split)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_shot_accuracy(
        self,
        shot_results: Dict[int, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot accuracy vs number of shots.
        
        Args:
            shot_results: Results from evaluate_different_shots
            save_path: Path to save the plot
        """
        shots = list(shot_results.keys())
        accuracies = [shot_results[shot]["mean_accuracy"] for shot in shots]
        stds = [shot_results[shot]["std_accuracy"] for shot in shots]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(shots, accuracies, yerr=stds, marker="o", capsize=5)
        plt.xlabel("Number of Support Shots")
        plt.ylabel("Accuracy")
        plt.title("Few-Shot Learning Performance vs Support Shots")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_way_accuracy(
        self,
        way_results: Dict[int, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot accuracy vs number of ways.
        
        Args:
            way_results: Results from evaluate_different_ways
            save_path: Path to save the plot
        """
        ways = list(way_results.keys())
        accuracies = [way_results[way]["mean_accuracy"] for way in ways]
        stds = [way_results[way]["std_accuracy"] for way in ways]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(ways, accuracies, yerr=stds, marker="o", capsize=5)
        plt.xlabel("Number of Ways (Classes)")
        plt.ylabel("Accuracy")
        plt.title("Few-Shot Learning Performance vs Number of Ways")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def generate_report(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("GRAPH FEW-SHOT LEARNING EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic metrics
        report.append("BASIC METRICS:")
        report.append(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        report.append(f"  Mean Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
        report.append(f"  Number of Episodes: {len(results['accuracies'])}")
        report.append("")
        
        # Accuracy distribution
        accuracies = results["accuracies"]
        report.append("ACCURACY DISTRIBUTION:")
        report.append(f"  Min: {min(accuracies):.4f}")
        report.append(f"  Max: {max(accuracies):.4f}")
        report.append(f"  Median: {np.median(accuracies):.4f}")
        report.append(f"  25th percentile: {np.percentile(accuracies, 25):.4f}")
        report.append(f"  75th percentile: {np.percentile(accuracies, 75):.4f}")
        report.append("")
        
        # Episode-level analysis
        episode_metrics = results["episode_metrics"]
        if episode_metrics:
            num_support = [m["num_support"] for m in episode_metrics]
            num_query = [m["num_query"] for m in episode_metrics]
            num_classes = [m["num_classes"] for m in episode_metrics]
            
            report.append("EPISODE ANALYSIS:")
            report.append(f"  Average support examples: {np.mean(num_support):.1f}")
            report.append(f"  Average query examples: {np.mean(num_query):.1f}")
            report.append(f"  Average classes per episode: {np.mean(num_classes):.1f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
        
        return report_text


def evaluate_model(
    model: PrototypicalNetwork,
    dataset: GraphDataset,
    device: Optional[torch.device] = None,
    num_episodes: int = 100,
) -> Evaluator:
    """Evaluate a model and return the evaluator.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        device: Device to evaluate on (auto-detected if None)
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Evaluator instance with results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize evaluator
    evaluator = Evaluator(model, dataset, device)
    
    # Run evaluation
    results = evaluator.evaluate_episodes(num_episodes=num_episodes)
    
    return evaluator
