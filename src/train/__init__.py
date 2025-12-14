"""Training utilities for graph few-shot learning."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torchmetrics import Accuracy, F1Score

from ..models import PrototypicalNetwork
from ..data import GraphDataset, FewShotSampler


class Trainer:
    """Trainer for graph few-shot learning models."""
    
    def __init__(
        self,
        model: PrototypicalNetwork,
        dataset: GraphDataset,
        device: torch.device,
        config: Dict[str, Any],
    ) -> None:
        """Initialize the trainer.
        
        Args:
            model: The model to train
            dataset: The dataset to train on
            device: Device to train on
            config: Training configuration
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=config.get("patience", 10),
            verbose=True,
        )
        
        # Setup metrics
        self.train_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=config.get("num_classes", 7)),
            "f1": F1Score(task="multiclass", num_classes=config.get("num_classes", 7)),
        }
        
        self.val_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=config.get("num_classes", 7)),
            "f1": F1Score(task="multiclass", num_classes=config.get("num_classes", 7)),
        }
        
        # Move metrics to device
        for metric in self.train_metrics.values():
            metric.to(device)
        for metric in self.val_metrics.values():
            metric.to(device)
        
        # Training state
        self.best_val_acc = 0.0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_episodes = self.config.get("episodes_per_epoch", 100)
        
        progress_bar = tqdm(range(num_episodes), desc="Training")
        
        for episode in progress_bar:
            # Sample episode
            support_idx, query_idx, support_labels, query_labels = self.dataset.sample_episode(
                split="train",
                num_ways=self.config.get("num_ways", 5),
                num_support=self.config.get("num_support", 1),
                num_query=self.config.get("num_query", 5),
            )
            
            # Move to device
            support_idx = support_idx.to(self.device)
            query_idx = query_idx.to(self.device)
            support_labels = support_labels.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, acc, metrics = self.model(
                self.dataset.data.x.to(self.device),
                self.dataset.data.edge_index.to(self.device),
                support_idx,
                query_idx,
                support_labels,
                query_labels,
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_acc += acc
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
            })
        
        avg_loss = total_loss / num_episodes
        avg_acc = total_acc / num_episodes
        
        return {
            "train_loss": avg_loss,
            "train_acc": avg_acc,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        num_episodes = self.config.get("val_episodes", 50)
        
        with torch.no_grad():
            progress_bar = tqdm(range(num_episodes), desc="Validation")
            
            for episode in progress_bar:
                # Sample episode
                support_idx, query_idx, support_labels, query_labels = self.dataset.sample_episode(
                    split="val",
                    num_ways=self.config.get("num_ways", 5),
                    num_support=self.config.get("num_support", 1),
                    num_query=self.config.get("num_query", 5),
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
                
                # Update metrics
                total_loss += loss.item()
                total_acc += acc
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.4f}",
                })
        
        avg_loss = total_loss / num_episodes
        avg_acc = total_acc / num_episodes
        
        return {
            "val_loss": avg_loss,
            "val_acc": avg_acc,
        }
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics["val_acc"])
            
            # Store metrics
            self.train_losses.append(train_metrics["train_loss"])
            self.val_losses.append(val_metrics["val_loss"])
            self.train_accs.append(train_metrics["train_acc"])
            self.val_accs.append(val_metrics["val_acc"])
            
            # Log metrics
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["train_loss"],
                    "train_acc": train_metrics["train_acc"],
                    "val_loss": val_metrics["val_loss"],
                    "val_acc": val_metrics["val_acc"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                })
            
            # Save best model
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.save_checkpoint("best_model.pt")
                print(f"  New best validation accuracy: {self.best_val_acc:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(self.config.get("checkpoint_dir", "checkpoints")) / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        
        print(f"Loaded checkpoint from epoch {self.epoch} with val_acc {self.best_val_acc:.4f}")


def train_model(
    model: PrototypicalNetwork,
    dataset: GraphDataset,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Trainer:
    """Train a model with the given configuration.
    
    Args:
        model: The model to train
        dataset: The dataset to train on
        config: Training configuration
        device: Device to train on (auto-detected if None)
        
    Returns:
        Trained trainer instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize trainer
    trainer = Trainer(model, dataset, device, config)
    
    # Train
    trainer.train(config.get("num_epochs", 100))
    
    return trainer
