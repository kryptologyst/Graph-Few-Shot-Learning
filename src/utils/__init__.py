"""Utility functions for graph few-shot learning."""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif config_path.suffix == ".json":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def setup_logging(log_dir: Union[str, Path], log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to store logs
        log_level: Logging level
    """
    import logging
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )


def create_directories(base_dir: Union[str, Path]) -> Dict[str, Path]:
    """Create necessary directories for the project.
    
    Args:
        base_dir: Base directory for the project
        
    Returns:
        Dictionary mapping directory names to paths
    """
    base_dir = Path(base_dir)
    
    directories = {
        "data": base_dir / "data",
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
        "assets": base_dir / "assets",
        "configs": base_dir / "configs",
        "results": base_dir / "results",
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def set_style() -> None:
    """Set matplotlib and seaborn styles."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    # Set default figure size
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history["train_losses"], label="Train Loss", color="blue")
    ax1.plot(history["val_losses"], label="Val Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history["train_accs"], label="Train Acc", color="blue")
    ax2.plot(history["val_accs"], label="Val Acc", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: torch.nn.Module) -> str:
    """Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary string
    """
    total_params = count_parameters(model)
    
    summary = []
    summary.append("=" * 50)
    summary.append("MODEL SUMMARY")
    summary.append("=" * 50)
    summary.append(f"Total parameters: {total_params:,}")
    summary.append("")
    summary.append("Architecture:")
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            summary.append(f"  {name}: {module}")
    
    summary.append("=" * 50)
    
    return "\n".join(summary)


def save_results(
    results: Dict[str, Any],
    save_path: Union[str, Path],
    format: str = "json",
) -> None:
    """Save evaluation results to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
        format: File format ('json', 'yaml', 'csv')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "yaml":
        with open(save_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)
    elif format == "csv":
        # Flatten nested dictionaries for CSV
        flattened = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}_{subkey}"] = subvalue
            else:
                flattened[key] = value
        
        df = pd.DataFrame([flattened])
        df.to_csv(save_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(load_path: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from file.
    
    Args:
        load_path: Path to load results from
        
    Returns:
        Results dictionary
    """
    load_path = Path(load_path)
    
    if load_path.suffix == ".json":
        with open(load_path, "r") as f:
            return json.load(f)
    elif load_path.suffix in [".yaml", ".yml"]:
        with open(load_path, "r") as f:
            return yaml.safe_load(f)
    elif load_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(load_path)
        return df.to_dict("records")[0]
    else:
        raise ValueError(f"Unsupported file format: {load_path.suffix}")


def compare_models(
    results: Dict[str, Dict[str, Any]],
    metric: str = "mean_accuracy",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Compare multiple models on a given metric.
    
    Args:
        results: Dictionary mapping model names to results
        metric: Metric to compare
        save_path: Path to save the comparison plot
    """
    set_style()
    
    model_names = list(results.keys())
    metric_values = [results[name][metric] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values)
    plt.xlabel("Model")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Model Comparison: {metric.replace('_', ' ').title()}")
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """Print configuration in a readable format.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        "model",
        "data",
        "training",
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    model_config = config["model"]
    if "type" not in model_config:
        raise ValueError("Missing model type")
    
    if model_config["type"] not in ["gcn", "gat"]:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    # Validate data config
    data_config = config["data"]
    if "dataset" not in data_config:
        raise ValueError("Missing dataset name")
    
    # Validate training config
    training_config = config["training"]
    if "num_epochs" not in training_config:
        raise ValueError("Missing number of epochs")
    
    if training_config["num_epochs"] <= 0:
        raise ValueError("Number of epochs must be positive")
