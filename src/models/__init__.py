"""Graph Few-Shot Learning Models.

This module contains implementations of graph neural networks for few-shot learning,
including GCN encoders and prototypical networks.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
from torch_geometric.data import Data


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder for few-shot learning.
    
    Args:
        in_channels: Number of input node features
        hidden_channels: Number of hidden channels
        out_channels: Number of output channels (embeddings)
        num_layers: Number of GCN layers
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
        use_residual: Whether to use residual connections
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout = dropout
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass through the GCN encoder.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                x = self.dropout_layer(x)
            
            if residual is not None and x.size(-1) == residual.size(-1):
                x = x + residual
        
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder for few-shot learning.
    
    Args:
        in_channels: Number of input node features
        hidden_channels: Number of hidden channels
        out_channels: Number of output channels (embeddings)
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        )
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                )
            )
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    out_channels,
                    heads=1,
                    dropout=dropout,
                )
            )
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass through the GAT encoder.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot learning on graphs.
    
    This module implements the prototypical network approach where prototypes
    are computed as the mean of support embeddings for each class.
    
    Args:
        encoder: Graph encoder (GCNEncoder, GATEncoder, etc.)
        distance_metric: Distance metric for computing similarities
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        distance_metric: str = "euclidean",
    ) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.distance_metric = distance_metric
        
        if distance_metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        support_idx: Tensor,
        query_idx: Tensor,
        support_labels: Tensor,
        query_labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        """Forward pass for few-shot learning.
        
        Args:
            x: Node feature matrix
            edge_index: Edge connectivity
            support_idx: Indices of support nodes
            query_idx: Indices of query nodes
            support_labels: Labels of support nodes
            query_labels: Labels of query nodes
            
        Returns:
            Tuple of (loss, accuracy, metrics_dict)
        """
        # Get embeddings
        embeddings = self.encoder(x, edge_index)
        
        # Compute prototypes
        prototypes = self._compute_prototypes(
            embeddings[support_idx], support_labels
        )
        
        # Compute distances and predictions
        query_embeddings = embeddings[query_idx]
        distances = self._compute_distances(query_embeddings, prototypes)
        
        # Compute loss and accuracy
        loss = self._compute_loss(distances, query_labels, support_labels)
        accuracy = self._compute_accuracy(distances, query_labels, support_labels)
        
        # Additional metrics
        metrics = {
            "accuracy": accuracy,
            "loss": loss.item(),
            "num_support": len(support_idx),
            "num_query": len(query_idx),
            "num_classes": len(support_labels.unique()),
        }
        
        return loss, accuracy, metrics
    
    def _compute_prototypes(
        self, support_embeddings: Tensor, support_labels: Tensor
    ) -> Tensor:
        """Compute prototypes as mean of support embeddings per class.
        
        Args:
            support_embeddings: Embeddings of support nodes
            support_labels: Labels of support nodes
            
        Returns:
            Prototypes [num_classes, embedding_dim]
        """
        classes = support_labels.unique()
        prototypes = []
        
        for c in classes:
            mask = support_labels == c
            prototype = support_embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def _compute_distances(
        self, query_embeddings: Tensor, prototypes: Tensor
    ) -> Tensor:
        """Compute distances between query embeddings and prototypes.
        
        Args:
            query_embeddings: Embeddings of query nodes
            prototypes: Class prototypes
            
        Returns:
            Distances [num_query, num_classes]
        """
        if self.distance_metric == "euclidean":
            return torch.cdist(query_embeddings, prototypes)
        elif self.distance_metric == "cosine":
            # Normalize embeddings
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            # Cosine distance = 1 - cosine similarity
            return 1 - torch.mm(query_norm, proto_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _compute_loss(
        self,
        distances: Tensor,
        query_labels: Tensor,
        support_labels: Tensor,
    ) -> Tensor:
        """Compute prototypical loss.
        
        Args:
            distances: Distances between queries and prototypes
            query_labels: Labels of query nodes
            support_labels: Labels of support nodes
            
        Returns:
            Loss value
        """
        classes = support_labels.unique()
        class_to_idx = {c.item(): i for i, c in enumerate(classes)}
        
        # Convert query labels to indices
        query_indices = torch.tensor(
            [class_to_idx[label.item()] for label in query_labels],
            device=query_labels.device,
        )
        
        # Use negative distances as logits (closer = higher probability)
        logits = -distances
        return F.cross_entropy(logits, query_indices)
    
    def _compute_accuracy(
        self,
        distances: Tensor,
        query_labels: Tensor,
        support_labels: Tensor,
    ) -> float:
        """Compute classification accuracy.
        
        Args:
            distances: Distances between queries and prototypes
            query_labels: Labels of query nodes
            support_labels: Labels of support nodes
            
        Returns:
            Accuracy value
        """
        classes = support_labels.unique()
        class_to_idx = {c.item(): i for i, c in enumerate(classes)}
        
        # Convert query labels to indices
        query_indices = torch.tensor(
            [class_to_idx[label.item()] for label in query_labels],
            device=query_labels.device,
        )
        
        # Predictions are argmin of distances
        predictions = distances.argmin(dim=1)
        
        # Convert predictions back to class labels
        pred_labels = classes[predictions]
        
        return (pred_labels == query_labels).float().mean().item()


def create_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    **kwargs,
) -> PrototypicalNetwork:
    """Factory function to create different model types.
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'graphsage')
        in_channels: Number of input features
        hidden_channels: Number of hidden channels
        out_channels: Number of output channels
        **kwargs: Additional arguments for the encoder
        
    Returns:
        PrototypicalNetwork instance
    """
    if model_type.lower() == "gcn":
        encoder = GCNEncoder(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type.lower() == "gat":
        encoder = GATEncoder(in_channels, hidden_channels, out_channels, **kwargs)
    elif model_type.lower() == "graphsage":
        # Note: GraphSAGE implementation would go here
        raise NotImplementedError("GraphSAGE encoder not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return PrototypicalNetwork(encoder)
