"""Streamlit demo for Graph Few-Shot Learning."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network

from src.models import create_model
from src.data import GraphDataset, create_synthetic_dataset, get_device, set_seed
from src.eval import Evaluator
from src.utils import load_config


def load_model_and_dataset(config_path: str):
    """Load model and dataset based on configuration."""
    config = load_config(config_path)
    
    # Set seed for reproducibility
    if config["device"]["deterministic"]:
        set_seed(config["device"]["seed"])
    
    # Load dataset
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
    
    # Create model
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
    
    model.distance_metric = config["model"]["distance_metric"]
    
    return model, dataset, config


def create_network_visualization(data, node_colors=None, title="Graph Visualization"):
    """Create an interactive network visualization."""
    # Convert to NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i, label=f"Node {i}")
    
    # Add edges
    edge_list = data.edge_index.t().numpy()
    for edge in edge_list:
        G.add_edge(int(edge[0]), int(edge[1]))
    
    # Create pyvis network
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes and edges
    for node in G.nodes():
        color = node_colors[node] if node_colors else "#97c2fc"
        net.add_node(node, label=f"Node {node}", color=color)
    
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    return net


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Graph Few-Shot Learning Demo",
        page_icon="🕸️",
        layout="wide",
    )
    
    st.title("🕸️ Graph Few-Shot Learning Demo")
    st.markdown("Interactive demonstration of few-shot learning on graph-structured data")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["gcn", "gat"],
        help="Choose the graph neural network architecture"
    )
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["Cora", "CiteSeer", "PubMed", "synthetic"],
        help="Choose the dataset to work with"
    )
    
    # Few-shot parameters
    st.sidebar.subheader("Few-Shot Parameters")
    num_ways = st.sidebar.slider("Number of Ways", 2, 10, 5)
    num_support = st.sidebar.slider("Support Shots", 1, 10, 1)
    num_query = st.sidebar.slider("Query Examples", 1, 20, 5)
    
    # Load model and dataset
    try:
        config_path = f"configs/{model_type}.yaml"
        model, dataset, config = load_model_and_dataset(config_path)
        
        # Update config for demo
        config["data"]["dataset"] = dataset_name
        config["few_shot"]["num_ways"] = num_ways
        config["few_shot"]["num_support"] = num_support
        config["few_shot"]["num_query"] = num_query
        
        st.sidebar.success("Model and dataset loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Few-Shot Learning", "Model Analysis", "Interactive Demo"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            stats = {
                "Dataset": dataset.dataset_name,
                "Nodes": dataset.data.num_nodes,
                "Edges": dataset.data.edge_index.size(1),
                "Features": dataset.num_features,
                "Classes": dataset.num_classes,
                "Model": model_type.upper(),
            }
            
            for key, value in stats.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("Class Distribution")
            labels = dataset.data.y.numpy()
            class_counts = pd.Series(labels).value_counts().sort_index()
            
            fig = px.bar(
                x=class_counts.index,
                y=class_counts.values,
                title="Node Class Distribution",
                labels={"x": "Class", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Graph visualization
        st.subheader("Graph Visualization")
        
        # Create node colors based on labels
        labels = dataset.data.y.numpy()
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set3
        node_colors = {}
        
        for i, label in enumerate(labels):
            color_idx = int(label) % len(colors)
            node_colors[i] = colors[color_idx]
        
        # Create network visualization
        net = create_network_visualization(
            dataset.data,
            node_colors=node_colors,
            title=f"{dataset.dataset_name} Graph"
        )
        
        # Save and display
        net.save_graph("temp_graph.html")
        with open("temp_graph.html", "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=500)
    
    with tab2:
        st.header("Few-Shot Learning")
        
        if st.button("Run Few-Shot Episode", type="primary"):
            with st.spinner("Running few-shot learning episode..."):
                # Sample episode
                support_idx, query_idx, support_labels, query_labels = dataset.sample_episode(
                    split="test",
                    num_ways=num_ways,
                    num_support=num_support,
                    num_query=num_query,
                )
                
                # Move to device
                device = get_device()
                model = model.to(device)
                
                # Run inference
                model.eval()
                with torch.no_grad():
                    loss, acc, metrics = model(
                        dataset.data.x.to(device),
                        dataset.data.edge_index.to(device),
                        support_idx.to(device),
                        query_idx.to(device),
                        support_labels.to(device),
                        query_labels.to(device),
                    )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{acc:.4f}")
                
                with col2:
                    st.metric("Loss", f"{loss:.4f}")
                
                with col3:
                    st.metric("Classes", num_ways)
                
                # Show episode details
                st.subheader("Episode Details")
                
                episode_df = pd.DataFrame({
                    "Node Type": ["Support"] * len(support_idx) + ["Query"] * len(query_idx),
                    "Node ID": support_idx.tolist() + query_idx.tolist(),
                    "Label": support_labels.tolist() + query_labels.tolist(),
                })
                
                st.dataframe(episode_df, use_container_width=True)
                
                # Visualize episode
                st.subheader("Episode Visualization")
                
                # Create episode visualization
                episode_colors = {}
                for i in range(dataset.data.num_nodes):
                    if i in support_idx:
                        episode_colors[i] = "#ff6b6b"  # Red for support
                    elif i in query_idx:
                        episode_colors[i] = "#4ecdc4"  # Teal for query
                    else:
                        episode_colors[i] = "#95a5a6"  # Gray for others
                
                episode_net = create_network_visualization(
                    dataset.data,
                    node_colors=episode_colors,
                    title="Few-Shot Episode"
                )
                
                episode_net.save_graph("temp_episode.html")
                with open("temp_episode.html", "r") as f:
                    episode_html = f.read()
                st.components.v1.html(episode_html, height=500)
    
    with tab3:
        st.header("Model Analysis")
        
        if st.button("Analyze Model Performance", type="primary"):
            with st.spinner("Analyzing model performance..."):
                device = get_device()
                evaluator = Evaluator(model, dataset, device)
                
                # Run evaluation
                results = evaluator.evaluate_episodes(
                    num_episodes=50,
                    split="test",
                    num_ways=num_ways,
                    num_support=num_support,
                    num_query=num_query,
                )
                
                # Display results
                st.subheader("Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Mean Accuracy",
                        f"{results['mean_accuracy']:.4f}",
                        delta=f"±{results['std_accuracy']:.4f}"
                    )
                
                with col2:
                    st.metric(
                        "Mean Loss",
                        f"{results['mean_loss']:.4f}",
                        delta=f"±{results['std_loss']:.4f}"
                    )
                
                # Accuracy distribution
                st.subheader("Accuracy Distribution")
                
                fig = px.histogram(
                    x=results["accuracies"],
                    nbins=20,
                    title="Accuracy Distribution Across Episodes",
                    labels={"x": "Accuracy", "y": "Count"},
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Shot analysis
                st.subheader("Shot Analysis")
                
                shot_results = evaluator.evaluate_different_shots(
                    num_ways=num_ways,
                    shots=[1, 2, 3, 5, 10],
                    num_query=num_query,
                    num_episodes=20,
                    split="test",
                )
                
                shots = list(shot_results.keys())
                accuracies = [shot_results[shot]["mean_accuracy"] for shot in shots]
                stds = [shot_results[shot]["std_accuracy"] for shot in shots]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=shots,
                    y=accuracies,
                    error_y=dict(type='data', array=stds),
                    mode='markers+lines',
                    name='Accuracy',
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title="Accuracy vs Number of Support Shots",
                    xaxis_title="Support Shots",
                    yaxis_title="Accuracy",
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Interactive Demo")
        
        st.markdown("""
        ### Interactive Few-Shot Learning
        
        This demo allows you to:
        - Select different model architectures (GCN, GAT)
        - Choose different datasets
        - Adjust few-shot learning parameters
        - Visualize the graph structure and episodes
        - Analyze model performance
        
        ### How Few-Shot Learning Works
        
        1. **Support Set**: A few labeled examples from each class
        2. **Query Set**: Unlabeled examples to classify
        3. **Prototype Computation**: Compute class prototypes from support examples
        4. **Classification**: Classify query examples based on distance to prototypes
        
        ### Key Features
        
        - **Episodic Learning**: Each episode is a mini-classification task
        - **Prototypical Networks**: Use mean of support examples as class prototypes
        - **Graph Structure**: Leverage graph connectivity for better representations
        - **Few-Shot Adaptation**: Learn to generalize from very few examples
        """)
        
        # Model comparison
        st.subheader("Model Comparison")
        
        if st.button("Compare Models", type="primary"):
            with st.spinner("Comparing models..."):
                # Load both models
                gcn_model, gcn_dataset, _ = load_model_and_dataset("configs/default.yaml")
                gat_model, gat_dataset, _ = load_model_and_dataset("configs/gat.yaml")
                
                device = get_device()
                
                # Evaluate both models
                gcn_evaluator = Evaluator(gcn_model, gcn_dataset, device)
                gat_evaluator = Evaluator(gat_model, gat_dataset, device)
                
                gcn_results = gcn_evaluator.evaluate_episodes(
                    num_episodes=30,
                    split="test",
                    num_ways=5,
                    num_support=1,
                    num_query=5,
                )
                
                gat_results = gat_evaluator.evaluate_episodes(
                    num_episodes=30,
                    split="test",
                    num_ways=5,
                    num_support=1,
                    num_query=5,
                )
                
                # Create comparison plot
                models = ["GCN", "GAT"]
                accuracies = [gcn_results["mean_accuracy"], gat_results["mean_accuracy"]]
                stds = [gcn_results["std_accuracy"], gat_results["std_accuracy"]]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=models,
                    y=accuracies,
                    error_y=dict(type='data', array=stds),
                    name='Accuracy',
                    marker_color=['#1f77b4', '#ff7f0e']
                ))
                
                fig.update_layout(
                    title="Model Comparison: GCN vs GAT",
                    xaxis_title="Model",
                    yaxis_title="Accuracy",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display comparison table
                comparison_df = pd.DataFrame({
                    "Model": models,
                    "Mean Accuracy": accuracies,
                    "Std Accuracy": stds,
                })
                
                st.dataframe(comparison_df, use_container_width=True)


if __name__ == "__main__":
    main()
