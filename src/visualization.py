"""
Visualization utilities for neural network analysis.

This module provides functions for visualizing training progress, decision
boundaries, and model behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from src.models import OneHiddenLayerMLP

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss over epochs.

    Args:
        history: Dictionary with 'train_loss' and optionally 'val_loss' keys
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Examples:
        >>> history = {'train_loss': [0.5, 0.3, 0.2], 'val_loss': [0.6, 0.4, 0.3]}
        >>> plot_training_history(history)
    """
    plt.figure(figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_diagnostics(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot training diagnostics: loss, gradient norms, and ReLU activity.

    Args:
        history: Dictionary with 'train_loss', 'grad_norms', and 'relu_activity' keys
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Examples:
        >>> history = {
        ...     'train_loss': [0.5, 0.3, 0.2],
        ...     'grad_norms': [1.2, 0.8, 0.5],
        ...     'relu_activity': [65.0, 58.0, 52.0]
        ... }
        >>> plot_diagnostics(history)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Training Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
        axes[0].legend(fontsize=10)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Gradient Norms
    if 'grad_norms' in history and len(history['grad_norms']) > 0:
        axes[1].plot(epochs, history['grad_norms'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Gradient L2 Norm', fontsize=12)
        axes[1].set_title('Gradient Norms', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

    # Plot 3: ReLU Activity
    if 'relu_activity' in history and len(history['relu_activity']) > 0:
        # Check if values are valid (not all NaN - which indicates non-ReLU activation)
        if not all(np.isnan(val) for val in history['relu_activity']):
            axes[2].plot(epochs, history['relu_activity'], 'm-', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Active Neurons (%)', fontsize=12)
            axes[2].set_title('ReLU Activity', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([0, 100])
        else:
            # ReLU activity not applicable (e.g., using sigmoid activation)
            axes[2].text(0.5, 0.5, 'ReLU Activity\n(Not Applicable)',
                        ha='center', va='center', fontsize=12,
                        transform=axes[2].transAxes)
            axes[2].set_xticks([])
            axes[2].set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_decision_boundary(
    model: OneHiddenLayerMLP,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 200,
    threshold: float = 0.5,
    title: str = 'Decision Boundary',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot model decision boundary with data points.

    Args:
        model: Trained OneHiddenLayerMLP
        X: Input data, shape (n_samples, 2)
        y: Labels, shape (n_samples,)
        resolution: Grid resolution for boundary
        threshold: Decision threshold for class boundary line
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Examples:
        >>> from src.data import generate_xor_data
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4)
        >>> X, y = generate_xor_data(n_samples=200)
        >>> plot_decision_boundary(model, X, y, title='XOR Decision Boundary')
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plot requires 2D input data")

    plt.figure(figsize=figsize)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward_batch(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.colorbar(label='Model Output')

    # Plot decision threshold
    plt.contour(xx, yy, Z, levels=[threshold], colors='black', linewidths=2, linestyles='--')

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='black', s=50, linewidth=1.5)
    plt.colorbar(scatter, label='True Label')

    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dataset(
    X: np.ndarray,
    y: np.ndarray,
    title: str = 'Dataset',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot 2D dataset.

    Args:
        X: Input data, shape (n_samples, 2)
        y: Labels, shape (n_samples,)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Examples:
        >>> from src.data import generate_xor_data
        >>> X, y = generate_xor_data(n_samples=200)
        >>> plot_dataset(X, y, title='XOR Problem')
    """
    if X.shape[1] != 2:
        raise ValueError("Dataset plot requires 2D input data")

    plt.figure(figsize=figsize)

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='black', s=50, linewidth=1.5)
    plt.colorbar(scatter, label='Label')

    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_activation_functions(
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
):
    """
    Plot ReLU and Sigmoid activation functions.

    Args:
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Examples:
        >>> plot_activation_functions()
    """
    z = np.linspace(-5, 5, 200)

    # ReLU
    relu = np.maximum(0, z)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-z))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot ReLU
    ax1.plot(z, relu, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('$z$', fontsize=12)
    ax1.set_ylabel('ReLU$(z)$', fontsize=12)
    ax1.set_title('ReLU Activation', fontsize=14, fontweight='bold')

    # Plot Sigmoid
    ax2.plot(z, sigmoid, 'r-', linewidth=2)
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('$z$', fontsize=12)
    ax2.set_ylabel('Sigmoid$(z)$', fontsize=12)
    ax2.set_title('Sigmoid Activation', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_surface(
    model: OneHiddenLayerMLP,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str = 'W1',
    param_indices: Tuple[int, int] = (0, 0),
    range_factor: float = 2.0,
    resolution: int = 50,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot loss surface with respect to two parameters.

    Args:
        model: OneHiddenLayerMLP to analyze
        X: Input data for loss computation
        y: Target data for loss computation
        param_name: Parameter to vary ('W1', 'b1', 'W2', 'b2')
        param_indices: Which two elements of parameter to vary
        range_factor: How far to vary from optimal value (multiplicative)
        resolution: Grid resolution
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        >>> X = np.random.randn(50, 2)
        >>> y = np.random.randint(0, 2, 50)
        >>> plot_loss_surface(model, X, y, param_name='W1', param_indices=(0, 0))
    """
    from src.loss import squared_error_loss

    # Get original parameters
    params = model.get_parameters()
    param = params[param_name]

    # Get optimal values
    idx1, idx2 = param_indices
    if param.ndim == 1:
        val1_opt = param[idx1]
        val2_opt = param[idx2] if idx2 < len(param) else param[0]
    else:
        val1_opt = param.flat[idx1]
        val2_opt = param.flat[idx2]

    # Create ranges
    val1_range = np.linspace(val1_opt - range_factor * abs(val1_opt),
                            val1_opt + range_factor * abs(val1_opt),
                            resolution)
    val2_range = np.linspace(val2_opt - range_factor * abs(val2_opt),
                            val2_opt + range_factor * abs(val2_opt),
                            resolution)

    V1, V2 = np.meshgrid(val1_range, val2_range)
    losses = np.zeros_like(V1)

    # Compute loss for each parameter combination
    original_param = param.copy()

    for i in range(resolution):
        for j in range(resolution):
            # Set parameters
            param_modified = original_param.copy()
            if param.ndim == 1:
                param_modified[idx1] = V1[i, j]
                if idx2 < len(param):
                    param_modified[idx2] = V2[i, j]
            else:
                param_modified.flat[idx1] = V1[i, j]
                param_modified.flat[idx2] = V2[i, j]

            model.set_parameters({param_name: param_modified})

            # Compute loss
            predictions = model.forward_batch(X)
            losses[i, j] = squared_error_loss(predictions, y)

    # Restore original parameters
    model.set_parameters({param_name: original_param})

    # Plot
    plt.figure(figsize=figsize)
    contour = plt.contourf(V1, V2, losses, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Loss')
    plt.contour(V1, V2, losses, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # Mark optimal point
    plt.plot(val1_opt, val2_opt, 'r*', markersize=15, label='Optimal')

    plt.xlabel(f'{param_name}[{idx1}]', fontsize=12)
    plt.ylabel(f'{param_name}[{idx2}]', fontsize=12)
    plt.title(f'Loss Surface: {param_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
