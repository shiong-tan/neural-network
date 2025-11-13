"""
PyTorch implementations of neural network models.

This module provides four progressively sophisticated PyTorch implementations
of the one-hidden-layer MLP, demonstrating different levels of abstraction:

1. Manual tensors with explicit forward/backward
2. Autograd for automatic differentiation
3. nn.Module with custom layers
4. nn.Sequential with built-in layers

These implementations demonstrate equivalence with the NumPy version while
showcasing PyTorch's features and best practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# =============================================================================
# Implementation 1: Manual Tensors with Explicit Operations
# =============================================================================

class ManualTensorMLP:
    """
    One-hidden-layer MLP using manual tensor operations.

    Most similar to NumPy implementation. Uses PyTorch tensors but
    computes forward and backward passes explicitly.

    Mathematical formulation:
        a₁ = W₁x + b₁  ∈ ℝʰ
        h₁ = φ(a₁)      ∈ ℝʰ
        f  = W₂h₁ + b₂  ∈ ℝ

    Attributes:
        W1: First layer weights, shape (h, d)
        b1: First layer bias, shape (h,)
        W2: Second layer weights, shape (1, h)
        b2: Second layer bias, shape (1,)
        activation: 'relu' or 'sigmoid'
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'relu'):
        """
        Initialize manual tensor MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function type
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation.lower()

        # Initialize parameters as tensors with requires_grad=True
        self.W1 = torch.randn(hidden_dim, input_dim, dtype=torch.float32) * 0.1
        self.b1 = torch.zeros(hidden_dim, dtype=torch.float32)
        self.W2 = torch.randn(1, hidden_dim, dtype=torch.float32) * 0.1
        self.b2 = torch.zeros(1, dtype=torch.float32)

        # Enable gradient computation
        self.W1.requires_grad = True
        self.b1.requires_grad = True
        self.W2.requires_grad = True
        self.b2.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (d,) or (batch_size, d)

        Returns:
            Output tensor, shape (1,) or (batch_size,)
        """
        # First affine: a₁ = W₁x + b₁
        if x.ndim == 1:
            a1 = self.W1 @ x + self.b1
        else:
            a1 = x @ self.W1.T + self.b1

        # Activation: h₁ = φ(a₁)
        if self.activation == 'relu':
            h1 = torch.relu(a1)
        elif self.activation == 'sigmoid':
            h1 = torch.sigmoid(a1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Second affine: f = W₂h₁ + b₂
        if h1.ndim == 1:
            f = self.W2 @ h1 + self.b2
        else:
            f = h1 @ self.W2.T + self.b2

        return f.squeeze()

    def parameters(self) -> list:
        """Return list of parameters."""
        return [self.W1, self.b1, self.W2, self.b2]


# =============================================================================
# Implementation 2: Autograd with Custom Forward
# =============================================================================

class AutogradMLP:
    """
    One-hidden-layer MLP using PyTorch autograd.

    Similar to ManualTensorMLP but uses torch.nn.Parameter for automatic
    parameter management and relies on autograd for backward pass.

    Demonstrates PyTorch's automatic differentiation capabilities.
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'relu'):
        """
        Initialize autograd MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function type
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation.lower()

        # Use nn.Parameter for automatic parameter registration
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with autograd.

        Args:
            x: Input tensor, shape (d,) or (batch_size, d)

        Returns:
            Output tensor, shape (1,) or (batch_size,)
        """
        # First affine
        if x.ndim == 1:
            a1 = self.W1 @ x + self.b1
        else:
            a1 = x @ self.W1.T + self.b1

        # Activation
        if self.activation == 'relu':
            h1 = F.relu(a1)
        elif self.activation == 'sigmoid':
            h1 = torch.sigmoid(a1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Second affine
        if h1.ndim == 1:
            f = self.W2 @ h1 + self.b2
        else:
            f = h1 @ self.W2.T + self.b2

        return f.squeeze()

    def parameters(self) -> list:
        """Return list of parameters."""
        return [self.W1, self.b1, self.W2, self.b2]


# =============================================================================
# Implementation 3: nn.Module with Custom Layers
# =============================================================================

class ModuleMLP(nn.Module):
    """
    One-hidden-layer MLP using nn.Module.

    Proper PyTorch nn.Module implementation with custom forward method.
    Uses nn.Linear for affine transformations.

    This is the standard way to implement neural networks in PyTorch.
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'relu'):
        """
        Initialize nn.Module MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function type
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation.lower()

        # Define layers using nn.Linear
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

        # Initialize weights with small values
        nn.init.normal_(self.layer1.weight, mean=0, std=0.1)
        nn.init.zeros_(self.layer1.bias)
        nn.init.normal_(self.layer2.weight, mean=0, std=0.1)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (d,) or (batch_size, d)

        Returns:
            Output tensor, shape (1,) or (batch_size,)
        """
        # First layer with activation
        a1 = self.layer1(x)

        if self.activation == 'relu':
            h1 = F.relu(a1)
        elif self.activation == 'sigmoid':
            h1 = torch.sigmoid(a1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Second layer
        f = self.layer2(h1)

        return f.squeeze()


# =============================================================================
# Implementation 4: nn.Sequential
# =============================================================================

class SequentialMLP(nn.Module):
    """
    One-hidden-layer MLP using nn.Sequential.

    Most concise implementation using PyTorch's Sequential container.
    Demonstrates how to build models compositionally.

    This is the simplest way to define feedforward networks in PyTorch.
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'relu'):
        """
        Initialize sequential MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function type
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Select activation module
        if activation.lower() == 'relu':
            activation_fn = nn.ReLU()
        elif activation.lower() == 'sigmoid':
            activation_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build sequential model
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, 1),
            nn.Flatten(0)  # Flatten to scalar for single sample
        )

        # Initialize weights
        nn.init.normal_(self.model[0].weight, mean=0, std=0.1)
        nn.init.zeros_(self.model[0].bias)
        nn.init.normal_(self.model[2].weight, mean=0, std=0.1)
        nn.init.zeros_(self.model[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (d,) or (batch_size, d)

        Returns:
            Output tensor, shape (1,) or (batch_size,)
        """
        return self.model(x)


# =============================================================================
# Training Utilities
# =============================================================================

def train_pytorch_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    verbose: bool = False
) -> Dict[str, list]:
    """
    Train a PyTorch model using SGD.

    Args:
        model: PyTorch model to train
        X_train: Training inputs, shape (n_samples, input_dim)
        y_train: Training targets, shape (n_samples,)
        n_epochs: Number of training epochs
        learning_rate: Learning rate for SGD
        batch_size: Mini-batch size
        verbose: Whether to print progress

    Returns:
        Training history with 'train_loss' key

    Examples:
        >>> model = ModuleMLP(input_dim=2, hidden_dim=4)
        >>> X = torch.randn(100, 2)
        >>> y = torch.randn(100)
        >>> history = train_pytorch_model(model, X, y, n_epochs=50, verbose=False)
        >>> len(history['train_loss'])
        50
    """
    # Setup optimizer and loss
    if isinstance(model, (ManualTensorMLP, AutogradMLP)):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    history = {'train_loss': []}

    n_samples = X_train.shape[0]

    for epoch in range(n_epochs):
        # Shuffle data
        indices = torch.randperm(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            predictions = model.forward(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(X_batch)
            n_batches += 1

        avg_loss = epoch_loss / n_samples
        history['train_loss'].append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.6f}")

    return history


def convert_numpy_to_pytorch(model_numpy, model_pytorch):
    """
    Convert NumPy model parameters to PyTorch model.

    Args:
        model_numpy: NumPy OneHiddenLayerMLP
        model_pytorch: One of the PyTorch model implementations

    Examples:
        >>> from src.models import OneHiddenLayerMLP
        >>> np_model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        >>> pt_model = ModuleMLP(input_dim=2, hidden_dim=3)
        >>> convert_numpy_to_pytorch(np_model, pt_model)
    """
    import numpy as np

    # Get NumPy parameters
    params = model_numpy.get_parameters()

    # Convert to PyTorch and assign
    if isinstance(model_pytorch, (ManualTensorMLP, AutogradMLP)):
        model_pytorch.W1.data = torch.from_numpy(params['W1'].astype(np.float32))
        model_pytorch.b1.data = torch.from_numpy(params['b1'].astype(np.float32))
        model_pytorch.W2.data = torch.from_numpy(params['W2'].astype(np.float32))
        model_pytorch.b2.data = torch.from_numpy(params['b2'].astype(np.float32))
    elif isinstance(model_pytorch, (ModuleMLP, SequentialMLP)):
        if isinstance(model_pytorch, SequentialMLP):
            layer1 = model_pytorch.model[0]
            layer2 = model_pytorch.model[2]
        else:
            layer1 = model_pytorch.layer1
            layer2 = model_pytorch.layer2

        # PyTorch Linear uses transposed weight matrices
        layer1.weight.data = torch.from_numpy(params['W1'].astype(np.float32))
        layer1.bias.data = torch.from_numpy(params['b1'].astype(np.float32))
        layer2.weight.data = torch.from_numpy(params['W2'].astype(np.float32))
        layer2.bias.data = torch.from_numpy(params['b2'].astype(np.float32))
