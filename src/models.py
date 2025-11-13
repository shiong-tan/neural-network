"""
Complete neural network models.

This module implements the one-hidden-layer MLP that composes affine layers
and activation functions, with both forward and backward passes.
"""

import numpy as np
from typing import Dict, Optional
from src.layers import AffineLayer
from src.activations import Activation, ReLU, Sigmoid


class OneHiddenLayerMLP:
    """
    One-hidden-layer Multi-Layer Perceptron.

    Mathematical formulation:
        a₁ = W₁x + b₁  ∈ ℝʰ    (first affine map)
        h₁ = φ(a₁)      ∈ ℝʰ    (activation)
        f  = W₂h₁ + b₂  ∈ ℝ     (output affine map, scalar output)

    Where:
        W₁ ∈ ℝ^(h×d): first layer weights
        b₁ ∈ ℝʰ: first layer bias
        φ: activation function (ReLU or sigmoid)
        W₂ ∈ ℝ^(1×h): second layer weights
        b₂ ∈ ℝ: second layer bias (scalar)

    Attributes:
        layer1: First affine layer (d → h)
        activation: Activation function (ReLU or Sigmoid)
        layer2: Second affine layer (h → 1)
        a1_cache: Cached pre-activation values for backward pass
        h1_cache: Cached post-activation values for backward pass
    """

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'relu'):
        """
        Initialize one-hidden-layer MLP.

        Args:
            input_dim: Dimensionality of input features (d)
            hidden_dim: Number of hidden units (h)
            activation: Activation function type ('relu' or 'sigmoid')

        Raises:
            ValueError: If activation is not 'relu' or 'sigmoid'
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Create layers
        self.layer1 = AffineLayer(input_dim, hidden_dim)
        self.layer2 = AffineLayer(hidden_dim, 1)

        # Create activation function
        if activation.lower() == 'relu':
            self.activation = ReLU()
        elif activation.lower() == 'sigmoid':
            self.activation = Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'relu' or 'sigmoid'.")

        self.activation_name = activation.lower()

        # Caches for backward pass
        self.a1_cache: Optional[np.ndarray] = None
        self.h1_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> float:
        """
        Complete forward pass for a single sample.

        Computes:
            a₁ = W₁x + b₁
            h₁ = φ(a₁)
            f = W₂h₁ + b₂

        Args:
            x: Input sample, shape (input_dim,)

        Returns:
            f: Scalar prediction

        Examples:
            >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4)
            >>> x = np.array([1.0, -1.0])
            >>> f = model.forward(x)
            >>> isinstance(f, float)
            True
        """
        # First affine: a₁ = W₁x + b₁
        a1 = self.layer1.forward(x)
        self.a1_cache = a1

        # Activation: h₁ = φ(a₁)
        h1 = self.activation.forward(a1)
        self.h1_cache = h1

        # Second affine: f = W₂h₁ + b₂
        f = self.layer2.forward(h1)

        # Return scalar (squeeze if needed)
        if isinstance(f, np.ndarray):
            if f.shape == (1,):
                return float(f[0])
            elif f.shape == ():
                return float(f)
        return float(f)

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch forward pass.

        Computes forward pass for multiple samples simultaneously.

        Args:
            X: Input batch, shape (batch_size, input_dim)

        Returns:
            predictions: Shape (batch_size,)

        Examples:
            >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=4)
            >>> X = np.random.randn(10, 2)
            >>> predictions = model.forward_batch(X)
            >>> predictions.shape
            (10,)
        """
        # First affine: A₁ = XW₁ᵀ + b₁
        A1 = self.layer1.forward(X)
        self.a1_cache = A1

        # Activation: H₁ = φ(A₁)
        H1 = self.activation.forward(A1)
        self.h1_cache = H1

        # Second affine: F = H₁W₂ᵀ + b₂
        F = self.layer2.forward(H1)

        # Return as 1D array (batch_size,)
        return F.ravel()

    def backward(self, x: np.ndarray, y: float, f: float) -> Dict[str, np.ndarray]:
        """
        Complete backward pass for a single sample.

        Given cached intermediate values from forward pass, compute gradients
        of loss L = (1/2)(f - y)² with respect to all parameters.

        Mathematical derivation (chain rule):
            δf = ∂L/∂f = f - y                          (loss gradient)

            Layer 2 gradients:
            ∂L/∂W₂ = δf · h₁ᵀ                           (eq. 8)
            ∂L/∂b₂ = δf                                  (eq. 8)

            Backprop through layer 2:
            δh₁ = W₂ᵀ · δf                               (eq. 9)

            Backprop through activation:
            δa₁ = δh₁ ⊙ φ'(a₁)                          (eq. 9, elementwise)

            Layer 1 gradients:
            ∂L/∂W₁ = δa₁ · xᵀ                           (eq. 10)
            ∂L/∂b₁ = δa₁                                 (eq. 10)

        Args:
            x: Input sample, shape (input_dim,)
            y: Target value (scalar)
            f: Model prediction (scalar, from forward pass)

        Returns:
            Dictionary with gradient arrays:
            {
                'dL_dW2': gradient for W₂, shape (1, hidden_dim),
                'dL_db2': gradient for b₂, shape (1,),
                'dL_dW1': gradient for W₁, shape (hidden_dim, input_dim),
                'dL_db1': gradient for b₁, shape (hidden_dim,)
            }

        Raises:
            ValueError: If forward() was not called before backward()
        """
        if self.a1_cache is None or self.h1_cache is None:
            raise ValueError("Must call forward() before backward()")

        # Get cached values
        a1 = self.a1_cache
        h1 = self.h1_cache

        # Loss gradient: δf = f - y
        df = f - y

        # ========== Layer 2 Backward ==========
        # Gradient w.r.t. W₂: ∂L/∂W₂ = δf · h₁ᵀ
        # df: scalar, h1: (h,) → dL_dW2: (1, h)
        dL_dW2 = df * h1[None, :]  # Outer product, shape (1, h)

        # Gradient w.r.t. b₂: ∂L/∂b₂ = δf
        dL_db2 = np.array([df])  # Shape (1,)

        # Backprop to hidden layer: δh₁ = W₂ᵀ · δf
        # W2: (1, h), df: scalar → dh1: (h,)
        dh1 = self.layer2.W.T.ravel() * df  # Shape (h,)

        # ========== Activation Backward ==========
        # δa₁ = δh₁ ⊙ φ'(a₁)  (elementwise multiplication)
        activation_derivative = self.activation.derivative(a1)
        da1 = dh1 * activation_derivative  # Shape (h,)

        # ========== Layer 1 Backward ==========
        # Gradient w.r.t. W₁: ∂L/∂W₁ = δa₁ · xᵀ
        # da1: (h,), x: (d,) → dL_dW1: (h, d)
        dL_dW1 = da1[:, None] @ x[None, :]  # Outer product, shape (h, d)

        # Gradient w.r.t. b₁: ∂L/∂b₁ = δa₁
        dL_db1 = da1  # Shape (h,)

        return {
            'dL_dW2': dL_dW2,
            'dL_db2': dL_db2,
            'dL_dW1': dL_dW1,
            'dL_db1': dL_db1
        }

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get all model parameters.

        Returns:
            Dictionary containing all parameters:
            {'W1', 'b1', 'W2', 'b2'}
        """
        return {
            'W1': self.layer1.W,
            'b1': self.layer1.b,
            'W2': self.layer2.W,
            'b2': self.layer2.b
        }

    def set_parameters(self, params: Dict[str, np.ndarray]):
        """
        Set model parameters (useful for testing).

        Args:
            params: Dictionary with keys 'W1', 'b1', 'W2', 'b2'
        """
        if 'W1' in params:
            self.layer1.W = params['W1']
        if 'b1' in params:
            self.layer1.b = params['b1']
        if 'W2' in params:
            self.layer2.W = params['W2']
        if 'b2' in params:
            self.layer2.b = params['b2']

    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"OneHiddenLayerMLP(input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"activation='{self.activation_name}')")
