"""
Gradient checking utilities for validating backpropagation.

This module implements numerical gradient computation using finite differences
to verify that analytical gradients from backpropagation are correct.

The central difference method is used for higher accuracy:
    f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)

Typical choice: ε = 1e-5 (balances numerical precision vs truncation error)
"""

import numpy as np
from typing import Dict, Tuple, Callable
from src.models import OneHiddenLayerMLP


def compute_numerical_gradient(
    model: OneHiddenLayerMLP,
    x: np.ndarray,
    y: float,
    param_name: str,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient of loss w.r.t. a parameter using finite differences.

    Uses central difference method for each element of the parameter:
        ∂L/∂θᵢ ≈ [L(θ + εeᵢ) - L(θ - εeᵢ)] / (2ε)

    where eᵢ is the i-th unit vector.

    Args:
        model: OneHiddenLayerMLP instance
        x: Input sample, shape (input_dim,)
        y: Target value (scalar)
        param_name: Name of parameter ('W1', 'b1', 'W2', 'b2')
        epsilon: Perturbation size (default 1e-5)

    Returns:
        Numerical gradient with same shape as parameter

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        >>> x = np.array([1.0, -1.0])
        >>> y = 1.0
        >>> grad_num = compute_numerical_gradient(model, x, y, 'W1', epsilon=1e-5)
        >>> grad_num.shape
        (3, 2)
    """
    # Get original parameter
    params = model.get_parameters()
    param = params[param_name]

    # Initialize numerical gradient with same shape
    grad_numerical = np.zeros_like(param)

    # Create a flat view for iteration
    flat_param = param.ravel()
    flat_grad = grad_numerical.ravel()

    # Compute gradient for each element
    for i in range(len(flat_param)):
        # Save original value
        original_value = flat_param[i]

        # Perturb positively: θᵢ + ε
        flat_param[i] = original_value + epsilon
        model.set_parameters({param_name: param})
        f_plus = model.forward(x)
        loss_plus = 0.5 * (f_plus - y) ** 2

        # Perturb negatively: θᵢ - ε
        flat_param[i] = original_value - epsilon
        model.set_parameters({param_name: param})
        f_minus = model.forward(x)
        loss_minus = 0.5 * (f_minus - y) ** 2

        # Central difference
        flat_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # Restore original value
        flat_param[i] = original_value

    # Restore original parameters
    model.set_parameters({param_name: param})

    return grad_numerical


def check_gradient(
    model: OneHiddenLayerMLP,
    x: np.ndarray,
    y: float,
    param_name: str,
    epsilon: float = 1e-5,
    threshold: float = 1e-7
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Check if analytical gradient matches numerical gradient.

    Computes both analytical (backprop) and numerical (finite diff) gradients,
    then compares using relative error metric:

        relative_error = ||grad_analytical - grad_numerical||₂ /
                        (||grad_analytical||₂ + ||grad_numerical||₂)

    Args:
        model: OneHiddenLayerMLP instance
        x: Input sample, shape (input_dim,)
        y: Target value (scalar)
        param_name: Parameter name ('W1', 'b1', 'W2', 'b2')
        epsilon: Perturbation size for numerical gradient
        threshold: Maximum acceptable relative error (default 1e-7)

    Returns:
        Tuple of (passed, relative_error, grad_analytical, grad_numerical):
        - passed: True if relative_error < threshold
        - relative_error: Computed relative error
        - grad_analytical: Gradient from backpropagation
        - grad_numerical: Gradient from finite differences

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        >>> x = np.array([1.0, -1.0])
        >>> y = 1.0
        >>> passed, error, g_anal, g_num = check_gradient(model, x, y, 'W1')
        >>> passed
        True
        >>> error < 1e-7
        True
    """
    # Compute analytical gradient via backpropagation
    f = model.forward(x)
    grads = model.backward(x, y, f)

    # Map parameter name to gradient key
    grad_key_map = {
        'W1': 'dL_dW1',
        'b1': 'dL_db1',
        'W2': 'dL_dW2',
        'b2': 'dL_db2'
    }
    grad_analytical = grads[grad_key_map[param_name]]

    # Compute numerical gradient via finite differences
    grad_numerical = compute_numerical_gradient(model, x, y, param_name, epsilon)

    # Compute relative error
    numerator = np.linalg.norm(grad_analytical - grad_numerical)
    denominator = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical)

    # Handle edge case where both gradients are zero
    if denominator < 1e-12:
        relative_error = 0.0
    else:
        relative_error = numerator / denominator

    passed = relative_error < threshold

    return passed, relative_error, grad_analytical, grad_numerical


def check_all_gradients(
    model: OneHiddenLayerMLP,
    x: np.ndarray,
    y: float,
    epsilon: float = 1e-5,
    threshold: float = 1e-7
) -> Dict[str, Dict]:
    """
    Check gradients for all parameters in the model.

    Runs gradient checking for W1, b1, W2, b2 and returns detailed results.

    Args:
        model: OneHiddenLayerMLP instance
        x: Input sample, shape (input_dim,)
        y: Target value (scalar)
        epsilon: Perturbation size for numerical gradients
        threshold: Maximum acceptable relative error

    Returns:
        Dictionary with results for each parameter:
        {
            'W1': {'passed': bool, 'error': float, 'analytical': array, 'numerical': array},
            'b1': {'passed': bool, 'error': float, 'analytical': array, 'numerical': array},
            'W2': {'passed': bool, 'error': float, 'analytical': array, 'numerical': array},
            'b2': {'passed': bool, 'error': float, 'analytical': array, 'numerical': array}
        }

    Examples:
        >>> model = OneHiddenLayerMLP(input_dim=2, hidden_dim=3)
        >>> x = np.random.randn(2)
        >>> y = np.random.randn()
        >>> results = check_all_gradients(model, x, y)
        >>> all(r['passed'] for r in results.values())
        True
    """
    param_names = ['W1', 'b1', 'W2', 'b2']
    results = {}

    for param_name in param_names:
        passed, error, grad_analytical, grad_numerical = check_gradient(
            model, x, y, param_name, epsilon, threshold
        )
        results[param_name] = {
            'passed': passed,
            'error': error,
            'analytical': grad_analytical,
            'numerical': grad_numerical
        }

    return results


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute relative error between two arrays.

    Uses the standard formula:
        relative_error = ||a - b||₂ / (||a||₂ + ||b||₂)

    This metric is scale-invariant and symmetric.

    Args:
        a: First array
        b: Second array

    Returns:
        Relative error (scalar)

    Examples:
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.0, 2.0, 3.0])
        >>> relative_error(a, b)
        0.0
        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([1.0, 0.0001])
        >>> relative_error(a, b) < 1e-4
        True
    """
    numerator = np.linalg.norm(a - b)
    denominator = np.linalg.norm(a) + np.linalg.norm(b)

    if denominator < 1e-12:
        return 0.0

    return numerator / denominator
