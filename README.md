# Neural Network from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shiong-tan/neural-network/blob/master/notebooks/neural_network_tutorial.ipynb)

A one-hidden-layer MLP implemented from first principles in NumPy, with gradient checking to verify correctness and a progression through four levels of PyTorch abstraction.

This implementation includes:
- Complete manual backpropagation with numerical gradient verification (< 1e-7 relative error)
- Detailed mathematical derivations in code comments
- Comprehensive test suite validating forward pass, backward pass, and NumPy-PyTorch equivalence
- Training diagnostics including loss curves, gradient norms, and ReLU activation statistics

## What You'll Learn

Write a one-hidden-layer neural network from scratch and understand exactly what's happening at each step:

1. Forward propagation through affine layers and activation functions
2. Backpropagation using the chain rule to compute gradients
3. Manual implementation in NumPy with explicit gradient calculations
4. PyTorch progression: manual tensors → autograd → nn.Module → nn.Sequential

## Quick Start

**Requirements:** Python 3.11+

**Local Installation:**
```bash
git clone https://github.com/shiong-tan/neural-network.git
cd neural-network
pip install -r requirements.txt
```

**Run the interactive tutorial:**
```bash
jupyter notebook notebooks/neural_network_tutorial.ipynb
```

**Run tests:**
```bash
pytest tests/
```

**Run minimal example:**
```bash
python examples/minimal_example.py
```

## Project Structure

```
├── src/                     # Core implementation
│   ├── models.py           # MLP with manual backpropagation
│   ├── layers.py           # Affine transformations
│   ├── activations.py      # ReLU and Sigmoid (with derivatives)
│   ├── loss.py             # Squared error loss
│   ├── optimization.py     # SGD and training loop
│   ├── gradient_check.py   # Numerical gradient verification
│   ├── pytorch_models.py   # 4 PyTorch abstraction levels
│   ├── data.py             # XOR and spiral datasets
│   └── visualization.py    # Training diagnostics plots
├── tests/                   # Comprehensive pytest suite
├── notebooks/               # Interactive Jupyter tutorial
└── examples/                # Minimal runnable examples
```

## Network Architecture

The MLP computes:

```
a₁ = W₁x + b₁  ∈ ℝʰ    (first affine transformation)
h₁ = φ(a₁)      ∈ ℝʰ    (ReLU or Sigmoid activation)
f  = W₂h₁ + b₂  ∈ ℝ     (scalar output)
```

**Loss function:** L(θ; x, y) = ½(f(x; θ) - y)²

**Backpropagation** applies the chain rule to compute gradients:
```
∂L/∂W₂ = δf · h₁ᵀ
∂L/∂b₂ = δf
δh₁ = W₂ᵀ · δf
δa₁ = δh₁ ⊙ φ'(a₁)
∂L/∂W₁ = δa₁ · xᵀ
∂L/∂b₁ = δa₁
```

where δf = f - y.

## Implementation Details

**Gradient Checking:** Every gradient is verified using finite differences. The relative error between analytical and numerical gradients is consistently below 1e-7, confirming the backpropagation implementation is correct.

```python
from src.gradient_check import check_gradients

errors = check_gradients(model, x, y, epsilon=1e-5)
# All relative errors < 1e-7
```

**Testing:** The test suite covers activation functions, forward pass shape consistency, backward pass correctness, NumPy-PyTorch equivalence, and edge cases like zero gradients and activation saturation.

```bash
pytest tests/ -v
```

**PyTorch Progression:** Four implementations showing the evolution from low-level to framework-idiomatic code:
1. `ManualTensorMLP` - Manual operations with PyTorch tensors (manual backward)
2. `AutogradMLP` - nn.Parameter with automatic differentiation
3. `ModuleMLP` - nn.Module with nn.Linear layers
4. `SequentialMLP` - High-level nn.Sequential API

**Training Diagnostics:** The training loop tracks and visualizes:
- Training and validation loss curves
- Gradient L2 norms per layer
- ReLU activation percentage across iterations

## What's Included

**Completed implementation:**
- Manual NumPy forward/backward for 1-hidden-layer MLP
- Numerical gradient checking with < 1e-7 relative error
- PyTorch implementations (manual tensors, autograd, nn.Module, nn.Sequential)
- Training diagnostics (loss curves, gradient norms, ReLU activity)
- Interactive Jupyter notebook tutorial
- One-click Colab execution

**Visualizations:**
- Activation functions and their derivatives
- Computation graph with forward/backward dependencies
- Decision boundaries on XOR and spiral datasets
- Training metrics over iterations
- Loss surface plots for individual parameters

## Notebook Structure

The tutorial walks through:

1. Motivation and mathematical setup
2. Building blocks: affine maps and activations
3. Network architecture and forward pass
4. Loss function and prediction quality
5. Backpropagation via chain rule
6. NumPy implementation with gradient verification
7. PyTorch evolution through abstraction levels
8. Optimization techniques and diagnostics

## References

- Chain rule and backpropagation along computation graphs
- PyTorch autograd (reverse-mode automatic differentiation)
- Gradient descent optimization
- Matrix calculus conventions

## Contributing

Found an issue or have suggestions for clarity? Open an issue.

## License

MIT License
