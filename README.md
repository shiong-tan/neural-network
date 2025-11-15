# Neural Network from Scratch: A Pedagogical Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shiong-tan/neural-network/blob/master/notebooks/neural_network_tutorial.ipynb)

A pedagogical implementation of a one-hidden-layer neural network built from mathematical first principles. This project demonstrates the full journey from manual NumPy backpropagation to a production-ready PyTorch implementation.

## 🎯 Learning Objectives

1. **Mathematical Foundation**: Write down the forward map for a one-hidden-layer MLP
2. **Backpropagation**: Derive the backward pass using the chain rule
3. **Manual Implementation**: Implement the network and its gradients manually in NumPy
4. **Framework Translation**: Reproduce in PyTorch (tensors → autograd → nn.Module → nn.Sequential)

## 🚀 Quick Start

### Requirements

- Python 3.11 or higher

### Local Installation

```bash
git clone https://github.com/shiong-tan/neural-network.git
cd neural-network
pip install -r requirements.txt
```

### Run the Tutorial

```bash
jupyter notebook notebooks/neural_network_tutorial.ipynb
```

### Run Tests

```bash
pytest tests/
```

### Run Minimal Example

```bash
python examples/minimal_example.py
```

## 📚 Project Structure

- `notebooks/` - Interactive Jupyter tutorial following the mathematical progression
- `src/` - Modular implementation of all components
  - `data.py` - Dataset generation (XOR, spiral patterns)
  - `activations.py` - ReLU and Sigmoid with derivatives
  - `layers.py` - Affine transformations and forward pass
  - `models.py` - Complete MLP with manual backpropagation
  - `loss.py` - Squared error loss and gradients
  - `optimization.py` - SGD implementation and training loop
  - `gradient_check.py` - Numerical gradient verification
  - `pytorch_models.py` - PyTorch implementations (3 levels of abstraction)
  - `visualization.py` - Plotting utilities
- `tests/` - Comprehensive test suite
- `examples/` - Minimal runnable examples

## 🧮 Mathematical Foundation

The one-hidden-layer MLP computes:

```
a₁ = W₁x + b₁  ∈ ℝʰ    (first affine map)
h₁ = φ(a₁)      ∈ ℝʰ    (activation)
f  = W₂h₁ + b₂  ∈ ℝ     (output)
```

**Loss**: L(θ; x, y) = ½(f(x; θ) - y)²

**Backpropagation** via chain rule:
```
∂L/∂W₂ = δf · h₁ᵀ
∂L/∂b₂ = δf
δh₁ = W₂ᵀ · δf
δa₁ = δh₁ ⊙ φ'(a₁)
∂L/∂W₁ = δa₁ · xᵀ
∂L/∂b₁ = δa₁
```

## ✅ Capstone Checklist

All items completed:

1. ✅ Manual NumPy forward/backward for 1-hidden-layer MLP with gradient check
2. ✅ PyTorch re-implementation without autograd; numerical equivalence verified
3. ✅ PyTorch with autograd; gradients match manual implementation
4. ✅ nn.Module and nn.Sequential implementations; train with SGD
5. ✅ Diagnostics: loss curves, gradient L2 norms, and ReLU activity % across iterations
6. ✅ GitHub hosting with README, one-click Colab badge, end-to-end execution

## 📊 Key Visualizations

- Activation functions (ReLU, Sigmoid) and their derivatives
- Computation graph showing forward/backward dependencies
- Decision boundaries learned on XOR dataset
- Training diagnostics (loss curves, gradient L2 norms, ReLU activity %)
- Loss surface visualization for single parameters

## 🔬 Gradient Checking

Numerical gradient verification using finite differences:

```python
from src.gradient_check import check_gradients

errors = check_gradients(model, x, y, epsilon=1e-5)
# All relative errors should be < 1e-3
```

## 🎓 Pedagogical Flow

The notebook follows a carefully structured progression:

1. **Motivation** - Why build from scratch?
2. **Building Blocks** - Affine maps and activations
3. **Network Architecture** - Composing a one-hidden-layer MLP
4. **Loss Function** - Measuring prediction quality
5. **Backpropagation** - Deriving gradients via chain rule
6. **NumPy Implementation** - Manual gradients with verification
7. **PyTorch Evolution** - From tensors to high-level APIs
8. **Best Practices** - Optimization tips and diagnostics
9. **Summary** - Key insights and takeaways

## 🧪 Testing

Comprehensive test suite covering:

- Activation function correctness
- Forward pass shape consistency
- Gradient correctness (numerical verification)
- NumPy-PyTorch equivalence
- Edge cases (zero gradients, saturation)

```bash
pytest tests/ -v
```

## 📖 References

- Backpropagation: The chain rule organized along the computation graph
- PyTorch autograd: Automatic differentiation via reverse-mode AD
- Gradient Descent optimization techniques
- Matrix calculus conventions

## 🤝 Contributing

This is a project designed for learning. If you find issues or have suggestions for improving clarity, please open an issue!

## 📝 License

MIT License - feel free to use for educational purposes.
