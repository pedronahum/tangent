# Tangent Tutorials & Notebooks

This directory contains interactive Jupyter notebooks demonstrating Tangent's capabilities.

## Available Notebooks

### [tangent_tutorial.ipynb](tangent_tutorial.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/notebooks/tangent_tutorial.ipynb)

A comprehensive tutorial covering:

1. **Installation & Setup** - Getting started with Tangent
2. **Basic Concepts** - Understanding source-to-source autodiff
3. **NumPy Integration** - Vector and matrix operations
4. **TensorFlow 2.x Integration** - Working with TensorFlow tensors
5. **JAX Integration** - High-performance numerical computing
6. **Advanced Features** - Multiple gradients, result preservation
7. **Visualization & Debugging** - Plotting gradients, gradient checking, benchmarking
8. **Real-World Examples** - Linear regression, logistic regression, neural networks

## Running the Notebooks

### Local Installation

```bash
# Install Jupyter
pip install jupyter

# Install Tangent and dependencies from GitHub
pip install git+https://github.com/pedronahum/tangent.git numpy matplotlib

# Optional: Install backends
pip install jax jaxlib tensorflow

# Start Jupyter
jupyter notebook notebooks/tangent_tutorial.ipynb
```

### Google Colab

Click the "Open in Colab" badge above to run the notebook in your browser with free GPU/TPU access!

## Features Demonstrated

- 📊 **Visualization**: Interactive plots of functions and gradients
- 🔬 **Gradient Checking**: Numerical vs autodiff gradient comparison
- ⚡ **Performance Benchmarks**: Speed comparison across NumPy/TensorFlow/JAX
- 🧠 **Machine Learning**: Complete training loops for ML models
- 📖 **Educational**: Step-by-step explanations with code

## Learning Path

1. Start with the tutorial notebook to understand basics
2. Experiment with the code cells
3. Try modifying functions to see gradient changes
4. Build your own models using the examples as templates

## Contributing

Have an interesting use case or tutorial idea? Contributions are welcome!

1. Create a new notebook in this directory
2. Add clear markdown explanations
3. Include visualizations where helpful
4. Update this README with a description
5. Submit a pull request

## Resources

- [Tangent Documentation](https://github.com/pedronahum/tangent)
- [NumPy Documentation](https://numpy.org/doc/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
