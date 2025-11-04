# Building Energy Optimization Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Building_Energy_Optimization_with_Tangent.ipynb)

## Overview

This comprehensive tutorial demonstrates Tangent's automatic differentiation capabilities through a real-world building energy optimization problem. Perfect for teaching, learning, or showcasing Tangent's features!

## What's Inside

### 1. Physical Simulation 🏢
Real building thermal dynamics:
- Heat transfer through walls
- Solar heat gain (time-varying)
- Thermal mass effects
- Temperature control

### 2. Optimization Problem 💰
Minimize total cost subject to comfort:
```
minimize: energy_cost + comfort_penalty
subject to: heating >= 0
```

Time-varying electricity prices create interesting optimization challenges:
- Peak hours: $0.25/kWh (8am-8pm)
- Off-peak: $0.10/kWh (night)

### 3. Educational Content 🎓

**Level 1: Basic Concepts**
- Forward simulation
- Cost function design
- Gradient computation

**Level 2: Code Generation**
- Unoptimized gradient code
- Optimized gradient code
- Side-by-side comparison

**Level 3: Optimizations**
- Common Subexpression Elimination (CSE)
- Dead Code Elimination (DCE)
- Algebraic Simplification
- Performance measurements

**Level 4: Advanced Topics**
- Multi-parameter optimization
- Sensitivity analysis
- Physical insights from gradients

## Key Features

### ✅ Self-Contained
- No external data files needed
- All code in notebook
- Runs immediately in Colab

### ✅ Visual
- Temperature profiles
- Price schedules
- Gradient sensitivity maps
- Optimization trajectories
- Cost convergence

### ✅ Interactive
- Modify parameters
- Try different scenarios
- Experiment with optimizations
- Compare algorithms

### ✅ Pedagogical
- Clear explanations
- Progressive complexity
- Mathematical formulas
- Code comments
- Learning objectives

## Learning Outcomes

After completing this notebook, you will understand:

1. **How automatic differentiation works**
   - Forward pass: compute function value
   - Backward pass: accumulate gradients
   - Chain rule application

2. **What Tangent generates**
   - Readable Python code
   - Not a computation graph!
   - Inspectable and debuggable

3. **Why optimizations matter**
   - Speedup without loss of correctness
   - CSE eliminates redundant computation
   - DCE removes dead code

4. **How to use gradients**
   - Gradient descent optimization
   - Sensitivity analysis
   - Parameter tuning

5. **Real-world applications**
   - Building control
   - Energy management
   - Smart grid optimization

## Use Cases

### Teaching 🎓
- University courses on optimization/control
- Automatic differentiation tutorials
- Machine learning fundamentals
- Numerical methods

### Research 🔬
- Building energy management
- Smart grid optimization
- Model predictive control
- Sensitivity analysis

### Industry 💼
- HVAC system optimization
- Demand response programs
- Energy cost reduction
- Facility management

## Prerequisites

**Required:**
- Basic Python knowledge
- Understanding of for loops
- Familiarity with NumPy (helpful but not required)

**Not Required:**
- No ML experience needed
- No calculus required (but helpful for understanding)
- No building engineering background

## Running the Notebook

### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge above
2. Wait for notebook to load
3. Run cells in order (Shift+Enter)
4. Experiment and have fun!

### Option 2: Local Jupyter
```bash
# Install dependencies
pip install tangent numpy matplotlib

# Clone repository
git clone https://github.com/pedronahum/tangent.git
cd tangent/examples

# Launch Jupyter
jupyter notebook Building_Energy_Optimization_with_Tangent.ipynb
```

### Option 3: JupyterLab
```bash
pip install jupyterlab tangent numpy matplotlib
jupyter lab Building_Energy_Optimization_with_Tangent.ipynb
```

## Notebook Structure

```
├── Setup & Installation
├── Problem Description
│   ├── Physical model equations
│   ├── Optimization formulation
│   └── Visual scenario setup
├── Forward Simulation
│   ├── Code implementation
│   └── Initial results
├── Automatic Differentiation
│   ├── Level 1: Unoptimized gradient
│   ├── Level 2: Optimized gradient
│   └── Code inspection
├── Performance Comparison
│   ├── Numerical differentiation
│   ├── Tangent (unoptimized)
│   └── Tangent (optimized)
├── Gradient Visualization
│   ├── Sensitivity analysis
│   └── Physical interpretation
├── Optimization
│   ├── Gradient descent
│   ├── Convergence analysis
│   └── Results visualization
├── Advanced Topics
│   ├── Multi-parameter optimization
│   └── Thermal mass sensitivity
└── Summary & Resources
```

## Typical Results

**Initial (Constant Heating):**
- Total cost: ~$180
- Energy cost: ~$120
- Comfort penalty: ~$60

**After Optimization:**
- Total cost: ~$140
- Energy cost: ~$95
- Comfort penalty: ~$45
- **Savings: ~22%**

**Performance:**
- Numerical differentiation: ~250ms
- Tangent (unoptimized): ~15ms (17x faster)
- Tangent (optimized): ~8ms (31x faster)

## Customization Ideas

Try modifying:

### Parameters
```python
params = {
    'R': 5.0,      # Try 3.0-10.0 (insulation quality)
    'C': 10.0,     # Try 5.0-20.0 (thermal mass)
    'lambda_comfort': 10.0,  # Try 1.0-50.0 (comfort importance)
}
```

### Scenarios
- Different outdoor temperature profiles (heat wave, cold snap)
- Various pricing schemes (ToU, real-time pricing)
- Multiple comfort zones
- Occupancy schedules

### Optimizations
- Different learning rates
- Alternative algorithms (Adam, momentum)
- Constraint handling methods
- Multi-objective optimization

### Extensions
- Add battery storage
- Include solar panels
- Model multiple rooms
- Predict weather

## Related Examples

- **[Tangent Tutorial](../notebooks/tangent_tutorial.ipynb)** - General introduction
- **Linear Regression** - Simple gradient descent
- **Neural Networks** - Deep learning with Tangent
- **Physics Simulation** - Pendulum dynamics

## Citation

If you use this example in research or teaching, please cite:

```bibtex
@misc{tangent_building_example,
  title={Building Energy Optimization with Tangent},
  author={Tangent Contributors},
  year={2025},
  url={https://github.com/pedronahum/tangent/tree/master/examples}
}
```

## Feedback & Contributions

Found a bug? Have a suggestion? Want to contribute?

- **Issues**: [GitHub Issues](https://github.com/pedronahum/tangent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pedronahum/tangent/discussions)
- **Pull Requests**: Always welcome!

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

---

**Happy Learning! 🎉**

*Made with ❤️ by the Tangent community*
