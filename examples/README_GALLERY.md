# Gallery of Gradients

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/examples/Gallery_of_Gradients.ipynb)

## Overview

The **Gallery of Gradients** is a curated collection of examples showcasing Tangent's killer feature: **readable gradient code**. Unlike other automatic differentiation libraries that build opaque computation graphs, Tangent generates pure, readable Python code that you can read, debug, and optimize.

## Why This Matters

When you use most autodiff libraries, you get a black box. With Tangent, you get **actual Python code** that shows exactly how gradients are computed. This is transformative for:

- 🎓 **Learning** - Understand how automatic differentiation really works
- 🐛 **Debugging** - Step through gradient code with a standard Python debugger
- ⚡ **Optimization** - Profile and optimize gradient computation like any Python code
- ✅ **Verification** - Inspect generated code to verify correctness
- 🔧 **Customization** - Modify generated code for special requirements

## What's Inside

The notebook contains 8 carefully chosen examples that progressively demonstrate Tangent's capabilities:

### 1. **Simple Polynomial** 🔢
*Foundation of automatic differentiation*
- Shows basic chain rule application
- Demonstrates power rule: d/dx(x³) = 3x²
- Perfect starting point for beginners

### 2. **For Loop** 🔄
*The magic of reverse execution*
- Forward pass: loop runs forward
- Backward pass: loop runs **in reverse**
- Shows gradient accumulation through iterations

### 3. **While Loop** 🌀
*Stack-based tape recording*
- Handles unknown iteration counts
- Records loop state on a stack during forward pass
- Pops state in reverse order during backward pass

### 4. **Conditional Logic** 🔀
*Branch-specific gradients*
- Shows how if/else branches are handled
- Only the executed branch contributes to gradient
- Demonstrates conditional gradient flow

### 5. **NumPy Array Operations** 📊
*Gradient broadcasting and reduction*
- Element-wise operations with arrays
- Reduction operations (sum, mean)
- Shows `unreduce` for broadcasting gradients back

### 6. **Nested Function Calls** 📦
*Function inlining and composition*
- Sigmoid activation function
- Neural network layer
- Demonstrates chain rule with function composition

### 7. **Matrix Operations** 🔢
*New colon slice feature*
- Array slicing: `A[0, :]`
- Gradient routing to specific elements
- Showcases recently added colon slice support

### 8. **Optimization Comparison** ⚡
*Before and after optimization*
- Unoptimized: All intermediate steps visible
- Optimized: Dead code eliminated, expressions simplified
- Same result, better performance

## How to Use

### Option 1: Run in Google Colab (Recommended)
Click the badge at the top to open in Colab. No installation required!

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/pedronahum/tangent.git
cd tangent

# Install dependencies
pip install numpy matplotlib jupyter

# Launch Jupyter
jupyter notebook examples/Gallery_of_Gradients.ipynb
```

### Option 3: Just Read It
The notebook is designed to be readable even without execution. Each example includes:
- The original Python function
- Explanation of what to expect
- Sample generated gradient code
- Verification of correctness

## Educational Approach

Each example follows the same structure:

1. **Original Function** - Simple, clear Python code
2. **Generated Gradient Code** - Call `tangent.grad(f, verbose=1)` to see it
3. **💡 Explanation** - What's happening and why
4. **Verification** - Numerical check that gradient is correct

This structure helps you:
- See the pattern Tangent is handling
- Understand how it's transformed to gradient code
- Learn why the gradient looks the way it does
- Verify everything is mathematically correct

## Key Insights

### Readable ≠ Simple
The generated code might look complex, but it's **readable**. You can:
- Follow the logic step by step
- Understand each operation
- See where gradients flow
- Debug when something goes wrong

### Forward + Backward = Complete
Every gradient function has two parts:
1. **Forward pass**: Compute the function and save intermediate values
2. **Backward pass**: Use saved values to compute gradients

This is the essence of reverse-mode automatic differentiation!

### Optimization is Optional
Tangent can optimize gradient code, but you can see both versions:
- **Unoptimized**: Easier to understand, shows all steps
- **Optimized**: Faster execution, removes redundancy

Both are mathematically equivalent!

## Who Should Use This

- **Students** learning automatic differentiation
- **Researchers** implementing new gradient-based algorithms
- **Engineers** debugging gradient computation issues
- **Educators** teaching calculus or machine learning
- **Curious minds** who want to understand what autodiff libraries do

## Next Steps

After exploring the gallery:

1. **Try your own functions** - Use the playground cell at the end
2. **Read the feature documentation** - See what Python features Tangent supports
3. **Explore real-world examples** - Check out the Building Energy Optimization notebook
4. **Contribute** - Found a bug or have an idea? We welcome contributions!

## Technical Details

**What Tangent Does:**
- Source-to-source automatic differentiation
- Generates Python code from Python code
- Uses AST (Abstract Syntax Tree) transformations
- Implements reverse-mode AD (backpropagation)

**What Makes It Unique:**
- No computation graph runtime
- No custom data structures
- Pure Python code generation
- Fully inspectable gradients

## Resources

- [Main Documentation](../docs/INDEX.md)
- [Python Feature Support](../docs/features/PYTHON_FEATURE_SUPPORT.md)
- [Error Messages Guide](../docs/features/ERROR_MESSAGES.md)
- [Building Energy Example](README_BUILDING_EXAMPLE.md)
- [Tutorial Notebooks](../notebooks/)

## Citation

If you use this gallery for teaching or research, please cite:

```bibtex
@misc{tangent_gallery,
  title={Gallery of Gradients: Readable Gradient Code with Tangent},
  author={Tangent Contributors},
  year={2025},
  url={https://github.com/pedronahum/tangent/tree/master/examples}
}
```

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details

---

**Made with ❤️ by the Tangent community**

Questions? [Open an issue](https://github.com/pedronahum/tangent/issues) or start a [discussion](https://github.com/pedronahum/tangent/discussions)!
