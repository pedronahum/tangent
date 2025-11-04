# Dict Construction Bug - Multi-Key Dicts

## Status
**Bug**: Multi-key dict construction with differentiated values generates invalid code
**Priority**: Medium
**Complexity**: High

## Summary

When constructing a dict with multiple keys where values depend on differentiated variables, Tangent generates code with undefined `_` placeholders, causing a `NameError` at runtime.

## Reproduction

```python
import tangent

def multi_key_dict(x):
    d = {'a': x, 'b': x ** 2}
    return d['a'] + d['b']

df = tangent.grad(multi_key_dict)
result = df(2.0)  # NameError: name '_' is not defined
```

## Generated Code (Buggy)

```python
def dmulti_key_dictdx(x, bd_a_plus_d_b=1.0):
    x_to_the_2 = x ** 2
    d = {'a': x, 'b': x_to_the_2}
    d_b = _   # ❌ Should be: d_b = d['b']
    d_a = _   # ❌ Should be: d_a = d['a']
    bd = tangent.init_grad(d)

    # ... rest of gradient code
```

## Root Cause

The `_` placeholder comes from `tangent/template.py:131`:

```python
if not isinstance(slice_value, (gast.Subscript, gast.Name, gast.Constant)):
    # This happens when the gradient of a constant is taken
    if self.replace_grad == Replace.TANGENT:
        new_node = gast.Constant(value=0, kind=None)
    else:
        new_node = gast.Name(id='_', ctx=None, annotation=None)
        self.remove(new_node)  # Marks for removal, but gets left in
```

### The Issue

1. After dict construction `d = {'a': x, 'b': x_to_the_2}`, the reverse-mode AD needs to save the dict element values for use in the backward pass
2. Variables `d_a` and `d_b` are created to hold these values
3. But instead of generating `d_a = d['a']` and `d_b = d['b']`, the system generates `d_a = _` and `d_b = _`
4. The `_` placeholder is supposed to be removed, but ends up in the final code

## What Works

- ✅ **Single-key dicts**: `d = {'a': x}` works fine
- ✅ **Dicts as parameters**: `def f(x, config={'a': 1, 'b': 2}):` works perfectly
- ✅ **Global dicts**: `CONFIG = {'a': 1, 'b': 2}` accessed in function works

## Technical Details

### Affected Components

1. **`tangent/template.py`** (line 131) - Creates `_` placeholder
2. **`tangent/reverse_ad.py`** (lines 859-885) - `visit_Dict()` handles dict construction
3. **`tangent/anf.py`** (lines 168-173) - ANF transformation for dicts
4. **Forward pass generation** - Should create statements to save dict subscript values

### Expected Behavior

After dict construction, the forward pass should generate:

```python
d = {'a': x, 'b': x_to_the_2}
d_a = d['a']  # Save for backward pass
d_b = d['b']  # Save for backward pass
```

Instead, it generates:

```python
d = {'a': x, 'b': x_to_the_2}
d_a = _  # Undefined placeholder
d_b = _  # Undefined placeholder
```

## Proposed Fix

The fix requires changes to how dict subscripts are extracted and saved during the forward pass generation. Specifically:

1. **Option A**: Modify `visit_Dict()` in `reverse_ad.py` to generate explicit forward statements for saving dict values
2. **Option B**: Add special handling in ANF transformation to extract dict elements (similar to tuple unpacking in lines 199-218 of `anf.py`)
3. **Option C**: Fix the template system to properly fill in subscript expressions instead of `_` placeholders

### Recommended Approach (Option B)

Modify `tangent/anf.py` to handle dict construction similar to tuple unpacking:

```python
def visit_Assign(self, node):
    # ... existing code ...

    # Add handling for Dict construction
    elif isinstance(node.value, gast.Dict):
        # After dict assignment, extract each value for later use
        name = self.namer.name(node.targets[0])
        target = gast.Name(id=name, ctx=gast.Store(), annotation=None)

        for key_node, value_node in zip(node.value.keys, node.value.values):
            # Create: temp_var = dict[key]
            temp_name = self.namer.name(gast.Subscript(value=target, slice=key_node))
            stmt = gast.Assign(
                targets=[gast.Name(id=temp_name, ctx=gast.Store())],
                value=gast.Subscript(
                    value=gast.Name(id=name, ctx=gast.Load()),
                    slice=key_node,
                    ctx=gast.Load()))
            self.mark(stmt)
            self.append(stmt)

        node.targets[0] = target
```

## Complexity Assessment

**High Complexity** due to:
- Interaction between multiple transformation phases (ANF, activity analysis, reverse AD)
- Template system replacement logic
- Need to understand forward/backward pass generation
- Risk of breaking existing functionality

## Workarounds

1. **Pass dicts as parameters** (recommended):
   ```python
   def compute(x, config={'a': 1, 'b': 2}):
       return x * config['a'] + x * config['b']
   ```

2. **Use global dicts**:
   ```python
   CONFIG = {'a': 1, 'b': 2}
   def compute(x):
       return x * CONFIG['a'] + x * CONFIG['b']
   ```

3. **Use separate variables**:
   ```python
   def compute(x):
       a = x
       b = x ** 2
       return a + b  # Instead of d['a'] + d['b']
   ```

## Test Cases

When fixed, should pass:

```python
def test_multi_key_dict():
    """Test dict construction with multiple differentiated values."""
    def f(x):
        d = {'a': x, 'b': x ** 2, 'c': x * 3}
        return d['a'] + d['b'] + d['c']

    df = tangent.grad(f)
    result = df(2.0)
    expected = 1 + 4 + 3  # d/dx(x + x^2 + 3x) = 1 + 2x + 3 = 8
    assert np.isclose(result, expected)

def test_nested_dict_construction():
    """Test nested dict construction."""
    def f(x):
        d = {
            'model': {'scale': x, 'offset': x ** 2},
            'lr': 0.1
        }
        return d['model']['scale'] + d['model']['offset']

    df = tangent.grad(f)
    result = df(2.0)
    expected = 1 + 4  # d/dx(x + x^2) = 1 + 2x = 5
    assert np.isclose(result, expected)

def test_dict_modification():
    """Test empty dict + element assignment."""
    def f(x):
        d = {}
        d['a'] = x
        d['b'] = x ** 2
        return d['a'] + d['b']

    df = tangent.grad(f)
    result = df(2.0)
    expected = 1 + 4
    assert np.isclose(result, expected)
```

## References

- `/tmp/test_dict_construction.py` - Initial test showing the bug
- `/tmp/test_dict_edge_cases.py` - Comprehensive edge case tests
- `/tmp/debug_dict_multikey.py` - Debug script showing generated code
- `tangent/template.py:131` - Where `_` placeholder is created
- `tangent/reverse_ad.py:859` - Dict visiting in reverse mode
- `tangent/anf.py:168` - ANF transformation for dicts

## Impact

**Current**: Users must work around by passing dicts as parameters or using globals
**After Fix**: Users can construct dicts naturally within differentiated functions
**User Benefit**: Medium-High - dicts are common in ML code for grouping parameters

## Estimation

- **Investigation**: 4 hours (✅ Complete)
- **Implementation**: 8-12 hours
- **Testing**: 4 hours
- **Documentation**: 2 hours
- **Total**: 18-22 hours

## Related Issues

- Tuple unpacking works correctly (see `tangent/anf.py:199-218`)
- Single-key dicts work (simpler code path)
- List construction may have similar issues (untested)
