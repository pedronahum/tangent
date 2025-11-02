# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for function caching system."""

import tangent
import pytest


def test_basic_caching():
    """Test that basic caching works correctly."""
    # Clear cache first
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    # First call should be a cache miss
    df1 = tangent.grad(f)
    stats = tangent.get_cache_stats()
    assert stats['misses'] == 1
    assert stats['hits'] == 0

    # Second call with same function should be a cache hit
    df2 = tangent.grad(f)
    stats = tangent.get_cache_stats()
    assert stats['misses'] == 1
    assert stats['hits'] == 1

    # Both gradient functions should produce same results
    assert df1(3.0) == df2(3.0)
    assert df1(3.0) == 6.0


def test_cache_different_functions():
    """Test that different functions get separate cache entries."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    def g(x):
        return x * x * x

    # Both should be cache misses
    df = tangent.grad(f)
    dg = tangent.grad(g)

    stats = tangent.get_cache_stats()
    assert stats['misses'] == 2
    assert stats['hits'] == 0

    # Results should be different
    assert df(3.0) == 6.0
    assert dg(3.0) == 27.0


def test_cache_different_parameters():
    """Test that different parameters create separate cache entries."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x, y):
        return x * x + y * y

    # Different wrt parameters
    df_x = tangent.grad(f, wrt=(0,))
    df_y = tangent.grad(f, wrt=(1,))

    stats = tangent.get_cache_stats()
    assert stats['misses'] == 2
    assert stats['hits'] == 0

    # Same parameters should hit cache
    df_x2 = tangent.grad(f, wrt=(0,))
    stats = tangent.get_cache_stats()
    assert stats['hits'] == 1


def test_cache_optimized_flag():
    """Test that optimized flag affects caching."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    # Different optimization settings
    df_opt = tangent.grad(f, optimized=True)
    df_unopt = tangent.grad(f, optimized=False)

    stats = tangent.get_cache_stats()
    assert stats['misses'] == 2
    assert stats['hits'] == 0


def test_cache_preserve_result_flag():
    """Test that preserve_result flag affects caching."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    # Different preserve_result settings
    df1 = tangent.grad(f, preserve_result=False)
    df2 = tangent.grad(f, preserve_result=True)

    stats = tangent.get_cache_stats()
    assert stats['misses'] == 2
    assert stats['hits'] == 0

    # Check they return different types
    result1 = df1(3.0)
    result2 = df2(3.0)

    assert isinstance(result1, (int, float))
    assert isinstance(result2, tuple)
    assert result2[0] == result1  # gradient should match
    assert result2[1] == 9.0      # original result


def test_clear_cache():
    """Test that clear_cache removes all cached entries."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    # Create a cached entry
    df1 = tangent.grad(f)
    stats = tangent.get_cache_stats()
    assert stats['size'] == 1

    # Clear cache
    tangent.clear_cache()
    stats = tangent.get_cache_stats()
    assert stats['size'] == 0

    # Next call should be a cache miss
    df2 = tangent.grad(f)
    stats = tangent.get_cache_stats()
    assert stats['misses'] == 2  # Original miss + this miss


def test_cache_size_limit():
    """Test that cache respects size limit."""
    tangent.clear_cache()
    tangent.reset_cache_stats()
    tangent.set_cache_size(2)

    def make_func(n):
        """Create a unique function."""
        def f(x):
            return x ** n
        # Make function unique by modifying its name
        f.__name__ = f'f{n}'
        return f

    # Create 3 functions (more than cache size)
    f1 = make_func(2)
    f2 = make_func(3)
    f3 = make_func(4)

    df1 = tangent.grad(f1)
    df2 = tangent.grad(f2)
    df3 = tangent.grad(f3)  # Should evict f1

    stats = tangent.get_cache_stats()
    assert stats['evictions'] >= 1
    assert stats['size'] == 2

    # Restore default cache size
    tangent.set_cache_size(128)


def test_set_cache_size():
    """Test set_cache_size function."""
    original_size = tangent.get_cache_size()

    tangent.set_cache_size(10)
    assert tangent.get_cache_size() == 10

    tangent.set_cache_size(100)
    assert tangent.get_cache_size() == 100

    # Test invalid size
    with pytest.raises(ValueError):
        tangent.set_cache_size(0)

    with pytest.raises(ValueError):
        tangent.set_cache_size(-1)

    # Restore original
    tangent.set_cache_size(original_size)


def test_reset_cache_stats():
    """Test that reset_cache_stats resets counters."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    # Generate some stats
    df1 = tangent.grad(f)
    df2 = tangent.grad(f)

    stats = tangent.get_cache_stats()
    assert stats['hits'] > 0 or stats['misses'] > 0

    # Reset stats
    tangent.reset_cache_stats()
    stats = tangent.get_cache_stats()
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    # But cache should still have entries
    assert stats['size'] > 0


def test_cache_hit_rate():
    """Test that hit rate is calculated correctly."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    # One miss, then three hits
    df = tangent.grad(f)
    df = tangent.grad(f)
    df = tangent.grad(f)
    df = tangent.grad(f)

    stats = tangent.get_cache_stats()
    assert stats['hits'] == 3
    assert stats['misses'] == 1
    assert abs(stats['hit_rate'] - 0.75) < 0.01


def test_autodiff_caching():
    """Test that autodiff function also uses caching."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x):
        return x * x

    # Test autodiff caching
    df1 = tangent.autodiff(f, mode='reverse')
    stats = tangent.get_cache_stats()
    assert stats['misses'] == 1

    df2 = tangent.autodiff(f, mode='reverse')
    stats = tangent.get_cache_stats()
    assert stats['hits'] == 1

    # Results should match
    assert df1(3.0, 1.0) == df2(3.0, 1.0)


def test_cache_with_multiple_wrt():
    """Test caching with multiple arguments."""
    tangent.clear_cache()
    tangent.reset_cache_stats()

    def f(x, y):
        return x * x + y * y * y

    # Different wrt tuples
    df_0 = tangent.grad(f, wrt=(0,))
    df_1 = tangent.grad(f, wrt=(1,))
    df_01 = tangent.grad(f, wrt=(0, 1))

    stats = tangent.get_cache_stats()
    assert stats['misses'] == 3
    assert stats['hits'] == 0

    # Same wrt should hit cache
    df_0_again = tangent.grad(f, wrt=(0,))
    stats = tangent.get_cache_stats()
    assert stats['hits'] == 1


def test_cache_correctness():
    """Test that cached functions produce correct results."""
    tangent.clear_cache()

    def polynomial(x):
        return 3.0 * x * x + 2.0 * x + 1.0

    # Get gradient (will be cached)
    df = tangent.grad(polynomial)

    # Test multiple times (should use cached version)
    for _ in range(5):
        assert abs(df(2.0) - 14.0) < 0.001  # 6*2 + 2 = 14
        assert abs(df(0.0) - 2.0) < 0.001   # 6*0 + 2 = 2
        assert abs(df(1.0) - 8.0) < 0.001   # 6*1 + 2 = 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
