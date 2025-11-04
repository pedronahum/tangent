"""Debug tuple return warning."""
import warnings
import tangent


def test_basic():
    """Test basic tuple return."""

    def f(x):
        return x ** 2, x * 3

    print("Creating gradient function...")
    warnings.simplefilter("always")

    df = tangent.grad(f)

    print(f"Gradient function created: {df}")

    result = df(2.0)
    print(f"Result: {result}")


if __name__ == '__main__':
    test_basic()
