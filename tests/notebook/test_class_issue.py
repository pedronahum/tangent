"""Debug class issue in notebook"""
import tangent

print("Test 1: Class defined in function (like test)")
print("="*50)

def test_local_class():
    class Polynomial:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def evaluate(self, x):
            return self.a * x ** 2 + self.b * x + self.c

    def loss_with_class(x):
        poly = Polynomial(2.0, 3.0, 1.0)
        return poly.evaluate(x)

    try:
        dloss = tangent.grad(loss_with_class)
        gradient = dloss(5.0)
        print(f"✓ Gradient: {gradient}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

result1 = test_local_class()

print("\n\nTest 2: Class defined at module level")
print("="*50)

# Define at module level
class Polynomial:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x):
        return self.a * x ** 2 + self.b * x + self.c

def loss_with_class(x):
    poly = Polynomial(2.0, 3.0, 1.0)
    return poly.evaluate(x)

try:
    dloss = tangent.grad(loss_with_class)
    gradient = dloss(5.0)
    expected = 4 * 5.0 + 3
    print(f"✓ Gradient: {gradient}")
    print(f"  Expected: {expected}")
    print(f"  Match: {abs(gradient - expected) < 1e-5}")
    result2 = True
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    result2 = False

print("\n\nSummary:")
print(f"Local class (test style): {'✓' if result1 else '✗'}")
print(f"Module class (notebook style): {'✓' if result2 else '✗'}")
