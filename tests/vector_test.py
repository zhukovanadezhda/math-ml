import math
import pytest
from src.vector import Vector

def test_initialization_valid():
    v = Vector([1, 2, 3])
    assert v.data == [1, 2, 3]
    assert len(v) == 3

def test_initialization_invalid_data_type():
    # Non-list input should raise a TypeError.
    with pytest.raises(TypeError):
        Vector("not a list")
    # List containing non-numeric data should raise a TypeError.
    with pytest.raises(TypeError):
        Vector([1, "a", 3])

def test_len():
    v = Vector([1, 2, 3])
    assert len(v) == 3

def test_negation():
    v = Vector([1, -2, 3])
    neg_v = -v
    assert neg_v == Vector([-1, 2, -3])

def test_addition():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    result = v1 + v2
    assert result == Vector([5, 7, 9])
    
    # Addition with mismatched dimensions should raise ValueError.
    with pytest.raises(ValueError):
        _ = v1 + Vector([1, 2])

def test_subtraction():
    v1 = Vector([5, 7, 9])
    v2 = Vector([1, 2, 3])
    result = v1 - v2
    assert result == Vector([4, 5, 6])
    
    # Subtraction with mismatched dimensions should raise ValueError.
    with pytest.raises(ValueError):
        _ = v1 - Vector([1, 2])

def test_scalar_multiplication():
    v = Vector([1, 2, 3])
    result = v * 3
    assert result == Vector([3, 6, 9])
    
    # Check right-hand scalar multiplication.
    result = 3 * v
    assert result == Vector([3, 6, 9])
    
    # Multiplication with a non-scalar should raise TypeError.
    with pytest.raises(TypeError):
        _ = v * "a"

def test_equality():
    v1 = Vector([1, 2, 3])
    v2 = Vector([1, 2, 3])
    v3 = Vector([3, 2, 1])
    assert v1 == v2
    assert v1 != v3

def test_iteration_and_indexing():
    v = Vector([1, 2, 3])
    # Test iteration.
    for idx, element in enumerate(v):
        assert element == v[idx]
    # Test indexing.
    assert v[0] == 1
    assert v[1] == 2
    assert v[2] == 3

def test_str():
    v = Vector([1, 2, 3])
    # __str__ should return the string representation of self.data.
    assert str(v) == str([1, 2, 3])

def test_dot_product():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    # Dot product: 1*4 + 2*5 + 3*6 = 32
    assert v1.dot(v2) == 32
    
    # Dot product with mismatched dimensions should raise ValueError.
    with pytest.raises(ValueError):
        v1.dot(Vector([1, 2]))

def test_norm():
    v = Vector([3, 4])
    # Norm should be sqrt(3^2 + 4^2) = 5.0.
    assert math.isclose(v.norm(), 5.0)
    
    v = Vector([1, 2, 2])
    # Norm should be sqrt(1+4+4)=3.
    assert math.isclose(v.norm(), 3.0)

def test_normalize():
    v = Vector([3, 4])
    normalized_v = v.normalize()
    expected = Vector([3/5, 4/5])
    for a, b in zip(normalized_v.data, expected.data):
        assert math.isclose(a, b)
    
    # Normalizing a zero vector should raise a ValueError.
    with pytest.raises(ValueError):
        Vector([0, 0, 0]).normalize()
