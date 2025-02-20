import numpy as np
import pytest

from src.linear_regression import (
    add_intercept,
    compute_normal_equation,
    predict,
    cost_function,
    gradient_descent
)

def test_add_intercept_normal():
    # Test with a standard 2D matrix
    X = np.array([[2, 3], [4, 5], [6, 7]])
    X_aug = add_intercept(X)
    m, n = X.shape
    # Check new shape is (m, n+1)
    assert X_aug.shape == (m, n + 1)
    # Check first column is all ones and the rest equals X
    np.testing.assert_array_equal(X_aug[:, 0], np.ones(m))
    np.testing.assert_array_equal(X_aug[:, 1:], X)

def test_add_intercept_empty():
    # Test with an empty matrix (0 rows)
    X = np.empty((0, 3))
    X_aug = add_intercept(X)
    # Expect shape (0, 4)
    assert X_aug.shape == (0, 4)

def test_add_intercept_single_row():
    # Test with a single row matrix
    X = np.array([[10, 20]])
    X_aug = add_intercept(X)
    np.testing.assert_array_equal(X_aug, np.array([[1, 10, 20]]))

def test_compute_normal_equation_normal():
    # Create a simple design matrix with intercept already included.
    # For instance, let X = [[1, 1], [1, 2], [1, 3]] and y = [2, 3, 4].
    # The normal equation solution can be computed by np.linalg.lstsq.
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([2, 3, 4])
    beta_expected, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    beta_computed = compute_normal_equation(X, y)
    np.testing.assert_allclose(beta_computed, beta_expected, rtol=1e-5)

def test_compute_normal_equation_singular():
    # Test with a singular design matrix.
    # For example, if the second column is a multiple of the first.
    X = np.array([[1, 2], [1, 2], [1, 2]])
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        compute_normal_equation(X, y)

def test_compute_normal_equation_dimension_mismatch():
    # If y has an incompatible shape with X.
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1, 2])  # should have 3 elements
    with pytest.raises(ValueError):
        compute_normal_equation(X, y)

def test_predict_normal():
    # If predict implements a simple dot product.
    X = np.array([[1, 2], [3, 4]])
    beta = np.array([0.5, -1])
    y_pred_expected = X.dot(beta)
    y_pred = predict(X, beta)
    np.testing.assert_allclose(y_pred, y_pred_expected, rtol=1e-5)

def test_predict_dimension_mismatch():
    # If beta's shape doesn't match X's number of columns.
    X = np.array([[1, 2], [3, 4]])
    beta = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        predict(X, beta)

def test_cost_function_perfect_fit():
    # When predictions are exactly equal to y, cost should be zero.
    X = np.array([[1, 2], [1, 3]])
    beta = np.array([0, 1])
    y = X.dot(beta)
    cost = cost_function(X, y, beta)
    assert np.isclose(cost, 0)

def test_cost_function_normal():
    # Test with known values.
    X = np.array([[1, 0], [0, 1]])
    beta = np.array([1, 2])
    y = np.array([2, 3])
    # cost = (1/(2*2))*((2-1)^2 + (3-2)^2) = 1/4*(1+1)=0.5
    cost_expected = 0.5
    cost = cost_function(X, y, beta)
    assert np.isclose(cost, cost_expected)

def test_cost_function_dimension_mismatch():
    X = np.array([[1, 0], [0, 1]])
    beta = np.array([1, 2])
    y = np.array([1])  # wrong length
    with pytest.raises(ValueError):
        cost_function(X, y, beta)

def test_gradient_descent_normal():
    # Use a simple linear regression example:
    # Let X be with intercept included.
    X = np.array([[1, 1],
                  [1, 2],
                  [1, 3]])
    y = np.array([2, 3, 4])
    beta_init = np.zeros(X.shape[1])
    learning_rate = 0.1
    num_iterations = 1000
    beta, history = gradient_descent(X, y, beta_init, learning_rate, num_iterations)
    # The least-squares solution via np.linalg.lstsq
    beta_expected, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(beta, beta_expected, rtol=1e-3)
    # Also, ensure the cost history is non-increasing (or at least decreases initially)
    assert len(history) == num_iterations
    assert history[0] > history[-1]

def test_gradient_descent_zero_iterations():
    X = np.array([[1, 1],
                  [1, 2]])
    y = np.array([3, 4])
    beta_init = np.array([0, 0])
    beta, history = gradient_descent(X, y, beta_init, learning_rate=0.1, num_iterations=0)
    # With zero iterations, beta should equal beta_init and history should be empty.
    np.testing.assert_array_equal(beta, beta_init)
    assert history == []

def test_gradient_descent_dimension_mismatch():
    # beta_init has wrong dimension.
    X = np.array([[1, 2], [1, 3]])
    y = np.array([4, 5])
    beta_init = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        gradient_descent(X, y, beta_init, learning_rate=0.1, num_iterations=100)

def test_gradient_descent_negative_learning_rate():
    # We might decide that a negative learning rate is invalid.
    X = np.array([[1, 1],
                  [1, 2]])
    y = np.array([3, 4])
    beta_init = np.array([0, 0])
    with pytest.raises(ValueError):
        gradient_descent(X, y, beta_init, learning_rate=-0.1, num_iterations=10)
