import numpy as np
import pytest

from src.matrix_ops import (
    add_matrices,
    subtract_matrices,
    scalar_multiply,
    hadamard_product,
    multiply_matrices,
    transpose,
    minor_matrix,
    determinant,
    trace,
    frobenius_norm,
    inverse_matrix,
    lu_decomposition,
    upper_triangular,
    back_substitution,
    solve_gaussian,
)


def test_add_matrices_normal():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    expected = np.array([[6, 8], [10, 12]])
    np.testing.assert_array_equal(add_matrices(A, B), expected)


def test_add_matrices_mismatched():
    A = np.array([[1, 2]])
    B = np.array([[3, 4], [5, 6]])
    with pytest.raises(ValueError):
        add_matrices(A, B)


def test_subtract_matrices_normal():
    A = np.array([[5, 6], [7, 8]])
    B = np.array([[1, 2], [3, 4]])
    expected = np.array([[4, 4], [4, 4]])
    np.testing.assert_array_equal(subtract_matrices(A, B), expected)


def test_subtract_matrices_mismatched():
    A = np.array([[1, 2]])
    B = np.array([[3, 4], [5, 6]])
    with pytest.raises(ValueError):
        subtract_matrices(A, B)


def test_scalar_multiply_normal():
    A = np.array([[1, -2], [3, 4]])
    scalar = 2
    expected = np.array([[2, -4], [6, 8]])
    np.testing.assert_array_equal(scalar_multiply(A, scalar), expected)


def test_scalar_multiply_zero():
    A = np.array([[1, 2], [3, 4]])
    scalar = 0
    expected = np.zeros_like(A)
    np.testing.assert_array_equal(scalar_multiply(A, scalar), expected)


def test_hadamard_product_normal():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    expected = np.array([[5, 12], [21, 32]])
    np.testing.assert_array_equal(hadamard_product(A, B), expected)


def test_hadamard_product_mismatched():
    A = np.array([[1, 2]])
    B = np.array([[3, 4], [5, 6]])
    with pytest.raises(ValueError):
        hadamard_product(A, B)


def test_multiply_matrices_matrix_matrix_valid():
    # (2x3) * (3x2) = (2x2)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])
    expected = np.dot(A, B)
    result = multiply_matrices(A, B)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_multiply_matrices_vector_vector_valid():
    # Both inputs 1D: dot product (should return a scalar).
    v = np.array([1, 2, 3])
    w = np.array([4, 5, 6])
    expected = np.dot(v, w)
    result = multiply_matrices(v, w)
    # np.dot returns a scalar.
    assert np.isscalar(result)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_multiply_matrices_matrix_vector_valid():
    # 2D matrix times 1D vector: np.dot returns a 1D array.
    A = np.array([[1, 2], [3, 4]])
    v = np.array([5, 6])
    expected = np.dot(A, v)  # Expected shape: (2,)
    result = multiply_matrices(A, v)
    assert result.ndim == 1
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_multiply_matrices_vector_matrix_valid():
    # 1D vector times 2D matrix: np.dot returns a 1D array.
    v = np.array([1, 2])
    A = np.array([[3, 4], [5, 6]])
    expected = np.dot(v, A)  # Expected shape: (2,)
    result = multiply_matrices(v, A)
    assert result.ndim == 1
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_multiply_matrices_incompatible_shapes_matrix_matrix():
    # (2x2) * (1x3) should raise ValueError.
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6, 7]])
    with pytest.raises(ValueError):
        multiply_matrices(A, B)


def test_multiply_matrices_incompatible_shapes_vector_vector():
    # Dot product with vectors of different lengths.
    v = np.array([1, 2, 3])
    w = np.array([4, 5])
    with pytest.raises(ValueError):
        multiply_matrices(v, w)


def test_multiply_matrices_incompatible_shapes_matrix_vector():
    # A is (2x3) and v is length 2 (inner dims mismatch).
    A = np.array([[1, 2, 3], [4, 5, 6]])
    v = np.array([7, 8])
    with pytest.raises(ValueError):
        multiply_matrices(A, v)


def test_multiply_matrices_incompatible_shapes_vector_matrix():
    # v (length 2) times A (3x2) should fail.
    v = np.array([1, 2])
    A = np.array([[3, 4], [5, 6], [7, 8]])
    with pytest.raises(ValueError):
        multiply_matrices(v, A)


def test_multiply_matrices_invalid_ndim_A():
    # A 3D array is not supported.
    A = np.ones((2, 2, 2))
    B = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        multiply_matrices(A, B)


def test_multiply_matrices_invalid_ndim_B():
    # B 3D array is not supported.
    A = np.array([[1, 2], [3, 4]])
    B = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        multiply_matrices(A, B)


def test_transpose_normal():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.transpose(A)
    np.testing.assert_array_equal(transpose(A), expected)


def test_transpose_empty():
    A = np.array([[]])
    expected = np.transpose(A)
    np.testing.assert_array_equal(transpose(A), expected)


def test_minor_matrix_normal():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Remove row 0 and column 1:
    expected = np.array([[4, 6], [7, 9]])
    np.testing.assert_array_equal(minor_matrix(A, 0, 1), expected)


def test_minor_matrix_invalid_index():
    A = np.array([[1, 2], [3, 4]])
    with pytest.raises(IndexError):
        minor_matrix(A, 2, 0)
    with pytest.raises(IndexError):
        minor_matrix(A, 0, 2)


def test_determinant_1x1():
    A = np.array([[5]])
    assert np.isclose(determinant(A), 5)


def test_determinant_2x2():
    A = np.array([[1, 2], [3, 4]])
    expected = 1 * 4 - 2 * 3  # -2
    assert np.isclose(determinant(A), expected)


def test_determinant_3x3():
    A = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    expected = -306
    assert np.isclose(determinant(A), expected)


def test_determinant_non_square():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        determinant(A)


def test_trace_normal():
    A = np.array([[1, 2], [3, 4]])
    expected = 1 + 4
    assert np.isclose(trace(A), expected)


def test_trace_non_square():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        trace(A)


def test_frobenius_norm_normal():
    A = np.array([[1, 2], [3, 4]])
    expected = np.linalg.norm(A, "fro")
    assert np.isclose(frobenius_norm(A), expected)


def test_inverse_matrix_normal():
    A = np.array([[4, 7], [2, 6]])
    expected = np.linalg.inv(A)
    np.testing.assert_allclose(inverse_matrix(A), expected, rtol=1e-5)


def test_inverse_matrix_singular():
    A = np.array([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        inverse_matrix(A)


def test_inverse_matrix_non_square():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        inverse_matrix(A)


def test_lu_decomposition_normal():
    A = np.array([[2, 3, 1], [4, 7, 5], [6, 18, 22]])
    L, U = lu_decomposition(A)
    np.testing.assert_allclose(np.dot(L, U), A, rtol=1e-5)
    # Check L is lower triangular with ones on the diagonal.
    assert np.allclose(np.diag(L), np.ones(L.shape[0]))
    # Check U is upper triangular.
    assert np.allclose(U, np.triu(U))


def test_lu_decomposition_non_square():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        lu_decomposition(A)


def test_upper_triangular_normal():
    A = np.array([[2, 3], [4, 5]], dtype=float)
    b = np.array([1, 2], dtype=float)
    U, c = upper_triangular(A, b)
    # U must be upper triangular.
    assert np.allclose(U, np.triu(U))
    # c should have same shape as b.
    assert c.shape == b.shape


def test_upper_triangular_zero_pivot():
    # This matrix forces a zero pivot unless pivoting is implemented.
    A = np.array([[0, 1], [1, 2]], dtype=float)
    b = np.array([1, 2], dtype=float)
    with pytest.raises(ValueError):
        upper_triangular(A, b)


def test_upper_triangular_mismatched_dimensions():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        upper_triangular(A, b)


def test_back_substitution_normal():
    U = np.array([[2, 3], [0, 4]], dtype=float)
    c = np.array([5, 6], dtype=float)
    x = back_substitution(U, c)
    np.testing.assert_allclose(np.dot(U, x), c, rtol=1e-5)


def test_back_substitution_zero_diagonal():
    U = np.array([[1, 2], [0, 0]], dtype=float)
    c = np.array([3, 4], dtype=float)
    with pytest.raises(ZeroDivisionError):
        back_substitution(U, c)


def test_solve_gaussian_normal():
    A = np.array([[2, 3, -1], [4, 1, 2], [-2, 5, 1]], dtype=float)
    b = np.array([5, 6, 1], dtype=float)
    x = solve_gaussian(A, b)
    np.testing.assert_allclose(np.dot(A, x), b, rtol=1e-5)


def test_solve_gaussian_singular():
    A = np.array([[1, 2], [2, 4]], dtype=float)
    b = np.array([3, 6], dtype=float)
    with pytest.raises(ValueError):
        solve_gaussian(A, b)


def test_solve_gaussian_non_square():
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    b = np.array([7, 8], dtype=float)
    with pytest.raises(ValueError):
        solve_gaussian(A, b)
