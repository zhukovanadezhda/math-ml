import math
import numpy as np
from numba import njit


@njit
def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Add two matrices element-wise.

    Args:
        A (np.ndarray): A matrix.
        B (np.ndarray): Another matrix.

    Raises:
        ValueError: If the matrices are not equal in shape.

    Returns:
        np.ndarray: The sum of the two matrices.
    """
    if A.shape != B.shape:
        raise ValueError('Arrays are not equal.')
    else:
        m, n = A.shape
        C = np.empty((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                C[i, j] = A[i, j] + B[i, j]
    return C


@njit
def subtract_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Subtract two matrices element-wise.

    Args:
        A (np.ndarray): A matrix.
        B (np.ndarray): Another matrix.

    Raises:
        ValueError: If the matrices are not equal in shape.

    Returns:
        np.ndarray: The difference of the two matrices.
    """
    if A.shape != B.shape:
        raise ValueError('Arrays are not equal.')
    else:
        m, n = A.shape
        C = np.empty((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                C[i, j] = A[i, j] - B[i, j]
    return C


@njit
def scalar_multiply(A: np.ndarray, scalar: float) -> np.ndarray:
    """Multiply a matrix by a scalar.

    Args:
        A (np.ndarray): A matrix.
        scalar (float): A scalar

    Returns:
        np.ndarray: The matrix multiplied by the scalar.
    """
    m, n = A.shape
    C = np.empty((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            C[i, j] = A[i, j] * scalar
    return C


@njit
def hadamard_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the Hadamard product of two matrices.
    
    Args:
        A (np.ndarray): A matrix.
        B (np.ndarray): Another matrix.
        
    Raises:
        ValueError: If the matrices are not equal in shape.
        
    Returns:
        np.ndarray: The Hadamard product of the two matrices.
    """
    if A.shape != B.shape:
        raise ValueError('Arrays are not equal.')
    else:
        m, n = A.shape
        C = np.empty((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                C[i, j] = A[i, j] * B[i, j]
    return C


def to_2d(A: np.ndarray, side: str) -> np.ndarray:
    """
    Convert a 1D numpy array A to a 2D array.
      - If side=="left", returns a row vector (1, n).
      - If side=="right", returns a column vector (n, 1).
    If A is already 2D, returns it unchanged.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if A.ndim == 1:
        n = A.shape[0]
        if side == "left":
            res = np.empty((1, n), dtype=float)
            for i in range(n):
                res[0, i] = A[i]
            return res
        elif side == "right":
            res = np.empty((n, 1), dtype=float)
            for i in range(n):
                res[i, 0] = A[i]
            return res
        else:
            raise ValueError("Side must be 'left' or 'right'")
    elif A.ndim == 2:
        return A
    else:
        raise ValueError("Only 1D or 2D arrays are supported")


@njit
def matrix_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two 2D arrays A and B.
    Assumes that A and B are 2D and that A.shape[1] == B.shape[0].
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError('Incompatible dimensions for matrix multiplication.')
    rows = A.shape[0]
    cols = B.shape[1]
    common = A.shape[1]
    result = np.empty((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            s = 0
            for k in range(common):
                s += A[i, k] * B[k, j]
            result[i, j] = s
    return result


def multiply_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two numpy arrays A and B, mimicking np.dot.
    
    Conversion rules:
      - If A is 1D, treat it as a row vector.
      - If B is 1D, treat it as a column vector.
      
    After multiplication, post-process the result:
      - Matrix (2D) * vector (1D) returns a 1D array.
      - Vector (1D) * matrix (2D) returns a 1D array.
      - Vector (1D) * vector (1D) returns a scalar.
      - Otherwise, returns a 2D array.
    
    Raises:
      - TypeError if inputs are not numpy arrays.
      - ValueError if inputs are not 1D or 2D.
      - ValueError if the inner dimensions do not match.
    """
    # Only 1D and 2D arrays are supported.
    if A.ndim not in (1, 2):
        raise ValueError("Only 1D or 2D arrays are supported for A")
    if B.ndim not in (1, 2):
        raise ValueError("Only 1D or 2D arrays are supported for B")
    
    orig_A_ndim = A.ndim
    orig_B_ndim = B.ndim

    # Convert 1D inputs to 2D.
    A2 = to_2d(A, "left")
    B2 = to_2d(B, "right")
    
    # Call the jitted function.
    res = matrix_matrix_multiply(A2, B2)
    
    # Post-process to mimic np.dot output.
    if orig_A_ndim == 2 and orig_B_ndim == 1:
        # Matrix * vector: return a 1D array.
        return res[:, 0]
    elif orig_A_ndim == 1 and orig_B_ndim == 2:
        # Vector * matrix: return a 1D array.
        return res[0, :]
    elif orig_A_ndim == 1 and orig_B_ndim == 1:
        # Vector dot product: return a scalar.
        return res[0, 0]
    else:
        return res


@njit
def transpose(A: np.ndarray) -> np.ndarray:
    """Compute the transpose of a matrix or vector.
    
    For a 2D array, swaps its dimensions. For a 1D array, interprets it as a 
    row vector and returns a column vector.
    
    Args:
        A (np.ndarray): A matrix or vector.
        
    Returns:
        np.ndarray: The transpose of the matrix.
    """
    if A.ndim == 1:
        # Treat a 1D array as a row vector.
        n = A.shape[0]
        C = np.empty((n, ), dtype=float)
        for i in range(n):
            C[i, ] = A[i]
        return C
    elif A.ndim == 2:
        m, n = A.shape
        C = np.empty((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                C[i, j] = A[j, i]
        return C
    else:
        raise ValueError("Input array must be 1D or 2D.")


@njit
def minor_matrix(A: np.ndarray, row_index: int, col_index: int) -> np.ndarray:
    """Compute the minor matrix of a given matrix.
    
    Args:
        A (np.ndarray): A matrix.
        row_index (int): The row index to remove.
        col_index (int): The column index to remove.
        
    Returns:
        np.ndarray: The minor matrix.
    """
    m, n = A.shape
    if row_index >= m or col_index >= n:
        raise IndexError('Index is out of range.')
    C = np.empty((m-1, n-1), dtype=float)
    for i in range(m-1):
        for j in range(n-1):
            if i < row_index and j < col_index:
                C[i, j] = A[i, j]
            elif i < row_index and j >= col_index:
                C[i, j] = A[i, j+1]
            elif i >= row_index and j < col_index:
                C[i, j] = A[i+1, j]
            elif i >= row_index and j >= col_index:
                C[i, j] = A[i+1, j+1]
    return C


@njit
def determinant(A: np.ndarray) -> float:
    """Compute the determinant of a matrix.
    
    Args:
        A (np.ndarray): A square matrix.
        
    Raises:
        ValueError: If the matrix is not square.
        
    Returns:
        float: The determinant of the matrix.
    """
    m, n = A.shape
    if m != n:
        raise ValueError('Matrix should be square.')
    if n == 1 and m == 1:
        return A[0, 0]
    else:
        det = 0.0
        for i in range(m):
            det += (-1)**i * A[i, 0] * determinant(minor_matrix(A, i, 0))
        return det


@njit
def trace(A: np.ndarray) -> float:
    """Compute the trace of a matrix.
    
    Args:
        A (np.ndarray): A square matrix.
        
    Raises:
        ValueError: If the matrix is not square.
        
    Returns:
        float: The trace of the matrix.
    """
    m, n = A.shape
    if m != n:
        raise ValueError('Matrix should be square.')
    trace = 0.0
    for i in range(m):
        trace += A[i, i]
    return trace


@njit
def frobenius_norm(A: np.ndarray) -> float:
    """Compute the Frobenius norm of a matrix.
    
    Args:
        A (np.ndarray): A matrix.
        
    Returns:
        float: The Frobenius norm of the matrix.
    """
    m, n = A.shape
    sum_sq = 0.0
    for i in range(m):
        for j in range(n):
            sum_sq += A[i, j]**2
    return math.sqrt(sum_sq)


@njit
def inverse_matrix(A: np.ndarray) -> np.ndarray:
    """Compute the inverse of a matrix.
    
    Args:
        A (np.ndarray): A square matrix.
        
    Raises:
        ValueError: If the matrix is not square or singular.
        
    Returns:
        np.ndarray: The inverse of the matrix.
    """
    if determinant(A) == 0:
        raise ValueError('Matrix should be non-singular.')
    m, n = A.shape
    C = A * 1.0
    I = np.identity(m)
    for row in range(m):
        pivot = C[row, row]
        for col in range(n):
            C[row, col] /= pivot
            I[row, col] /= pivot
        for another_row in range(m):
            if another_row != row:
                coeff = C[another_row, row]
                for col in range(n):
                    C[another_row, col] -= C[row, col] * coeff
                    I[another_row, col] -= I[row, col] * coeff
    return I


@njit
def lu_decomposition(A: np.ndarray) -> tuple:
    """Compute the LU decomposition of a matrix with Doolittle’s Algorithm.
    
    Args:
        A (np.ndarray): A square matrix.
        
    Raises:
        ValueError: If the matrix is not square.
        
    Returns:
        np.ndarray: The lower triangular matrix.
        np.ndarray: The upper triangular matrix.
    """
    m, n = A.shape
    if m != n:
        raise ValueError('Matrix should be square.')
    U = np.identity(m)
    L = np.identity(m)
    for j in range(n):
        U[0, j] = A[0, j]
    for i in range(m):
        for j in range(i, n):
            u_tmp_sum = 0.0
            for k in range(i):
                u_tmp_sum += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - u_tmp_sum
        for j in range(i+1, n):
            l_tmp_sum = 0.0
            for k in range(i):
                l_tmp_sum += L[j, k] * U[k, i]
            L[j, i] = 1/U[i, i] * (A[j, i] - l_tmp_sum)
    print(L, '\n', U)
    return L, U


@njit
def upper_triangular(A: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    """Compute the upper triangular form of a matrix using Gaussian elimination.
    
    Args:
        A (np.ndarray): A matrix.
        b (np.ndarray): A vector.
        
    Raises:
        ValueError: If the matrix and vector are not compatible.
        
    Returns:
        np.ndarray: The upper triangular matrix.
        np.ndarray: The transformed vector
    """
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("b must have the same number of rows as A.")
    U = A * 1.0
    c = b * 1.0
    for row in range(m):
        pivot = U[row, row]
        for another_row in range(row+1, m):
            coeff = U[another_row, row] / pivot
            for col in range(n):
                U[another_row, col] -= U[row, col] * coeff
            c[another_row] -= c[row] * coeff
    return U, c


@njit
def back_substitution(U: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Perform back substitution to solve a system of equations.
    
    Args:
        U (np.ndarray): An upper triangular matrix.
        c (np.ndarray): A transformed vector.
        
    Returns:
        np.ndarray: The solution vector.
    """
    m, n = U.shape
    x = np.empty(n, dtype=float)
    x[n-1] = c[n-1] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum_others = 0.0
        for j in range(i+1, n):
            sum_others += U[i, j] * x[j]
        x[i] = 1 / U[i, i] * (c[i] - sum_others)
    return x


@njit
def solve_gaussian(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a system of linear equations using Gaussian elimination.
    
    Args:
        A (np.ndarray): A matrix.
        b (np.ndarray): A vector.
        
    Raises:
        ValueError: If the matrix is singular.
        
    Returns:
        np.ndarray: The solution vector.
    """
    if determinant(A) == 0:
        raise ValueError('Matrix should be non-singular.')
    U, c = upper_triangular(A, b)
    x = back_substitution(U, c)
    return x
