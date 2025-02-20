import numpy as np
import matplotlib.pyplot as plt
import matrix_ops


def add_intercept(X):
    """
    Adds an intercept term (a column of ones) to the design matrix X.
    
    Parameters:
        X (np.ndarray): An (m x n) matrix of features.
    
    Returns:
        X_augmented (np.ndarray): An (m x (n+1)) matrix with a column of ones.
    """
    m, n = X.shape
    X_augmented = np.empty((m, n+1), dtype=float)
    for i in range(m):
        X_augmented[i, 0] = 1
        for j in range(1, n+1):
            X_augmented[i, j] = X[i, j-1]
    return X_augmented


def compute_normal_equation(X, y):
    """
    Computes the OLS solution using the normal equation:
    
        beta = (X^T X)^{-1} X^T y
        
    Parameters:
        X (np.ndarray): An (m x n) design matrix.
        y (np.ndarray): An (m,) target vector.
        
    Returns:
        beta (np.ndarray): An (n,) vector of estimated parameters.
    """
    X_T = matrix_ops.transpose(X)
    X_T_X = matrix_ops.multiply_matrices(X_T, X)
    inverse = matrix_ops.inverse_matrix(X_T_X)
    product = matrix_ops.multiply_matrices(inverse, X_T)
    beta = matrix_ops.multiply_matrices(product, y)
    return beta


def predict(X, beta):
    """
    Predicts the output for a given design matrix X and parameter vector beta.
    
    Parameters:
        X (np.ndarray): An (m x n) design matrix.
        beta (np.ndarray): An (n,) parameter vector.
        
    Returns:
        y_pred (np.ndarray): An (m,) vector of predictions.
    """
    return matrix_ops.multiply_matrices(X, beta)


def cost_function(X, y, beta):
    """
    Computes the cost (loss) function for linear regression:
    
        J(beta) = (1/2m) * ||y - X beta||^2
        
    Parameters:
        X (np.ndarray): An (m x n) design matrix.
        y (np.ndarray): An (m,) vector of true targets.
        beta (np.ndarray): An (n,) vector of parameters.
        
    Returns:
        cost (float): The computed cost.
    """
    m = X.shape[0]
    y_T = matrix_ops.transpose(y)
    X_T = matrix_ops.transpose(X)
    beta_T = matrix_ops.transpose(beta)
    beta_T_X_T = matrix_ops.multiply_matrices(beta_T, X_T)
    beta_T_X_T_X = matrix_ops.multiply_matrices(beta_T_X_T, X)
    beta_T_X_T_X_beta = matrix_ops.multiply_matrices(beta_T_X_T_X, beta)
    X_T_X = matrix_ops.multiply_matrices(X_T, X)
    y_T_y = matrix_ops.multiply_matrices(y_T, y)
    beta_T_X_T = matrix_ops.multiply_matrices(beta_T, X_T)
    beta_T_X_T_y = matrix_ops.multiply_matrices(beta_T_X_T, y)
    cost = (1/(2*m))*(y_T_y - 2 * beta_T_X_T_y + beta_T_X_T_X_beta)
    return cost


def gradient_descent(X, y, beta_init, learning_rate, num_iterations):
    """
    Performs gradient descent to minimize the cost function.
    
    Parameters:
        X (np.ndarray): An (m x n) design matrix.
        y (np.ndarray): An (m,) vector of true targets.
        beta_init (np.ndarray): Initial guess for beta, shape (n,).
        learning_rate (float): The learning rate (alpha).
        num_iterations (int): Number of iterations to run gradient descent.
    
    Returns:
        beta (np.ndarray): The estimated parameters after gradient descent.
        history (list): The history of cost values at each iteration.
    """
    if learning_rate < 0:
        raise ValueError('Learning rate should be positive')
    beta = beta_init.copy()
    history = []
    m = X.shape[0]
    for i in range(num_iterations):
        X_beta = matrix_ops.multiply_matrices(X, beta)
        X_T = matrix_ops.transpose(X)
        gradient = (1/m) * matrix_ops.multiply_matrices(X_T, (X_beta - y))
        beta -= learning_rate * gradient
        cost = cost_function(X, y, beta)
        history.append(cost)
    return beta, history


if __name__ == "__main__":
    
    # Simulate Data
    np.random.seed(42)
    m = 100         # number of samples
    n_features = 2  # using 2 features for visualization

    # Generate random features 
    X = np.random.rand(m, n_features) * 10

    # True parameters (intercept, coef for feature1, coef for feature2)
    beta_true = np.array([3, 1.5, -2])

    # Add intercept to X
    X_augmented = add_intercept(X)

    # Generate true target values with some noise
    y_true = X_augmented.dot(beta_true) + np.random.randn(m) * 2

    # Compute Estimates
    # Normal Equation
    beta_normal = compute_normal_equation(X_augmented, y_true)
    y_pred_normal = matrix_ops.multiply_matrices(X_augmented, beta_normal)

    # Gradient Descent
    beta_init = np.zeros(X_augmented.shape[1])
    learning_rate = 0.01
    num_iterations = 500
    beta_grad, cost_history = gradient_descent(X_augmented, 
                                               y_true, 
                                               beta_init, 
                                               learning_rate, 
                                               num_iterations)
    y_pred_grad = matrix_ops.multiply_matrices(X_augmented, beta_grad)

    # Visualize the results
    plt.figure(figsize=(12, 5))

    # True vs Predicted values for both methods
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, 
                y_pred_normal, 
                label='Normal Equation', 
                color='blue',
                alpha=0.7)
    plt.scatter(y_true, 
                y_pred_grad, 
                label='Gradient Descent', 
                color='red', 
                marker='x', 
                alpha=0.7)

    # The ideal fit line (y = x)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'k--', 
             label='Ideal Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Comparison of Predictions')
    plt.legend()

    # Convergence of the Cost Function for Gradient Descent
    plt.subplot(1, 2, 2)
    plt.plot(cost_history, color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Cost Convergence')

    plt.tight_layout()
    plt.show()