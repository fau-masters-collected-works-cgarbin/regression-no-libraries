"""Fit a model using Ridge regression, without using libraries.

All calculations are done with vectors/matrices, to show each step of the process:
"""
import numpy as np


def fit(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int) -> np.ndarray:
    """Fit a linear regression model using ridge regression.

    Args:
        x (np.ndarray): The features (predictors). Must be encoded and scaled as needed.
        y (np.ndarray): The target (response).
        lr (float): The learning rate (a.k.a. "alpha").
        lmbda (float): The regularization parameter. If set to 0, the model is not regularized.
        iterations (int): The number of iterations to run.

    Returns:
        np.ndarray: The coefficients of the ridge regression model.
    """
    # Initialize the coefficients with random values with the best practices: uniformly distributed
    # within a small range (reshaped to a column vector)
    num_features = x.shape[1]  # a.k.a. "p"
    beta = np.random.uniform(-1, 1, num_features).reshape(num_features, 1)

    # Run the gradient descent algorithm
    # Could be done in one line, but this format allows to understand the role of each computation
    for _ in range(iterations):
        # The predictions with the current coefficients
        predictions = x @ beta
        # The residuals - how far off the predictions are from the actual values
        residuals = y - predictions
        # ridge regularization term (penalty) - it's an L2 norm, but we are calculating the
        # derivative at this point, so we no longer see the square of the norm
        penalty = lmbda * beta

        # Compute the regularized gradient, adjust with the learning rate and update
        beta = beta + lr * 2 * (x.T @ residuals - penalty)

    return beta


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict the output using the cofficients.

    Args:
        x (np.ndarray): The features (predictors). Must be encoded and scaled as needed.
        coefficients (np.ndarray): The coefficients for the model.

    Returns:
        np.ndarray: The predictions.
    """
    predictions = x @ coefficients
    return predictions
