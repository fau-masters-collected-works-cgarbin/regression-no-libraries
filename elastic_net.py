"""Fit a model using elastic net regression, without using libraries.

All calculations are done with vectors/matrices, to show each step of the process.

Reference: `An Introduction to Statistical Learrning`_, James et al., second edition, section
6.2 Shrinkage Methods.

.. _An Introduction to Statistical Learrning:
   https://web.stanford.edu/~hastie/ISLRv2_website.pdf
"""
import numpy as np


def fit(x: np.ndarray, y: np.ndarray, lmbda: float, alpha: float, iterations: int) -> np.ndarray:
    """Fit a linear regression model using elastic net regression.

    Args:
        x (np.ndarray): The features (predictors). Must be encoded and scaled as needed.
        y (np.ndarray): The target (response).
        lr (float): The learning rate (a.k.a. "alpha").
        lmbda (float): The regularization parameter. If set to 0, the model is not regularized (just least squares)
        alpha (float): The elastic net mixing parameter. If set to one, the model is ridge regression. If set to zero,
            the model is lasso regression.
        iterations (int): The number of iterations to run.

    Returns:
        np.ndarray: The coefficients of the elastic net regression model.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('X (inputs) and Y (outputs) must have the same number of rows')

    # Initialize the coefficients with random values with the best practices: uniformly distributed
    # within a small range (reshaped to a column vector)
    num_features = x.shape[1]  # a.k.a. "p"
    beta = np.random.uniform(-1, 1, num_features).reshape(num_features, 1)

    # Compute b_k only once because it doesn't depend on the coefficients (thus doesn't change
    # during the iterations)
    b_k = np.sum(np.square(x), axis=0)

    # Run the coordinate descent algorithm: update one coefficient (beta) at a time
    # Note that the update is cumulative, i.e. updating coefficient k+1 uses the updated value of coefficient k
    # Some of the expressions depend only on lambda and alpha and could be pulled out of the loop, but they are left
    # like to help illustrate the formulas
    for _ in range(iterations):
        for k in range(num_features):
            # The predictions with the current coefficients
            predictions = x @ beta
            # The residuals - how far off the predictions are from the actual values
            residuals = y - predictions

            # Some of the values we need, converted to column vectors to make the matrix multiplication work
            x_k = x[:, k].reshape(-1, 1)
            a_k = x_k.T @ (residuals + x_k @ beta[k].reshape(-1, 1))

            # Compute this term separately because we need to apply the positive function to it
            # Doing it all in one one would become difficult to follow
            regularized_a_k = abs(a_k) - (lmbda * (1 - alpha) / 2)
            regularized_a_k[regularized_a_k < 0] = 0

            # The elastic net upddate for the coefficient k
            beta[k] = np.sign(a_k) * regularized_a_k / (b_k[k] + lmbda * alpha)

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
