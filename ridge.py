"""Fit a model using Ridge regression, without using libraries.

All calculations are done with vectors/matrices, to show each step of the process.

Reference: `An Introduction to Statistical Learrning`_, James et al., second edition, section
6.2 Shrinkage Methods.

.. _An Introduction to Statistical Learrning:
   https://web.stanford.edu/~hastie/ISLRv2_website.pdf
"""
import numpy as np


def fit(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int) -> np.ndarray:
    """Fit a linear regression model using ridge regression.

    The loss function for ridge regression is:

        J(beta, lambda) = sum_1_N(y_i - sum_1_p(x_i @ beta)^2 + lambda * sum_1_p(beta_i^2)

    Where

        N: number of observations (rows)
        p: number of features (columns)
        beta: the coefficients
        lambda: the regularization parameter (shrinkage/penalty parameter)

    The goal is to minimize the loss function. We do that by computing the partial derivative of the loss function
    with respect to each coefficient. The partial derivative for each coefficient is:

       d(J(beta, lambda))/d(beta_i) = 2 * sum_1_N(x_i * (y_i - sum_1_p(x_i @ beta))) + 2 * lambda * beta_i

    This function uses gradient desceent to find the coefficients. We used vctorized operations to compute the
    graidient, thus all gradients are computed at once.

    Args:
        x (np.ndarray): The features (predictors). Must be encoded and scaled as needed.
        y (np.ndarray): The target (response). Must be centered.
        lr (float): The learning rate (a.k.a. "alpha").
        lmbda (float): The regularization parameter. If set to 0, the model is not regularized (just least squares).
        iterations (int): The number of iterations to run.

    Returns:
        np.ndarray: The coefficients of the ridge regression model.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('X (inputs) and Y (outputs) must have the same number of rows')

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
