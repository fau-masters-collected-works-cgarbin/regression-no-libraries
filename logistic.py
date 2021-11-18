"""Fit a model using logistic regression, without using libraries.

All calculations are done with vectors/matrices, to show each step of the process.

Reference: `An Introduction to Statistical Learrning`_, James et al., second edition, section
4.3 Logistic Regression.

.. _An Introduction to Statistical Learrning:
   https://web.stanford.edu/~hastie/ISLRv2_website.pdf
"""
import numpy as np
from sklearn.utils.extmath import softmax


def fit(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int) -> np.ndarray:
    """Fit a logistic regression model.

    Args:
        x (np.ndarray): The features (predictors). Must be encoded (categories to numbers) and scaled (standardized).
        y (np.ndarray): The target (response). Must be hot-encoded as an indicator matrix (one column per class).
        lr (float): The learning rate (a.k.a. "alpha").
        lmbda (float): The regularization parameter.
        iterations (int): The number of iterations to run.

    Returns:
        np.ndarray: The coefficients of the logistic regression model.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('X (inputs) and Y (outputs) must have the same number of rows')

    num_features = x.shape[1]  # a.k.a. "p"
    num_classes = y.shape[1]  # a.k.a. "k"

    # Add the column for beta 0 (intercept)
    x = np.insert(x, 0, 1.0, axis=1)

    # Initialize the coefficients matrix to zero (one per class in the target matrix)
    # Note that an extract row is added for beta 0 (intercept)
    beta = np.zeros((num_features+1, num_classes))

    # Transpose the features only once to save a bit of time
    x_t = x.T

    # Run the gradient descent algorithm
    # Could be done in one line, but this format allows to understand the role of each computation
    for _ in range(iterations):
        # The unnormalized class probability matrix
        u = np.exp(x @ beta)

        # The normalized class probability matrix
        # keepdims is needed to keep the same shape as u, so that the division is broadcasted
        # (see examples in https://stackoverflow.com/a/39442305)
        p = u / np.sum(u, axis=1, keepdims=True)

        # The intercept term, in matrix format, to faciliate the next step
        z = np.zeros_like(beta)
        z[0, :] = beta[0, :]

        # Update the coefficients
        beta = beta + lr * (x_t @ (y - p) - 2 * lmbda * (beta - z))

    return beta


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict the class probabilities for the input.

    Args:
        x (np.ndarray): The features (predictors). Must be encoded (categories to numbers) and scaled (standardized).
        coefficients (np.ndarray): The coefficients for the model.

    Returns:
        np.ndarray: The raw prediction values from the logistic regression model.
        np.ndarray: The class probabilities, calculated using softmax on the raw prediction values.
        np.ndarray: The class predictions, calculated using the class with the highest probability (argmax).

    """
    # Split the intercept from the coefficients
    beta0 = coefficients[0, :]
    betas = coefficients[1:, :]

    # keepdims is needed to keep the same shape, to broadcast the division
    predictions = np.exp(beta0 + x @ betas) / np.exp(beta0 + np.sum(x @ betas, axis=1, keepdims=True))

    # Probabilities with softmax
    probabilities = softmax(predictions)

    # Class predictions, as a row vector
    classes = np.argmax(probabilities, axis=1).reshape(-1, 1)

    return predictions, probabilities, classes
