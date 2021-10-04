"""Utility functions to work with datasets and perform statistical computations.

Some functions, e.g. predicting values, calculating MSE, etc. are simple and could have been done
in place. The code has functions even for those cases to make the intent of each piece of code
explicit. The goal is to be educational, not concise.
"""
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def check_python_version():
    """Check that we have the correct Python version for the code."""
    if sys.version_info < (3, 6):
        raise RuntimeError("Python 3.6 of higher is required to run this code.")


def read_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset from a file and return NumPy arrays.

    It reads first into a Pandas DataFrame, then converts to NumPy arrays because the NumPy function
    to read from a file, genfromtxt, is harder to use when strings are present. When specifying the
    `dtype` parameter, it returns a 1D array with structured data for each row, istead of the N x p
    matrix we want.

    Integer columns are converted to float, in preparation for code that needs to perform math
    operations on them.

    Categorical (string) columns are untouched. The caller code is responsible for encoding them
    as needed.

    Args:
        filename (str): The name of the file to read.

    Returns:
        np.ndarray, np.ndarray: The N x p feature matrix and N x 1 target vector.
    """
    dataset = pd.read_csv(filename)

    # Ensure that all numeric columns are floats (not integers) to help with the standardization step
    int_cols = dataset.columns[dataset.dtypes.eq('int')]
    dataset[int_cols] = dataset[int_cols].astype(float)

    # Split into input and output (asumming the last column is the output)
    # And convert to NumPy arrays
    x = dataset.iloc[:, :-1].to_numpy()
    y = dataset.iloc[:, -1:].to_numpy()
    return x, y


def encode_binary_cateogry(x: np.ndarray, column: int, one_value: str) -> np.ndarray:
    """Encode a binary (two values) column of a matrix in place.

    The function assumes that there are only two catoegories. One of them is encoded as 1.0, the
    other is encoded as 0.0. This is sometimes called "dummy enconding", other times it called
    "hot encoding". To avoid controversy, the function is called just "encode...".

    The original column is replaced with the encoded version.

    The encoded value is stored as float (not integer), in preparation for later steps that need
    to perform math operations on it.

    Args:
        x (np.ndarray): The feature matrix.
        column (int): The column to encode. IMPORTANT: assumes that it has only two categories.
        one_value (str): The string to use as "1.0" for the encoded column.
    """
    encoded = x[:, column] == one_value
    x[:, column] = encoded.astype(float)


def scale(m: np.ndarray):
    """Standardize a matrix: center (mean of zero) and standard deviation of one.

    Standardization is done in place. The original matrix is modified.

    All columns must be float. If there are categorical columns in the dataset, they must be
    encoded to float (1.0 or 0.0) before this function is called.

    Note that if the matrix has only one column, , but the standard
    deviation is not changed. We only need to scale a matrix if it has more than one column. This
    usually applies to the output matrix. It's more likely that it has only one column.

    Args:
        m (np.ndarray): The matrix to standardize. It will be changed in place.
    """
    _, columns = m.shape
    for column in range(columns):

        # Center in the mean value
        mean = np.mean(m[:, column])
        m[:, column] -= mean
        # Check that centering is correctly done
        # (we trust the NumPy code - we don't trust our own code)
        assert np.allclose(np.mean(m[:, column]), 0)

        # Adjust standard deviation to 1
        std = np.std(m[:, column])
        m[:, column] /= std
        # Check that centering is correctly done
        # (we trust the NumPy code - we don't trust our own code)
        assert np.allclose(np.std(m[:, column]), 1)


def center(m: np.ndarray) -> None:
    """Center a matrix in place.

    Args:
        y (np.ndarray): The matrix to be centerd. It will be changed in place.
    """
    _, columns = m.shape
    for column in range(columns):
        mean = np.mean(m[:, column])
        m[:, column] -= mean


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


def mse(y: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate the MSE of the predictions.

    Args:
        y (np.ndarray): The actual values (the targets/output).
        predictions (np.ndarray): The predictions from the model.

    Returns:
        float: The MSE of the predictions.
    """
    mse = np.mean((y - predictions) ** 2)
    return mse


def describe(title: str, data: np.ndarray) -> None:
    """Describe a NumPy matrix.

    Args:
        title (str): The title of the description.
        data (np.ndarray): The matrix to describe
    """
    print(f'\n\n{title}')
    print(f'Type: {type(data)}')
    print(f'Shape: {data.shape}')
    for column in data.T:
        print(f'  Column type: {type(column[0])}, sampe value: {column[0]}')
    print('Data (first few rows):')
    print(data[:3])

    # Use Pandas to describe - may waste some memory and time, but shows good amount of info
    print('\nStatistics')
    print(pd.DataFrame(data).describe())


if __name__ == "__main__":
    check_python_version()
