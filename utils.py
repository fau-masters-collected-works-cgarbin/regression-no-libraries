"""Utility functions to work with datasets and perform statistical computations.

Some functions, e.g. predicting values, calculating MSE, etc. are simple and could have been done
in place. The code has functions even for those cases to make the intent of each piece of code
explicit. The goal is to be educational, not concise.
"""
import sys
import copy
from typing import Tuple
from typing import List
import numpy as np
import pandas as pd


def check_python_version():
    """Check that we have the correct Python version for the code."""
    if sys.version_info < (3, 6):
        raise RuntimeError("Python 3.6 of higher is required to run this code.")


def read_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read a dataset from a CSV file and split into features and output.

    Integer columns are converted to float, in preparation for code that needs to perform math
    operations on them.

    Categorical (string) columns are untouched. The caller code is responsible for encoding them
    as needed.

    Args:
        filename (str): The name of the file to read.

    Returns:
        np.ndarray, np.ndarray: The N x p feature matrix and N x 1 target vector.
        List(str): The name of the features, read from the first line of the file.

    """
    # Read first into a Pandas DataFrame, then convert to NumPy arrays because the NumPy function
    # to read from a file, `genfromtxt`, is harder to use when strings are present. When specifying
    # the `dtype` parameter, it returns a 1D array with structured data for each row, instead of the
    # N x p matrix we want.
    dataset = pd.read_csv(filename)

    # Ensure that numeric columns are floats (not integers) to help with the standardization step
    int_cols = dataset.columns[dataset.dtypes.eq('int')]
    dataset[int_cols] = dataset[int_cols].astype(float)

    # Split the features (input) from the output, assumming the last column is the output
    features = dataset.iloc[:, :-1]
    output = dataset.iloc[:, -1:]

    # Convert to NumPy arrays in preparation to manipulate it
    x = features.to_numpy()
    y = output.to_numpy()

    return x, y, list(features.columns)


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


def scale(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize a matrix: center (mean of zero) and standard deviation of one.

    Notes:
        * Scaling is done in place. The original matrix is modified.
        * All columns must be floats.
        * If there are categorical columns in the dataset, they must be encoded to float
          (0,0, 1,0, etc.) before this function is called.

    Args:
        m (np.ndarray): The matrix to scale. It will be changed in place.

    Returns:
        np.ndarray, np.ndarray: The mean and standard deviation of each column in the matrix.
    """
    _, columns = m.shape

    means = np.zeros(columns)
    stds = np.zeros(columns)

    for column in range(columns):

        # Center in the mean value
        mean = np.mean(m[:, column])
        m[:, column] -= mean

        # Adjust standard deviation to 1
        std = np.std(m[:, column])
        m[:, column] /= std

        means[column] = mean
        stds[column] = std

    return means, stds


def scale_val(m: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize a matrix and associated validation matrix.

    The validation matrix is standardized with the mean and std of the main matrix. This is usually
    used in cross-validation sets, when the validation set must be standardized with the values used
    for training, not woth its own mean and std.


    Notes:
        * Standardization is done in place. The original matrix is modified.
        * All columns must be floats.
        * If there are categorical columns in the dataset, they must be encoded to float
          (0,0, 1,0, etc.) before this function is called.

    Args:
        m (np.ndarray): The matrix to scale. It will be changed in place.
        val (np.ndarray): The associated validation matrix. It will be changed in place, using the
            mean and std of the main matrix, not its own values.

    Returns:
        np.ndarray, np.ndarray: The mean and standard deviation of each column in the main matrix.
    """
    _, columns = m.shape

    means = np.zeros(columns)
    stds = np.zeros(columns)

    for column in range(columns):

        # Center in the mean value
        mean = np.mean(m[:, column])
        m[:, column] -= mean

        # Adjust standard deviation to 1
        std = np.std(m[:, column])
        m[:, column] /= std

        means[column] = mean
        stds[column] = std

    val -= means
    val /= stds

    return means, stds


def center(m: np.ndarray) -> np.ndarray:
    """Center a matrix in place.

    Notes:
        * Centering is done in place. The original matrix is modified.
        * All columns must be floats.
        * If there are categorical columns in the dataset, they must be encoded to float
          (0,0, 1,0, etc.) before this function is called.

    Args:
        m (np.ndarray): The matrix to be centered. It will be changed in place.

    Returns:
        np.ndarray: The mean of each column in the matrix.
    """
    _, columns = m.shape

    means = np.zeros(columns)

    for column in range(columns):
        mean = np.mean(m[:, column])
        m[:, column] -= mean

        means[column] = mean

    return means


def center_val(m: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Center a matrix in place, then center the associated validation matrix in place.

    The validation matrix is centered with the mean of the main matrix. This is usually used in
    cross-validation sets, when the validation set must be centered around the values used for
    training, not aorund its own mean.

    Notes:
        * Centering is done in place. The original matrix is modified.
        * All columns must be floats.
        * If there are categorical columns in the dataset, they must be encoded to float
          (0,0, 1,0, etc.) before this function is called.
        * Both matrices must have the same number of columns

    Args:
        m (np.ndarray): The matrix to be centerd. It will be changed in place.
        val (np.ndarray): The associated validation matrix. It will be changed in place, using the
            mean of the main matrix, not its own mean.

    Returns:
        np.ndarray: The mean of each column in the matrix.
    """
    if m.shape[1] != val.shape[1]:
        raise ValueError('Matrices must have the same number of columns')

    _, columns = m.shape

    means = np.zeros(columns)

    for column in range(columns):
        mean = np.mean(m[:, column])
        m[:, column] -= mean

        means[column] = mean

    val -= means

    return means


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


def split_fold(x: np.ndarray, y: np.ndarray, num_folds: int, fold: int,
               make_copy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split input (features) and output (targets) matrices into folds and return the specified fold.

    Note:
        Remainder elements are used in the last train fold. For example, a matrix with 16 elements
        when split into 3 folds results in train folds of size 10, 10, and 11 elements. As a side
        effect, remainder elements are not used in a validation fold.

    Args:
        x (np.ndarray): The input (feature) matrix.
        y (np.ndarray): [The output (targets) matrix.
        num_folds (int): The number of folds to divided up the matrices (row-wise).
        fold (int): The fold to split out from the matrices. The first fold is 1 (not zero).
        make_copy (bool): Make a copy of the matrices before returnign them (defaults True). Most of
            the time we want a copy because the model fitting code may change the values (e.g.
            center and normalize them).

    Returns:
        np.ndarray, np.ndarray, np.ndarray, np.ndarray: The input (features) train and validation
            matrices, followed by the output (targets) train and validation matrices.
    """
    if len(x) != len(y):
        raise ValueError('Matrices must have the same length')

    fold_size = len(x) // num_folds
    start_index = (fold - 1) * fold_size
    end_index = fold * fold_size - 1

    if make_copy:
        x = copy.deepcopy(x)
        y = copy.deepcopy(y)

    x_train = np.concatenate((x[:start_index], x[end_index+1:]))
    y_train = np.concatenate((y[:start_index], y[end_index+1:]))
    x_val = x[start_index:end_index+1]
    y_val = y[start_index:end_index+1]

    # Check that we used all elements
    assert (len(x_train) + len(x_val)) == len(x)
    assert (len(y_train) + len(y_val)) == len(y)

    # Check that we preserved the shape (columns - rows will change)
    assert x_train.shape[1] == x_val.shape[1]
    assert y_train.shape[1] == y_val.shape[1]

    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    check_python_version()
