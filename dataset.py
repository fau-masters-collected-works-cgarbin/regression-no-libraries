
import numpy as np
import pandas as pd
import sys
import copy
from typing import Tuple


# TODO: add logging for verbose mode

def check_python_version():
    if sys.version_info < (3, 6):
        print('Requires Python 3.6 or higher')
        sys.exit(1)


def read_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset from a file and return NumPy arrays.

    It reads first into a Pandas DataFrame, then converts to NumPy arrays because the NumPy function
    to read from a file, genfromtxt, is harder to use when strings are present. When specifying the
    `dtype` parameter, it returns a 1D array with structured data for each row, istead of the N x p
    matrix we want.

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


def hot_encode_binary(x: np.ndarray, column: int, one_value: str) -> np.ndarray:
    """Hot encode a binary (two values) column in a matrix in place.

    The original column is replaced with the hot encoded version.

    Args:
        x (np.ndarray): The feature matrix.
        column (int): The column to hot encode.
        one_value (str): The string to use as "1" for the hot encoded column.
    """
    hot_encoded = x[:, column] == one_value
    x[:, column] = hot_encoded.astype(int)


def standardize_dataset(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize the dataset by removing the mean and scaling to unit variance.

    Note that if the input or output matrix has only one column, it is adjusted to the mean, but not
    scaled. We only need to scale a matrix with multiple columns. This usually applies to the ouptut
    matrix. It's more likely that it has only one column.

    Args:
        x (np.ndarray): The input data.
        y (np.ndarray): The output data.

    Returns:
        np.ndarray, np.ndarray: The standardized input and output matrixes, with their oririginal
        dimensions.
    """
    # TODO: assert the values and dimensions
    pass


def describe_data(title: str, data: np.ndarray) -> None:
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


def test():
    x_orig, y_orig = read_dataset('./Credit_N400_p9.csv')

    # Make a copy to preserve the original data
    x = copy.deepcopy(x_orig)
    y = copy.deepcopy(y_orig)

    # Hot encode the categorical values
    hot_encode_binary(x, column=6, one_value='Female')  # gender
    hot_encode_binary(x, column=7, one_value='Yes')  # student
    hot_encode_binary(x, column=8, one_value='Yes')  # married

    standardize_matrix(x)
    standardize_matrix(y)

    verify_standardization(x)
    verify_standardization(y)

    describe_data('X (input)', x)
    describe_data('Y (output)', y)


if __name__ == "__main__":
    check_python_version()
    test()
