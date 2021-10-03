
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


def hot_encode_binary(x: np.ndarray, column: int, one_value: str) -> np.ndarray:
    """Hot encode a binary (two values) column of a matrix in place.

    The original column is replaced with the hot encoded version.

    The hot-encoded value is stored as float (not integer), in preparation for later steps that need
    to perform math operations on it.

    Args:
        x (np.ndarray): The feature matrix.
        column (int): The column to hot encode.
        one_value (str): The string to use as "1.0" for the hot encoded column.
    """
    hot_encoded = x[:, column] == one_value
    x[:, column] = hot_encoded.astype(float)


def scale(m: np.ndarray) -> np.ndarray:
    """Standardize a matrix to have a mean of zero and a standard deviation of one.

    Standardization is done in place. The original matrix is modified.

    All columns must be float. If there are categorical columns in the dataset, they must be hot
    encoded to float (1.0 or 0.0) before this function is called.

    Note that if the matrix has only one column, the mean is adjusted to the mean, but the standard
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
        if columns > 1:
            std = np.std(m[:, column])
            m[:, column] /= std
            # Check that centering is correctly done
            # (we trust the NumPy code - we don't trust our own code)
            assert np.allclose(np.std(m[:, column]), 1)


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
    x_orig, y_orig = read_dataset('./test_dataset.csv')

    # Make a copy to preserve the original data
    x = copy.deepcopy(x_orig)
    y = copy.deepcopy(y_orig)

    scale(x)

    describe_data('X (input)', x)
    describe_data('Y (output)', y)


def credit():
    x_orig, y_orig = read_dataset('./Credit_N400_p9.csv')

    # Make a copy to preserve the original data
    x = copy.deepcopy(x_orig)
    y = copy.deepcopy(y_orig)

    # Hot encode the categorical values
    hot_encode_binary(x, column=6, one_value='Female')  # gender
    hot_encode_binary(x, column=7, one_value='Yes')  # student
    hot_encode_binary(x, column=8, one_value='Yes')  # married

    scale(x)

    describe_data('X (input)', x)
    describe_data('Y (output)', y)


if __name__ == "__main__":
    check_python_version()
    test()
