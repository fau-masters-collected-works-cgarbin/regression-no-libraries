
import numpy as np
import pandas as pd
import sys


def check_python_version():
    if sys.version_info < (3, 6):
        print('Requires Python 3.6 or higher')
        sys.exit(1)


def read_dataset(filename: str):
    """Read the dataset from a file and return NumPy arrays.

    It reads first into a Pandas DataFrame, then converts to NumPy arrays because the NumPy function
    to read from a file, genfromtxt, is harder to use when strings are present. When specifying the
    `dtype` parameter, it returns a 1D array with structured data for each row, istead of the N x p
    matrix we want.

    Args:
        filename (str): The name of the file to read.

    Returns:
        x, y: The N x p feature matrix and N x 1 target vector.
    """
    dataset = pd.read_csv(filename)
    print(dataset.shape)
    print(dataset)

    # Split into input and output (asumming the last column is the output)
    # And convert to NumPy arrays
    x = dataset.iloc[:, :-1].to_numpy()
    y = dataset.iloc[:, -1:].to_numpy()
    return x, y


def describe_data(title: str, data):
    print(f'\n\n{title}')
    print(f'Type: {type(data)}')
    print(f'Shape: {data.shape}')
    print('Data (first few rows):')
    print(data[:3])


if __name__ == "__main__":
    check_python_version()

    x, y = read_dataset('./Credit_N400_p9.csv')
    describe_data('X (input)', x)
    describe_data('Y (output)', y)
