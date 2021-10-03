
import numpy as np
import pandas as pd
import sys
import copy
from typing import Tuple


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


def scale(m: np.ndarray):
    """Standardize a matrix: center (mean of zero) and standard deviation of one.

    Standardization is done in place. The original matrix is modified.

    All columns must be float. If there are categorical columns in the dataset, they must be hot
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
    m -= np.mean(m)


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


def fit_ridge(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int) -> np.ndarray:
    """Fit a linear regression model using ridge regression.

    Args:
        x (np.ndarray): The features (predictors). Must be hot-encoded and scaled as needed.
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

        # Compute the gradient, regularize (apply penalty), adjust with the learning rate and update
        beta = beta + 2 * lr * (x.T @ residuals - penalty)

    return beta


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict the output using the cofficients.

    Args:
        x (np.ndarray): The features (predictors). Must be hot-encoded and scaled as needed.
        coefficients (np.ndarray): The coefficients for the model.

    Returns:
        np.ndarray: The predictions.
    """
    predictions = x @ coefficients
    return predictions


def test():
    """Create a very simple dataset and test the ridge regression fitting code.

    The dataset is a simple linear regression problem. The fit function should find coefficients
    that model the data almost perfectly. The prediction errors should be very small.

    If this function fails, there is a likely a problem in the fit function.
    """
    # Create a very simple dataset
    test_file_name = 'test_simple_dataset.csv'
    with open(test_file_name, 'w', encoding='utf-8') as test_file:
        test_file.write('a,b,a+b\n')
        for i in range(1, 1001, 1):
            test_file.write(f'{i},{i*2},{i + i*2}\n')

    # Read the dataset and scale/center to prepare to fit
    x, y = read_dataset(test_file_name)
    scale(x)
    center(y)

    # Fit with a small learning rate, and only a few iterations because this dataset is very simple
    # (it's easy to overshoot the minimum)
    # We don't really need regularization for this case, but we use a small value to not hide a
    # possible error in the code if we simply set it to zero
    coefficients = fit_ridge(x, y, lr=0.0001, lmbda=0.1, iterations=100)

    # Predict the original dataset again...
    predictions = predict(x, coefficients)

    # ...and check that the model is almost perfect (on the training data)
    # Note that the output is centered, so we need to test for a positive and negative range of
    # residual values
    residuals = y - predictions
    assert ((residuals < 0.1) & (residuals > -0.1)).all()


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
    center(y)

    describe_data('X (input)', x)
    describe_data('Y (output)', y)


if __name__ == "__main__":
    check_python_version()
    test()
