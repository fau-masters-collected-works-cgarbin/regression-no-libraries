"""Test the ridge regression code.

Create simple datasets to test the ridge regression code.
"""
import copy
import numpy as np

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import metrics

import utils
import ridge


def _test(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int, max_mse: float,
          max_mse_diff: float) -> None:
    """Test the ridge regression code on a dataset, comparing with scikit-learn.

    Args:
        x (np.ndarray): The input values, with categorical values (if present) encoded as binary values.
        y (np.ndarray): The target values.
        data_file (str): The name of the dataset file.
        lr (float): The learning rate (alpha) to use for the gradient descent.
        lmbda (float): The regularization parameter (lambda) to use for the gradient descent.
        iterations (int): Number of iterations to run the gradient descent.
        max_mse (float): The maximum MSE allowed before the test fails.
        max_mse_diff (float): The maximum MSE difference allowed between our model and the model
            from scikit-learn before the test fails.
    """
    # Copy before we modify it
    x_orig = copy.deepcopy(x)
    y_orig = copy.deepcopy(y)

    utils.scale(x)
    utils.center(y)

    coefficients = ridge.fit(x, y, lr=lr, lmbda=lmbda, iterations=iterations)
    predictions = utils.predict(x, coefficients)
    mse = utils.mse(y, predictions)

    print('Our code')
    print(f'MSE: {mse}')
    print(f'Original input:\n{y[:3]}')
    print(f'Predicted values:\n{predictions[:3]}')

    # Check that the error is within a reasonable range
    mse = utils.mse(y, predictions)
    assert mse <= max_mse, f'MSE {mse} is too high, should be <= {max_mse}'

    # Now use scikit-learn on the same dataset
    x_sk = copy.deepcopy(x_orig)
    y_sk = copy.deepcopy(y_orig)
    model = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=lmbda))
    model.fit(x_sk, y_sk)
    predictions_sk = model.predict(x_sk)
    mse_sk = metrics.mean_squared_error(y_sk, predictions_sk)

    print('\nscikit-learn')
    print(f'MSE: {mse_sk}')
    print(f'Original input:\n{y_sk[:3]}')
    print(f'Predicted values:\n{predictions_sk[:3]}')

    # Check that our result is close to the scikit-learn result
    mse_diff = abs(mse - mse_sk)
    assert mse_diff < max_mse_diff, f'MSEs too far appart ({mse_diff}), should be <= {max_mse_diff}'


def test_simple() -> None:
    """Test the code with a very simple model - it must perform well on it."""
    print('\n\nSimple dataset')

    # Create a dataset with very simple features
    test_file_name = 'test_dataset_simple.csv'
    with open(test_file_name, 'w', encoding='utf-8') as test_file:
        test_file.write('a,b,a+b\n')
        for i in range(1, 1001, 1):
            x1 = i
            x2 = i * 2
            y = x1 + x2
            test_file.write(f'{x1},{x2},{y}\n')

    # Because this dataset is simple, it is expected to peform well
    # We should be able to train it with just a few iterations
    # MSE margins are low because we expect to perform well with these hyperparameters and very
    # close to what scikit-learn would do
    # We don't really need regularization for this case, but we use a small value to not hide a
    # possible error in the code if we simply set it to zero
    x, y = utils.read_dataset(test_file_name)
    _test(x, y, lr=0.0001, lmbda=0.001, iterations=100, max_mse=0.01, max_mse_diff=0.01)


def test_categorical() -> None:
    """Test the code with a dataset that simulates a categorical variable."""
    print('\n\nCategorical dataset')

    # Create a dataset with a categorical feature
    test_file_name = 'test_dataset_categorical.csv'
    with open(test_file_name, 'w', encoding='utf-8') as test_file:
        test_file.write('a,b,a+b\n')
        for i in range(1, 1001, 1):
            x1 = i
            x2 = 0 if i % 5 else 1  # simulates a categorical feature
            y = (i * 2) if i < 100 else (i * 3)
            test_file.write(f'{x1},{x2},{y}\n')

    # We don't really need regularization for this case, but we use a small value to not hide a
    # possible error in the code if we simply set it to zero
    x, y = utils.read_dataset(test_file_name)
    _test(x, y, lr=0.0001, lmbda=0.1, iterations=1000, max_mse=250, max_mse_diff=0.1)


def test_credit():
    """Test the code with the credit dataset."""
    print('\n\nCredit dataset')

    file_name = './Credit_N400_p9.csv'
    x, y = utils.read_dataset(file_name)

    # Encode the categorical values
    utils.encode_binary_cateogry(x, column=6, one_value='Female')  # gender
    utils.encode_binary_cateogry(x, column=7, one_value='Yes')  # student
    utils.encode_binary_cateogry(x, column=8, one_value='Yes')  # married

    _test(x, y, lr=0.00001, lmbda=1_000, iterations=10_000, max_mse=100_000, max_mse_diff=0.1)


def test_all() -> None:
    """Run all the tests."""
    test_simple()
    test_categorical()
    test_credit()


if __name__ == "__main__":
    utils.check_python_version()
    test_all()
