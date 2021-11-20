"""Test the ridge regression code.

Test the ridge regression code with a very simple dataset to verify its basic functionality,
then test again with a more complex dataset.

Each test is also compared with the scikit-learn regression. We expect to be very close to the
results that scikit-learn achieves.
"""
import copy
import numpy as np
import pathlib

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import metrics

# A hacky way to get around importing modules from the parent directory
# Without this, we get "ImportError: attempted relative import with no known parent package"
import sys
sys.path.append('../')  # When runnign from the test directory
sys.path.append('./')  # When running from the main directory from the debugger

# Must come after the path change above (noqa prevents VS Code moving it to the top)
import utils  # noqa
import ridge  # noqa

# Set false to not print results, just execute the tests
_verbose = True


def _test_ridge(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int,
                max_mse: float, max_mse_diff: float) -> None:
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
    # pylint: disable=too-many-arguments, too-many-locals

    # Copy because we modify it
    x_ours = copy.deepcopy(x)
    y_ours = copy.deepcopy(y)

    utils.scale(x_ours)
    utils.center(y_ours)

    coefficients = ridge.fit(x_ours, y_ours, lr=lr, lmbda=lmbda, iterations=iterations)
    predictions = ridge.predict(x_ours, coefficients)
    mse = utils.mse(y_ours, predictions)

    if _verbose:
        print('\nRidge - our code')
        print(f'  MSE: {mse}')
        print(f'  Original input (standardized values):\n{y_ours[:3]}')
        print(f'  Predicted values (standardized values):\n{predictions[:3]}')

    # Check that the error is within a reasonable range
    mse = utils.mse(y_ours, predictions)
    assert mse <= max_mse, f'MSE {mse} is too high, should be <= {max_mse}'

    # Now use scikit-learn on the same dataset
    x_sk = copy.deepcopy(x)
    y_sk = copy.deepcopy(y)
    model = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=lmbda))
    model.fit(x_sk, y_sk)
    predictions_sk = model.predict(x_sk)
    mse_sk = metrics.mean_squared_error(y_sk, predictions_sk)

    if _verbose:
        print('scikit-learn')
        print(f'  MSE: {mse_sk}')
        print(f'  Original input:\n{y_sk[:3]}')
        print(f'  Predicted values:\n{predictions_sk[:3]}')

    # Check that our result is close to the scikit-learn result
    mse_diff = abs(mse - mse_sk)
    assert mse_diff < max_mse_diff, f'MSEs too far appart ({mse_diff}), should be <= {max_mse_diff}'


def test_simple_prediction() -> None:
    """Test the prediction code with a very simple model - it must perform well on it."""
    if _verbose:
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

    x, y, _, _, _ = utils.read_dataset(test_file_name)

    # Because this dataset is simple, it is expected to peform well

    # We should be able to train it with just a few iterations
    # MSE margins are low because we expect to perform well with these hyperparameters and very
    # close to what scikit-learn would do
    # We don't really need regularization for this case, but we use a small value to not hide a
    # possible error in the code if we simply set it to zero
    _test_ridge(x, y, lr=0.0001, lmbda=0.001, iterations=100, max_mse=0.01, max_mse_diff=0.01)


def test_categorical_prediction() -> None:
    """Test the prediction code with a dataset that simulates a categorical variable."""
    if _verbose:
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
    x, y, _, _, _ = utils.read_dataset(test_file_name)
    _test_ridge(x, y, lr=0.0001, lmbda=0.1, iterations=1000, max_mse=250, max_mse_diff=0.1)


def test_credit_prediction(data_dir: str):
    """Test the prediction code with the credit dataset."""
    if _verbose:
        print('\n\nCredit dataset')

    file = pathlib.Path(data_dir) / 'Credit_N400_p9.csv'
    x, y, _, _, _ = utils.read_dataset(file)

    # Encode the categorical values
    utils.encode_binary_cateogry(x, column=6, one_value='Female')  # gender
    utils.encode_binary_cateogry(x, column=7, one_value='Yes')  # student
    utils.encode_binary_cateogry(x, column=8, one_value='Yes')  # married

    _test_ridge(x, y, lr=0.00001, lmbda=1_000, iterations=1_000, max_mse=100_000, max_mse_diff=0.1)


def test_all(verbose: bool = True, data_dir: str = '../data') -> None:
    """Run all the tests."""
    global _verbose
    _verbose = verbose

    utils.check_python_version()

    # Adjust path when running from the main directory
    if not pathlib.Path(data_dir).exists():
        data_dir = pathlib.Path('.') / 'data'

    test_simple_prediction()
    test_categorical_prediction()
    test_credit_prediction(data_dir)

    if _verbose:
        print('\nRidge: all tests passed')


if __name__ == "__main__":
    test_all()
