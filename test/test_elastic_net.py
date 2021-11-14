"""Test the elastic net regression code.

Test the elsatic net regression code with a very simple dataset to verify its bascically functionality,
then test again with a more complex dataset.

Tests are also compared with our ridge regression code. The elastic net values must be close to the
ridge values and, when the elastic net penalty (alpha) is one, elastic net behaves like ridge regression.
In that case, the values computed with the elastic net code should be close to the values computed with
the ridge code.

IMPORTANT: assumes that the ridge regression code was tested first.
"""
import copy
import numpy as np
import os

# A hacky way to get around importing modules from the parent directory
# Without this, we get "ImportError: attempted relative import with no known parent package"
import sys
sys.path.append('../')

# Must come after the path change above (noqa prevents VS Code moving it to the top)
import utils  # noqa
import ridge  # noqa
import elastic_net  # noqa

# Set false to not print results, just execute the tests
_verbose = True


def _test_elastic_net(x: np.ndarray, y: np.ndarray, lmbda: float, alpha: float, lr: float, iterations: int,
                      max_mse: float, max_mse_diff: float) -> None:
    """Test the elastic net regression code on a dataset, comparing with our ridge code.

    IMPORTANT: assumes that our ridge code was tested first.

    Args:
        x (np.ndarray): The input values, with categorical values (if present) encoded as binary values.
        y (np.ndarray): The target values.
        data_file (str): The name of the dataset file.
        lmbda (float): The regularization parameter (lambda).
        alpha (float): The elastic net regularization parameter.
        lr (float): The learning rate (alpha) to use for the ridge code.
        iterations (int): Number of iterations to run the gradient descent.
        max_mse (float): The maximum MSE allowed before the test fails.
        max_mse_diff (float): The maximum MSE difference allowed between our model and the model
            from scikit-learn before the test fails.
    """
    # pylint: disable=too-many-arguments, too-many-locals

    # Copy because we modify it
    x_en = copy.deepcopy(x)
    y_en = copy.deepcopy(y)

    utils.scale(x_en)
    utils.center(y_en)

    coefficients = elastic_net.fit(x_en, y_en, lmbda=lmbda, alpha=alpha, iterations=iterations)
    predictions = elastic_net.predict(x_en, coefficients)
    mse = utils.mse(y_en, predictions)

    if _verbose:
        print(f'\nElastic net - our code with alpha {alpha}')
        print(f'  MSE: {mse}')
        print(f'  Original input (standardized values):\n{y_en[:3]}')
        print(f'  Predicted values (standardized values):\n{predictions[:3]}')

    # Check that the error is within a reasonable range
    mse = utils.mse(y_en, predictions)
    assert mse <= max_mse, f'MSE {mse} is too high, should be <= {max_mse}'

    # Now use our ridge code
    x_ridge = copy.deepcopy(x)
    y_ridge = copy.deepcopy(y)
    utils.scale(x_ridge)
    utils.center(y_ridge)
    coefficients_ridge = ridge.fit(x_ridge, y_ridge, lmbda=lmbda, lr=lr, iterations=iterations)
    predictions_ridge = ridge.predict(x_ridge, coefficients_ridge)
    mse_ridge = utils.mse(y_ridge, predictions_ridge)

    if _verbose:
        print('Ridge - our code')
        print(f'  MSE: {mse_ridge}')
        print(f'  Original input:\n{y_ridge[:3]}')
        print(f'  Predicted values:\n{predictions_ridge[:3]}')

    # Check that our result is close to the ridge code
    mse_diff = abs(mse - mse_ridge)
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

    x, y, _ = utils.read_dataset(test_file_name)

    # Because this dataset is simple, it is expected to peform well
    # Test elastic net with a mix of ridge and lasso regularization
    _test_elastic_net(x, y, lmbda=0.001, alpha=0.0, lr=0.0001, iterations=100, max_mse=0.01, max_mse_diff=0.01)
    _test_elastic_net(x, y, lmbda=0.001, alpha=0.5, lr=0.0001, iterations=100, max_mse=0.01, max_mse_diff=0.01)
    _test_elastic_net(x, y, lmbda=0.001, alpha=1.0, lr=0.0001, iterations=100, max_mse=0.01, max_mse_diff=0.01)


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
    # Test elastic net with a mix of ridge and lasso regularization
    x, y, _ = utils.read_dataset(test_file_name)
    _test_elastic_net(x, y, lmbda=0.0001, alpha=0.0, lr=0.0001, iterations=1000, max_mse=250, max_mse_diff=0.01)
    _test_elastic_net(x, y, lmbda=0.0001, alpha=0.5, lr=0.0001, iterations=1000, max_mse=250, max_mse_diff=0.01)
    _test_elastic_net(x, y, lmbda=0.0001, alpha=1.0, lr=0.0001, iterations=1000, max_mse=250, max_mse_diff=0.01)


def test_credit_prediction(data_dir: str):
    """Test the prediction code with the credit dataset."""
    if _verbose:
        print('\n\nCredit dataset')

    file = os.path.join(data_dir, 'Credit_N400_p9.csv')
    x, y, _ = utils.read_dataset(file)

    # Encode the categorical values
    utils.encode_binary_cateogry(x, column=6, one_value='Female')  # gender
    utils.encode_binary_cateogry(x, column=7, one_value='Yes')  # student
    utils.encode_binary_cateogry(x, column=8, one_value='Yes')  # married

    # Test elastic net with a mix of ridge and lasso regularization
    # Because this dataset is more complex, we accept a higher MSE difference when we use alpha values push the
    # elastic net towards a lasso regression (especially when alpha=0)
    # On the other hand, when using elastic net as a ridge regression (alpha=1), we expect to perform very close
    # to the ridge regression code
    _test_elastic_net(x, y, lmbda=1_000, alpha=0.0, lr=0.00001, iterations=1_000, max_mse=100_000, max_mse_diff=90_000)
    _test_elastic_net(x, y, lmbda=1_000, alpha=0.5, lr=0.00001, iterations=1_000, max_mse=100_000, max_mse_diff=30_000)
    _test_elastic_net(x, y, lmbda=1_000, alpha=1.0, lr=0.00001, iterations=1_000, max_mse=100_000, max_mse_diff=0.01)


def test_all(verbose: bool = True, data_dir: str = '../data') -> None:
    """Run all the tests."""
    global _verbose
    _verbose = verbose

    utils.check_python_version()

    test_simple_prediction()
    test_categorical_prediction()
    test_credit_prediction(data_dir)

    if _verbose:
        print('\nElastic net: all tests passed')


if __name__ == "__main__":
    test_all()
