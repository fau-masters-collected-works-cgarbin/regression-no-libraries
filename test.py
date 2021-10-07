"""Test the ridge regression code.

Test the ridge regression code with a very simple dataset to verify its bascically functionality,
then test again with a more complex dataset.

Each test is also compared with the scikit-learn ridge regression. We expect to be very close to the
results that scikit-learn achieves.
"""
import copy
import numpy as np

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import metrics

import utils
import ridge

# Set false to not print results, just execute the tests
verbose = True


def _test_prediction(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int,
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

    # Copy before we modify it
    x_orig = copy.deepcopy(x)
    y_orig = copy.deepcopy(y)

    utils.scale(x)
    utils.center(y)

    coefficients = ridge.fit(x, y, lr=lr, lmbda=lmbda, iterations=iterations)
    predictions = ridge.predict(x, coefficients)
    mse = utils.mse(y, predictions)

    if verbose:
        print('Our code')
        print(f'MSE: {mse}')
        print(f'Original input (standardized values):\n{y[:3]}')
        print(f'Predicted values (standardized values):\n{predictions[:3]}')

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

    if verbose:
        print('\nscikit-learn')
        print(f'MSE: {mse_sk}')
        print(f'Original input:\n{y_sk[:3]}')
        print(f'Predicted values:\n{predictions_sk[:3]}')

    # Check that our result is close to the scikit-learn result
    mse_diff = abs(mse - mse_sk)
    assert mse_diff < max_mse_diff, f'MSEs too far appart ({mse_diff}), should be <= {max_mse_diff}'


def test_simple_prediction() -> None:
    """Test the prediction code with a very simple model - it must perform well on it."""
    if verbose:
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
    _test_prediction(x, y, lr=0.0001, lmbda=0.001, iterations=100, max_mse=0.01, max_mse_diff=0.01)


def test_categorical_prediction() -> None:
    """Test the prediction code with a dataset that simulates a categorical variable."""
    if verbose:
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
    _test_prediction(x, y, lr=0.0001, lmbda=0.1, iterations=1000, max_mse=250, max_mse_diff=0.1)


def test_credit_prediction():
    """Test the prediction code with the credit dataset."""
    if verbose:
        print('\n\nCredit dataset')

    file_name = './Credit_N400_p9.csv'
    x, y = utils.read_dataset(file_name)

    # Encode the categorical values
    utils.encode_binary_cateogry(x, column=6, one_value='Female')  # gender
    utils.encode_binary_cateogry(x, column=7, one_value='Yes')  # student
    utils.encode_binary_cateogry(x, column=8, one_value='Yes')  # married

    _test_prediction(x, y, lr=0.00001, lmbda=1_000, iterations=10_000, max_mse=100_000,
                     max_mse_diff=0.1)


def test_split_fold() -> None:
    """Test code that splits matrices into folds."""
    # pylint: disable=too-many-locals

    # Create a dataset with a number of folds that is not a multiple of the folds
    # All elements have simple values so we can visually inspect the folds
    test_file_name = 'test_split_fold.csv'
    with open(test_file_name, 'w', encoding='utf-8') as test_file:
        test_file.write('a,b,a+b\n')
        for i in range(1, 17, 1):
            x1 = i
            x2 = i
            y = i
            test_file.write(f'{x1},{x2},{y}\n')

    x, y = utils.read_dataset(test_file_name)

    folds = 3
    val_size = len(x) // folds
    train_size = len(x) - val_size

    x_train1, x_val1, y_train1, y_val1 = utils.split_fold(x, y, folds, 1)
    x_train2, x_val2, y_train2, y_val2 = utils.split_fold(x, y, folds, 2)
    x_train3, x_val3, y_train3, y_val3 = utils.split_fold(x, y, folds, 3)

    # The folds must have the same length
    assert [len(x_train1), len(x_train2), len(x_train3)] == [train_size, train_size, train_size]
    assert [len(y_train1), len(y_train2), len(y_train3)] == [train_size, train_size, train_size]
    assert [len(x_val1), len(x_val2), len(x_val3)] == [val_size, val_size, val_size]
    assert [len(y_val1), len(y_val2), len(y_val3)] == [val_size, val_size, val_size]

    # Matching x and y folds must have the same length
    assert len(x_train1) == len(y_train1) and len(x_val1) == len(y_val1)
    assert len(x_train2) == len(y_train2) and len(x_val2) == len(y_val2)
    assert len(x_train3) == len(y_train3) and len(x_val3) == len(y_val3)

    # Folds must be different from each other
    # Multiple array comparison from https://stackoverflow.com/a/37777691
    assert not np.logical_and((x_train1 == x_train2).all(), (x_train2 == x_train3).all())
    assert not np.logical_and((y_train1 == y_train2).all(), (y_train2 == y_train3).all())
    assert not np.logical_and((x_val1 == x_val2).all(), (x_val2 == x_val3).all())
    assert not np.logical_and((y_val1 == y_val2).all(), (y_val2 == y_val3).all())

    # A change in a split must not change the original data (we asekd for a copy)
    # Written in this form in case we shuffle the data before extracting folds (can't compare on
    # particular element in that case)
    x_val1[0] = x.max() + 1_000
    y_val1[0] = y.max() + 1_000
    assert x_val1.max() > x.max()
    assert y_val1.max() > y.max()


def test_scale_center() -> None:
    """Test the cetnering and scaling code."""
    # column means = [[2], [6]], std = [[1], [2]]
    test_array = np.array([[1., 4.],
                           [3., 8.]])

    # Test centering
    center_test = copy.deepcopy(test_array)
    utils.center(center_test)
    assert np.array_equal(center_test, np.array([[-1., -2.], [1., 2.]]))

    # Test centering with a validation set
    # The validation set must be centered with the mean of the main set
    center_main = np.array([[1.], [3.]])
    center_val = np.array([[11.], [44.]])
    utils.center_val(center_main, center_val)
    assert np.array_equal(center_val, np.array([[9.], [42.]]))

    # Test scaling
    scale_test = copy.deepcopy(test_array)
    utils.scale(scale_test)
    assert np.array_equal(scale_test, np.array([[-1., -1.], [1., 1.]]))

    # Test scaling with a validation set
    # The validation set must be scaled with the mean and std of the main set
    scale_main = np.array([[4.], [8.]])   # mean: 6, std: 2
    scale_val = np.array([[12.], [36.]])
    utils.scale_val(scale_main, scale_val)
    assert np.array_equal(scale_val, np.array([[3.], [15.]]))  # ( .. - mean) / std


def test_all() -> None:
    """Run all the tests."""
    utils.check_python_version()
    test_scale_center()
    test_split_fold()
    test_simple_prediction()
    test_categorical_prediction()
    test_credit_prediction()

    if verbose:
        print('\nAll tests passed')


if __name__ == "__main__":
    test_all()
