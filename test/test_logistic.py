"""Test the logistic regression code.

Test the logistic regression code with a very simple dataset to verify its basic functionality,
then test again with a more complex dataset.

Each test is also compared with the scikit-learn regression. We expect to be very close to the
results that scikit-learn achieves.
"""
import copy
import numpy as np
import os

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
import logistic  # noqa

# Set false to not print results, just execute the tests
_verbose = True


def _test_logistic(x: np.ndarray, y: np.ndarray, lr: float, lmbda: float, iterations: int) -> None:
    """Test the logistic regression code on a dataset, comparing with scikit-learn.

    Args:
        x (np.ndarray): The input values, with categorical values (if present) encoded as binary values.
        y (np.ndarray): The target values.
        data_file (str): The name of the dataset file.
        lr (float): The learning rate (alpha) to use for the gradient descent.
        lmbda (float): The regularization parameter (lambda) to use for the gradient descent.
        iterations (int): Number of iterations to run the gradient descent.
    """
    # pylint: disable=too-many-arguments, too-many-locals

    # Copy because we modify it
    x_ours = copy.deepcopy(x)
    y_ours = copy.deepcopy(y)

    utils.scale(x_ours)

    coefficients = logistic.fit(x_ours, y_ours, lr=lr, lmbda=lmbda, iterations=iterations)
    probabilities, classes = logistic.predict(x_ours, coefficients)

    # Calculated probabilities should be very close to the actual probabilities
    assert np.allclose(y_ours, probabilities, atol=0.1)
    # Classes must be the same
    assert (np.argmax(probabilities, axis=1).reshape(-1, 1) == classes).all()

    if _verbose:
        print('\nLogistic - our code')
        print(f'  Original classes:\n{y_ours[:3]}')
        print(f'  Caculated probabilities:\n{probabilities[:3]}')

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
    #assert ...


def test_simple_prediction() -> None:
    """Test the prediction code with a very simple model - it must perform well on it."""
    if _verbose:
        print('\n\nSimple dataset')

    # Create a dataset with very simple features
    # We need about one thousand samples of each class to train the classifier to preditct the correct class with
    # almost certainty (class probability very close to 100%)
    test_file_name = 'test_dataset_simple.csv'
    with open(test_file_name, 'w', encoding='utf-8') as test_file:
        test_file.write('Feature 1, Feature 2, Feature 3, Feature 4, Class\n')
        for _ in range(1, 1001, 1):
            test_file.write('1,0,0,0,Class 1\n')
        for _ in range(1, 1001, 1):
            test_file.write('0,1,0,0,Class 2\n')
        for _ in range(1, 1001, 1):
            test_file.write('0,0,1,0,Class 3\n')
        # Note that this is class 1 again, so that we have number of features > number of classes, to exercise the code
        for _ in range(1, 1001, 1):
            test_file.write('0,0,0,1,Class 1\n')

    x, y, _, _ = utils.read_dataset(test_file_name, hot_encode=True)

    # Because this dataset is simple, it is expected to peform well

    # We should be able to train it with just a few iterations and the classes should match exactly in this case
    _test_logistic(x, y, lr=0.0001, lmbda=0.001, iterations=100)


def test_ancestry(data_dir: str):
    """Test the prediction code with the ancestry dataset."""
    if _verbose:
        print('\n\nAncestry dataset')

    file = os.path.join(data_dir, 'TestData_N111_p10.csv')
    x, y, _, _ = utils.read_dataset(file, hot_encode=True)

    _test_logistic(x, y, lr=0.00001, lmbda=1_000, iterations=1_000)


def test_all(verbose: bool = True, data_dir: str = '../data') -> None:
    """Run all the tests."""
    global _verbose
    _verbose = verbose

    utils.check_python_version()

    test_simple_prediction()
    test_ancestry(data_dir)

    if _verbose:
        print('\nRegression: all tests passed')


if __name__ == "__main__":
    test_all()
