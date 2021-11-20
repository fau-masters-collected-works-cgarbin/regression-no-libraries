"""Test the utility functions.

Test the utitlity functions used by all regression algorithms.

Each test is also compared with the scikit-learn ridge regression. We expect to be very close to the
results that scikit-learn achieves.
"""
import numpy as np
import copy

# A hacky way to get around importing modules from the parent directory
# Without this, we get "ImportError: attempted relative import with no known parent package"
import sys
sys.path.append('../')  # When runnign from the test directory
sys.path.append('./')  # When running from the main directory from the debugger

# Must come after the path change above (noqa prevents VS Code moving it to the top)
import utils  # noqa


# Set false to not print results, just execute the tests
_verbose = True


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

    x, y, _, _, _ = utils.read_dataset(test_file_name)

    folds = 3
    val_size = len(x) // folds
    train_size = len(x) - val_size

    x_train1, x_val1, y_train1, y_val1 = utils.split_fold(x, y, folds, 1)
    x_train2, x_val2, y_train2, y_val2 = utils.split_fold(x, y, folds, 2)
    x_train3, x_val3, y_train3, y_val3 = utils.split_fold(x, y, folds, 3)

    # The folds must have the same length
    assert [len(x_train1), len(x_train2), len(x_train3)] == [
        train_size, train_size, train_size]
    assert [len(y_train1), len(y_train2), len(y_train3)] == [
        train_size, train_size, train_size]
    assert [len(x_val1), len(x_val2), len(x_val3)] == [
        val_size, val_size, val_size]
    assert [len(y_val1), len(y_val2), len(y_val3)] == [
        val_size, val_size, val_size]

    # Matching x and y folds must have the same length
    assert len(x_train1) == len(y_train1) and len(x_val1) == len(y_val1)
    assert len(x_train2) == len(y_train2) and len(x_val2) == len(y_val2)
    assert len(x_train3) == len(y_train3) and len(x_val3) == len(y_val3)

    # Folds must be different from each other
    # Multiple array comparison from https://stackoverflow.com/a/37777691
    assert not np.logical_and(
        (x_train1 == x_train2).all(), (x_train2 == x_train3).all())
    assert not np.logical_and(
        (y_train1 == y_train2).all(), (y_train2 == y_train3).all())
    assert not np.logical_and((x_val1 == x_val2).all(),
                              (x_val2 == x_val3).all())
    assert not np.logical_and((y_val1 == y_val2).all(),
                              (y_val2 == y_val3).all())

    # A change in a split must not change the original data (we asekd for a copy)
    # Written in this form in case we shuffle the data before extracting folds (can't compare on
    # particular element in that case)
    x_val1[0] = x.max() + 1_000
    y_val1[0] = y.max() + 1_000
    assert x_val1.max() > x.max()
    assert y_val1.max() > y.max()


def test_scale_center() -> None:
    """Test the centering and scaling code."""
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
    assert np.array_equal(scale_val, np.array(
        [[3.], [15.]]))  # ( .. - mean) / std


def test_all(verbose: bool = True) -> None:
    """Run all the tests."""
    global _verbose
    _verbose = verbose

    utils.check_python_version()
    test_scale_center()
    test_split_fold()

    if _verbose:
        print('\nUtils: all tests passed')


if __name__ == "__main__":
    test_all()
