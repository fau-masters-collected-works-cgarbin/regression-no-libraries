"""Test the ridge regression code.

Create simple datasets to test the ridge regression code.
"""
import copy

import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline

import utils
import ridge


def test_categorical():
    """Create a simple dataset that simulates a categorial features with two values..

    The dataset is a simple linear regression problem. The fit function should find coefficients
    that model the data almost perfectly. The prediction errors should be very small.

    If this function fails, there is a likely a problem in the fit function.
    """
    # Create a dataset with a categorical feature
    test_file_name = 'test_dataset_categorical.csv'
    with open(test_file_name, 'w', encoding='utf-8') as test_file:
        test_file.write('a,b,a+b\n')
        for i in range(1, 1001, 1):
            x1 = i
            x2 = 0 if i % 5 else 1  # simulates a categorical feature
            y = (i * 2) if i < 100 else (i * 3)
            test_file.write(f'{x1},{x2},{y}\n')

    # Read the dataset and scale/center to prepare to fit
    x, y = utils.read_dataset(test_file_name)
    utils.scale(x)
    utils.center(y)

    # Fit with a small learning rate, and only a few iterations because this dataset is simple
    # We don't really need regularization for this case, but we use a small value to not hide a
    # possible error in the code if we simply set it to zero
    coefficients = ridge.fit(x, y, lr=0.0001, lmbda=0.1, iterations=1000)

    # Predict the original dataset again
    predictions = utils.predict(x, coefficients)

    # Check that the error is within a reasonable range
    mse = utils.mse(y, predictions)
    assert mse < 250, f'MSE is too high: {mse}'

    # Show some predictions and the error
    print('Our code')
    print(f'MSE: {mse}')
    print(f'Original input:\n{y[:3]}')
    print(f'Predicted values:\n{predictions[:3]}')

    # Now use scikit-learn on the same dataset
    x_sk, y_sk = utils.read_dataset('./test_simple_dataset.csv')
    model = pipeline.make_pipeline(sk.preprocessing.StandardScaler(), linear_model.Ridge(alpha=0.1))
    model.fit(x_sk, y_sk)
    predictions_sk = model.predict(x_sk)
    mse_sk = sk.metrics.mean_squared_error(y_sk, predictions_sk)

    print('\nscikit-learn')
    print(f'MSE: {mse_sk}')
    print(f'Original input:\n{y_sk[:3]}')
    print(f'Predicted values:\n{predictions_sk[:3]}')

    # Check that our result is close to the scikit-learn result
    assert abs(mse - mse_sk) < 2, 'MSEs are too far appart'


def credit():
    x_orig, y_orig = utils.read_dataset('./Credit_N400_p9.csv')

    # Make a copy to preserve the original data
    x = copy.deepcopy(x_orig)
    y = copy.deepcopy(y_orig)

    # Encode the categorical values
    utils.encode_binary_cateogry(x, column=6, one_value='Female')  # gender
    utils.encode_binary_cateogry(x, column=7, one_value='Yes')  # student
    utils.encode_binary_cateogry(x, column=8, one_value='Yes')  # married

    utils.scale(x)
    utils.center(y)

    coefficients = ridge.fit(x, y, lr=0.00005, lmbda=10000, iterations=10000)

    predictions = utils.predict(x, coefficients)
    mse = utils.mse(y, predictions)

    print(y[:3])
    print(predictions[:3])
    print(mse)


def test_scikit():
    x_sk, y_sk = utils.read_dataset('./test_simple_dataset.csv')
    model = pipeline.make_pipeline(sk.preprocessing.StandardScaler(), linear_model.Ridge(alpha=0.1))
    model.fit(x_sk, y_sk)
    predictions_sk = model.predict(x_sk)
    mse_sk = sk.metrics.mean_squared_error(y_sk, predictions_sk)

    print('\n\nStandaloone sk test-------')
    print(y_sk[:3])
    print(predictions_sk[:3])
    print(mse_sk)


if __name__ == "__main__":
    utils.check_python_version()

    print('Testing our code')
    test_categorical()
    test_scikit()

    # credit()
