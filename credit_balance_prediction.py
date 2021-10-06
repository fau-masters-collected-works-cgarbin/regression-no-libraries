"""Predict credit balance using a ridge regression model.

Reference: `An Introduction to Statistical Learrning`_, James et al., second edition, section
6.2 Shrinkage Methods.

.. _An Introduction to Statistical Learrning:
   https://web.stanford.edu/~hastie/ISLRv2_website.pdf
"""

import copy

import test
import utils
import ridge

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import metrics

# Ensure the code is working before using it
test.verbose = False
test.test_all()

x_orig, y_orig = utils.read_dataset('Credit_N400_p9.csv')

# Encode the categorical values
utils.encode_binary_cateogry(x_orig, column=6, one_value='Female')  # gender
utils.encode_binary_cateogry(x_orig, column=7, one_value='Yes')  # student
utils.encode_binary_cateogry(x_orig, column=8, one_value='Yes')  # married


def experiment(lmbda: float) -> None:
    # Make a copy to preserve the original values
    x = copy.deepcopy(x_orig)
    y = copy.deepcopy(y_orig)

    utils.scale(x)
    utils.center(y)

    model = ridge.fit(x, y, lr=0.00001, lmbda=lmbda, iterations=1000)
    predictions = ridge.predict(x, model)
    mse = utils.mse(y, predictions)
    coef_str = ' '.join('{:8.2f}'.format(c) for c in model.flatten())

    # Now use scikit-learn on the same dataset
    x_sk = copy.deepcopy(x_orig)
    y_sk = copy.deepcopy(y_orig)
    model_sk = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=lmbda))
    model_sk.fit(x_sk, y_sk)
    predictions_sk = model_sk.predict(x_sk)
    mse_sk = metrics.mean_squared_error(y_sk, predictions_sk)
    coef_sk = model_sk.named_steps['ridge'].coef_
    coef_str_sk = ' '.join('{:8.2f}'.format(c) for c in coef_sk.flatten())

    print(f'Our code:     MSE: {mse} coefficients: {coef_str} lambda: {lmbda}')
    print(f'scikit-learn: MSE: {mse_sk} coefficients: {coef_str_sk} lambda: {lmbda}')


experiment(0.01)
experiment(0.1)
experiment(1)
experiment(10)
experiment(100)
experiment(1_000)
experiment(10_000)
