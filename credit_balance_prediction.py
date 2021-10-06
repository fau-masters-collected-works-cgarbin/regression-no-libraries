"""Predict credit balance using a ridge regression model.

Reference: `An Introduction to Statistical Learrning`_, James et al., second edition, section
6.2 Shrinkage Methods.

.. _An Introduction to Statistical Learrning:
   https://web.stanford.edu/~hastie/ISLRv2_website.pdf
"""
import copy
from typing import Tuple
import numpy as np

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


def _read_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Read the dataset and encode categorical values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The features (input) and targets (output) values.
    """
    x, y = utils.read_dataset('Credit_N400_p9.csv')

    # Encode the categorical values
    utils.encode_binary_cateogry(x, column=6, one_value='Female')  # gender
    utils.encode_binary_cateogry(x, column=7, one_value='Yes')  # student
    utils.encode_binary_cateogry(x, column=8, one_value='Yes')  # married

    return x, y


def experiment1(lmbda: float) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Run the experiments for deliverable 1, the effect of the tunning parameter (lambda) on the
    ridge coefficients (beta).

    It runs our code, then runs the scikit-learn ridge regression code to compare the results.

    Args:
        lmbda (float): The value of lambda to use for the regression.

    Returns:
        np.ndarray: The ridge coefficients (beta) calculated with our code.
        float: The MSE for the coefficients using our code.
        np.ndarray: The ridge coefficients calculated with scikit-learn.
        float: [description]: The MSE for the scikit-learn coefficients.
    """
    x_orig, y_orig = _read_dataset()

    # Make a copy to preserve the original values
    x = copy.deepcopy(x_orig)
    y = copy.deepcopy(y_orig)

    utils.scale(x)
    utils.center(y)

    model = ridge.fit(x, y, lr=0.00001, lmbda=lmbda, iterations=10_000)
    predictions = ridge.predict(x, model)
    mse = utils.mse(y, predictions)

    # Now use scikit-learn on the same dataset
    x_sk = copy.deepcopy(x_orig)
    y_sk = copy.deepcopy(y_orig)
    model_sk = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=lmbda))
    model_sk.fit(x_sk, y_sk)
    predictions_sk = model_sk.predict(x_sk)
    mse_sk = metrics.mean_squared_error(y_sk, predictions_sk)
    coef_sk = model_sk.named_steps['ridge'].coef_

    return model, mse, coef_sk, mse_sk


LAMDBAS_TO_TEST = [0.01, 0.1, 1, 10, 100, 1_000, 10_000]

for lmbda in LAMDBAS_TO_TEST:
    coef, mse, coef_sk, mse_sk = experiment1(lmbda)

    coef_str = ' '.join('{:8.2f}'.format(c) for c in coef.flatten())
    coef_str_sk = ' '.join('{:8.2f}'.format(c) for c in coef_sk.flatten())

    print(f'Our code:     MSE: {mse:12.5f} coefficients: {coef_str} lambda: {lmbda}')
    print(f'scikit-learn: MSE: {mse_sk:12.5f} coefficients: {coef_str_sk} lambda: {lmbda}')
