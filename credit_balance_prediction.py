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

# Ensure the code is working before using it
test.verbose = False
test.test_all()

x, y = utils.read_dataset('Credit_N400_p9.csv')

# Encode the categorical values
utils.encode_binary_cateogry(x, column=6, one_value='Female')  # gender
utils.encode_binary_cateogry(x, column=7, one_value='Yes')  # student
utils.encode_binary_cateogry(x, column=8, one_value='Yes')  # married

# Save a copy of the original data because the model fitting code may change it
x_orig = copy.deepcopy(x)
y_orig = copy.deepcopy(y)

utils.scale(x)
utils.center(y)

model = ridge.fit(x, y, lr=0.0001, lmbda=1000, iterations=1000)
predictions = ridge.predict(x, model)
mse = utils.mse(y, predictions)

print('Our code')
print(f'MSE: {mse}')
print(f'Original input (standardized values):\n{y[:3]}')
print(f'Predicted values (standardized values):\n{predictions[:3]}')
