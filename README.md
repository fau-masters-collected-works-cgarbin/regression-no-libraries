# Linear and logistic regressions without using libraries

Implementation of linear and logistic regression without using libraries. All steps are done by hand, using matrix operations as much as possible.

- Ridge regression with batch gradient descent (the loss function is differentiable)
- Elastic net with coordinate descent (the loss function is not differentiable)
- Logistic regression with batch gradient descent (the loss function is differentiable)

References

- [The Elements of Statistical Learning, Hastie et al.](https://web.stanford.edu/~hastie/ISLRv2_website.pdf), second edition, section 3.4 Shrinkage Methods and section 4.4 Logistic Regression.
- [An Introduction to Statistical Learning, James et al.](https://web.stanford.edu/~hastie/ISLRv2_website.pdf), second edition, section 6.2 Shrinkage Methods and section 4.3 Logistic Regression.

## How the code is organized

The code is organized as follows:

- Notebooks to demonstrate the different methods:
  - [`ridge_regression.ipynb`](./ridge_regression.ipynb): The Jupyter notebook with the code to run the ridge regression experiments and display the results, including the graphs.
  - [`elastic_net_regression.ipynb`](./elastic_net_regression.ipynb): The Jupyter notebook with the code to run the elastic net regression experiments and display the results, including the graphs.
  - [`logistic_regression.ipynb`](./logistic_regression.ipynb): The Jupyter notebook with the code to run the logistic regression experiments and display the results, including the graphs.
- Implementation of the algorithms:
  - [`ridge.py`](./ridge.py): The code to calculate coefficients using ridge regression.
  - [`elastic_net.py`](./elastic_net.py): The code to calculate coefficients using elastic net regression.
  - [`logistic.py`](./logistic.py): The code to calculate coefficients using logistic regression.
- Supporting files:
  - [`utils.py`](./utils.py): Supporting functions, e.g. read a dataset from a CSV file, scale and center matrices, split matrices into folds, etc.
  - `test`: The code to test the utility functions and the regression code.
  - `data`: The datasets used in the experiments.

## Setting up the project

- Install Python 3.6 or higher.
- Go into this repository's directory.
- Create a Python [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment):
  - `python3 -m venv env`
- Activate the environmnet:
  - `source env/bin/activate` (Linux or Mac)
  - `.\env\Scripts\activate` (Windows).
- Upgrade pip:
  - `python -m pip install --upgrade pip`
- Install the Python packages:
  - `pip install -r requirements.txt`

## Running the code in a Jupyter notebook

Because we are using Jupyter, we need one more step to make the virtual environment visible to Jupyter ([source 1](https://stackoverflow.com/a/49309403), [source 2](https://ripon-banik.medium.com/jupyter-notebook-is-unable-to-find-module-in-virtual-environment-fa0725c3f8fd)):

- `ipython kernel install --user --name=env`

With that in place, we can open and execute the Jupyter notebook:

- Start Jupyter (will open a browser window), with the kernel set to the local environment)
  - `jupyter lab`
- Open one of the notebooks: `ridge_regression.ipynb`, `elastic_net_regression.ipynb` or `logistic_regression.ipynb`.
- Set the kernel to `env`, the virtual environment.

If you get errors when importing Python modules, stop Jupyter, exit the virtual environment and re-enter it:

- `Ctrl-C` twice on the terminal where you ran `jupyter lab`.
- Close the notebook browser window.
- `deactivate` (Linux or Mac) or `.\env\Scripts\deactivate` (Windows)
- `source env/bin/activate` (Linux or Mac) or `.\env\Scripts\activate` (Windows)
- `jupyter lab`

## Testing code changes

If you change the utility code or the regression code, test the changes with:

```bash
cd test
python test_all.py
```

All tests must pass (or be adjusted to the new code).

If you add more public functions, please add the corresponding test.
