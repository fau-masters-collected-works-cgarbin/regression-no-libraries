# Ridge regression without using libraries

An implementation of batch gradient descent using Ridge regularization without any statistical or machine learning library. All steps are done by hand, using matrix operations as much as possible.

Reference: [An Introduction to Statistical Learrning, James et al.](https://web.stanford.edu/~hastie/ISLRv2_website.pdf), second edition, section 6.2 Shrinkage Methods.

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

With that in place, we can open and execute the Juptyer notebook. Because we are using Jupyter, we need one more step to make the virtual environment visible to Jupyter ([source 1](https://stackoverflow.com/a/49309403), [source 2](https://ripon-banik.medium.com/jupyter-notebook-is-unable-to-find-module-in-virtual-environment-fa0725c3f8fd)):

- Start Jupyter (will open a browser window), with the kernel set to the local environment)
  - `ipython kernel install --user --name=env`
  - `jupyter lab`.
- Open the notebook `ridge_regression.ipynb`.
- Set the kernel to `env`, the virtual environment.

## How the code is organized

The code is organized in these files:

- `ridge_regression.ipynb`: The Jupyter notebook with the code to run the experiments and display the results, including the graphs.
- `ridge.py`: The code to calculate coefficients using ridge regression.
- `utils.py`: Supporting functions, e.g. read a dataset from a CSV file, scale and center matrices, split matrices into folds, etc.
- `test.py`: The code to test the utility functions and the ridge regression code.

## Testing code changes

If you change the utility code or the ridge code, test the changes with `python test.py`. All tests must pass (or be adjusted to the new code).

If you add more public functions, please add the corresponding test for them.
