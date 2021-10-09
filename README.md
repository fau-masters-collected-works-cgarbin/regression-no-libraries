# Ridge regression without using libraries

An implementation of batch gradient descent using Ridge regularization without any statistical or machine learning library. All steps are done by hand, using matrix operations as much as possible.

## Setting up the project

- Install Python 3.6 or higher
- Go into this repository's directory
- Create a Python [environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment):
  `python3 -m venv env`
- Activate the environmnet: `source env/bin/activate` (Linux/Mac) or `.\env\Scripts\activate` (Windows)
- Upgrade pip: `python -m pip install --upgrade pip`
- Install the Python packages: `pip install -r requirements.txt`

Because we are using Jupyter, we need one more step to make the virtual environment visible to Jupyter ([source 1](https://stackoverflow.com/a/49309403), [source 2](https://ripon-banik.medium.com/jupyter-notebook-is-unable-to-find-module-in-virtual-environment-fa0725c3f8fd)):

- `ipython kernel install --user --name=env`

With that in place, we can open and execute the Juptyer notebook:

- Start Jupyter (will open a browser window): `jupyter lab`
- Open the notebook
- Set the kernel to `env`, the virtual environment

## How the code is organized

TBD: add list of files and their contents

## Testing code changes

TBD: how to test coe changes
