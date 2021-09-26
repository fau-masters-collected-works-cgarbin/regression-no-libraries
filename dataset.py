
import numpy as np
import sys


def check_python_version():
    if sys.version_info < (3, 6):
        print('Requires Python 3.6 or higher')
        sys.exit(1)


def read_dataset(filename: str):
    dataset = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # Split into input and output (asumming the last column is the output)
    x = dataset[:, :-1]
    y = dataset[:, -1:]
    return x, y


def describe_data(title: str, data):
    print(f'\n\n{title}')
    print(f'Type: {type(data)}')
    print(f'Shape: {data.shape}')
    print('Data (first and last few rows):')
    print(data[:3])


if __name__ == "__main__":
    check_python_version()

    x, y = read_dataset('./Credit_N400_p9.csv')
    describe_data('X (input)', x)
    describe_data('Y (output)', y)
