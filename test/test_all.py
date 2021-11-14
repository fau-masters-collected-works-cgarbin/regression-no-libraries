"""Test all the modules in the package.

Test the utility and regression code in the package, in a logicial order: utilities first,
then ridge, then the other modules (because in some cases we use ridge to test other modules).
"""

import test_utils
import test_ridge
import test_elastic_net


def test_all(verbose: bool = True) -> None:
    """Run all the tests."""
    test_utils.test_all(verbose)
    test_ridge.test_all(verbose)
    test_elastic_net.test_all(verbose)

    if verbose:
        print('\nAll tests passed')


if __name__ == "__main__":
    test_all()
