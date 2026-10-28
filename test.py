import argparse as arg
import logging
import sys
from argparse import Namespace

import numpy as np

from tpv.csp import CSP
from tpv.diagonalize import Diagonalize


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
    return av.parse_args()


def main() -> int:
    """Test main.

    Returns:
        int: return status 0 (success) 1 (error).
    """
    try:
        av = get_args(main.__doc__)
        fmt = "%(asctime)s | %(levelname)s - %(message)s"
        if av.debug:
            logging.basicConfig(level=logging.DEBUG, format=fmt)
        else:
            logging.basicConfig(level=logging.INFO, format=fmt)
        data1 = np.random.random((10, 5, 100))
        data2 = np.random.random((10, 5, 100))
        csp = CSP(data1, data2)
        csp.assert_csp()
        sym = np.random.random((10, 10))
        sym = (sym + sym.T) / 2
        dia = Diagonalize(sym)
        dia.assert_dia()
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
