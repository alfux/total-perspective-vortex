import argparse as arg
import logging
import sys
from argparse import Namespace

import numpy as np

from tpv.diagonalized import Diagonalized


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
        sym = np.array([
            [1, 2, 3, 1, 1],
            [2, 7, 0, 0, 1],
            [3, 0, 3, 9, 1],
            [3, 0, 9, 9, 1],
            [1, 1, 1, 1, 1]
        ])
        sym = (sym + sym.T) / 2
        tri = Diagonalized(sym)
        q, r = tri._qr_householder(sym)
        print(sym, np.round(q), np.round(r), np.round(q @ r), sep="\n\n")
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
