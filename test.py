import argparse as arg
import logging
import sys
from argparse import Namespace

import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci


def test_features() -> None:
    """Test MNE features"""
    raw: mne.io.Raw = mne.io.read_raw_edf("./data/S001/S001R02.edf")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.compute_psd().plot(picks="data", exclude="bads", amplitude=False)
    raw.plot(duration=5, n_channels=64)
    plt.show()


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
        test_features()
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
