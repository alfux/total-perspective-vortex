"""Common Spatial Module for brainwave deep learning."""

import argparse as arg
import logging
import sys
from argparse import Namespace
from pathlib import Path

import mne
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from numpy import ndarray
from numpy import random as rng


def sym_to_diag(m: ndarray) -> tuple[ndarray, ndarray]:
    """Diagonalise a symetric real valued matrix.

    Args:
        m (ndarray): The matrix.
    Returns:
        tuple[ndarray, ndarray]: (orthogonal basis matrix, diagonal matrix)
    """
    n = len(n)
    q = rng.standard_normal((n, 1))
    basis = []


def csp(a: ndarray, b: ndarray) -> ndarray:
    """Perform Common Spatial Pattern on matrix a and matrix b.

    Args:
        a (ndarray): matrix A.
        b (ndarray): matrix B.
    Returns:
        ndarray: basis vectors.
    """
    pass


def to_csp(file: Path) -> None:
    """Perform Common Spatial Pattern algorithm on the dataset.

    Args:
        file (Path): data file.
    """
    raw = mne.io.read_raw_edf(file)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    events, event_ids = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw, events, event_id={"T1": event_ids["T1"], "T2": event_ids["T2"]},
        preload=True, picks="eeg"
    )
    X = epochs.get_data(copy=True)
    y = epochs.events
    print(X.shape)
    print(y.shape)


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("file", type=str, help="data file")
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
        csp(av.file)
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
