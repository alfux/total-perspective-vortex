"""Filter module."""

from typing import Self

import mne
from mne.datasets import eegbci
from mne.io.edf.edf import RawEDF

from tpv.csp import CSP
from tpv.loader import Loader


class Filter:
    """Filtering class used to process eeg signals in the pipeline."""

    def __init__(self: Self, data: list[RawEDF]) -> None:
        """Filter class for EEG signals."""
        self._raw = list(self._filter)

    def
