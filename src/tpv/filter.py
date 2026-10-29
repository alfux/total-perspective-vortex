"""Filter module."""

from typing import Self

import mne
from mne.datasets import eegbci
from mne.io import BaseRaw

from tpv.csp import CSP


class Filter:
    """Filtering class used to process eeg signals in the pipeline."""

    def __init__(self: Self, file: BaseRaw) -> None:
        """Filter class for EEG signals."""
        raw = mne.io.read_raw_edf(file)
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        events, event_ids = mne.events_from_annotations(raw)
        epochs = mne.Epochs(
            raw, events, event_id={
                "T1": event_ids["T1"], "T2": event_ids["T2"]},
            preload=True, picks="eeg"
        )
        x = epochs["T1"].get_data(copy=True)
        y = epochs["T2"].get_data(copy=True)
        csp = CSP(x, y)
        x = csp.filter(x)
        y = csp.filter(y)
        print(x.shape)
        print(y.shape)
