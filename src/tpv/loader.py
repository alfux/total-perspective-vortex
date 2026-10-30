"""Loader class is used to load an EEG profile from a subject and a task."""

from pathlib import Path
from typing import Self, Generator

import mne
from mne.datasets import eegbci
from mne.io.edf.edf import RawEDF


class Loader:
    """Loader class's instances hold a subject experiments."""

    def __init__(self: Self, subject_id: int, root: Path = "./data") -> None:
        """"""
        self._subject = "S" + str(subject_id).zfill(3)
        self._directory = Path(root + "/" + self._subject)
        self._raw = {
            "IDLE_OPENED": list(self._load("R01")),
            "IDLE_CLOSED": list(self._load("R02")),
            "LEFT_RIGHT_OPENED": list(self._load("R03", "R07", "R011")),
            "LEFT_RIGHT_CLOSED": list(self._load("R04", "R08", "R012")),
            "UP_DOWN_OPENED": list(self._load("R05", "R09", "R013")),
            "UP_DOWN_CLOSED": list(self._load("R06", "R10", "R014")),
        }

    def __getitem__(self: Self, key: str) -> list[RawEDF]:
        """Select an experiment type.

        Args:
            key (str):
            ```
                IDLE_OPENED
                IDLE_CLOSED
                LEFT_RIGHT_OPENED
                LEFT_RIGHT_CLOSED
                UP_DOWN_OPENED
                UP_DOWN_CLOSED
            ```
        """
        return self._raw[key]

    def _load(self: Self, *experiments: list) -> Generator[RawEDF]:
        """Yield a subject experiment's data.

        Args:
            *experiments (str): experiments to laod (R01, R13, etc...)
        Yields:
            BaseRaw: raw edf object.
        """
        for exp in experiments:
            raw = mne.io.read_raw_edf(self._directory / exp + ".edf")
            eegbci.standardize(raw)
            raw.set_montage("standard_1005")
            raw.pick("eeg")
            yield raw
