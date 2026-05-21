"""Manage data preprocessing point of the pipeline."""
import os
from pathlib import Path
from typing import Generator, Self

import mne
from mne.datasets import eegbci
from mne.io import Raw
import matplotlib.pyplot as plt

import json


class Preprocessing:
    """Data preprocessing object."""

    def __init__(self: Self, directory: str) -> None:
        """Initialize the object.

        Args:
            directory (str): directory holding all files.
        """
        self._data = dict(self._create_data_structure(Path(directory)))
        with open("output.json", "w") as file:
            file.write(json.dumps(self._data, indent=2, default=str))

    def _create_data_structure(self: Self, directory: Path) -> Generator:
        """Create a data structure for easy navigation.

        Args:
            directory (Path): Directory containing brain waves data.
        Returns:
            dict: Data structure.
        """
        for item in os.listdir(directory):
            path = directory / item
            if path.is_dir():
                yield (item, dict(self._create_data_structure(path)))
            elif path.suffix == ".edf":
                yield (item, self._data_structure(path))

    def _data_structure(self: Self, path: Path) -> dict:
        """Create the file node of the data structure.

        Args:
            path (Path): path of a data file.
        Returns:
            dict: data structure node of a file.
        """
        raw = mne.io.read_raw_edf(path)
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        raw.pick("eeg")
        raw.load_data()
        raw.set_eeg_reference("average")
        events, eid = mne.events_from_annotations(raw)
        print(eid, events)
        return {
            "data": raw,
        }

    def filter(self: Self, key: str) -> Raw:
        """Filter a dataset.

        Args:
            key (str): The key path to access the data.
        """
        eegbci.standardize(self._raw)
        self._raw.set_montage("standard_1005")
        self._raw.pick("eeg")
        self._raw.set_eeg_reference("average")
        self._psd = self._raw.compute_psd()
        self._raw.filter(7, 30)
        self._psd = self._raw.compute_psd()
        self._psd.plot()
