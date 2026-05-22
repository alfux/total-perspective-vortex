"""Manage data preprocessing point of the pipeline."""
import os
from pathlib import Path
from typing import Callable, Generator, Self

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.widgets import Button
from mne.datasets import eegbci
from mne.io.edf.edf import RawEDF


class Preprocessing:
    """Data preprocessing object."""

    def __init__(self: Self, directory: str) -> None:
        """Initialize the object.

        Args:
            directory (str): directory holding all files.
        """
        mne.set_log_level("ERROR")
        self._path = Path(directory)
        self._data = dict(self._create_data_structure(self._path))
        self._buttons = set()

    def plot(self: Self) -> None:
        """Plot the dataset."""
        self._subplot(self._data, self._path)
        plt.show()

    def _create_data_structure(self: Self, directory: Path) -> Generator:
        """Create a data structure for easy navigation.

        Args:
            directory (Path): Directory containing brain waves data.
        Returns:
            dict: Data structure.
        """
        for item in sorted(os.listdir(directory)):
            path = directory / item
            if path.is_dir():
                yield (item, dict(self._create_data_structure(path)))
            elif path.suffix == ".edf":
                yield (item, self._data_structure(path))

    def _subplot(self: Self, data: dict, label: str) -> None:
        """Sub windows to choose run inside a directory.

        Args:
            data (dict): directory's datas.
        """
        lines = int(np.sqrt(len(data)))
        col = (len(data) // lines) + int(len(data) % lines != 0)
        data_keys = list(data.keys())
        fig, axes = plt.subplots(lines, col, label=label)
        buttons = set()
        for index in range(lines * col):
            i = index % lines
            j = index // lines
            if i + j * lines >= len(data_keys):
                axes[i, j].axis("off")
            else:
                key = data_keys[i + j * lines]
                sub = data[key]
                btn = Button(axes[i, j], Path(key).stem)
                if "data" in sub and isinstance(sub["data"], RawEDF):
                    btn.on_clicked(self._get_file_callback(sub["data"]))
                else:
                    btn.on_clicked(self._get_dir_callback(sub, key))
                buttons.add(btn)
        self._buttons.update(buttons)
        fig.canvas.mpl_connect("close_event", self._get_cleaner(buttons))
        fig.show()

    def _get_dir_callback(self: Self, sub_data: dict, label: str) -> Callable:
        """Create a directory button's callback to navigate inside.

        Args:
            sub_data (dict): data associated to the button's directory.
            label (str): name of the new window.
        """
        return lambda _: self._subplot(sub_data, label)

    def _get_cleaner(self: Self, buttons: set) -> Callable:
        """Create a buttons' cleaner for a closed window.

        Args:
            buttons (set): buttons to remove from memory.
        """
        return lambda _: self._buttons.difference_update(buttons)

    @staticmethod
    def _data_structure(path: Path) -> dict:
        """Create the file node of the data structure.

        Args:
            path (Path): path of a data file.
        Returns:
            dict: data structure node of a file.
        """
        raw = mne.io.read_raw_edf(path)
        events, eid = mne.events_from_annotations(raw)
        mapping = Preprocessing._mapping(path)
        eid = {v: mapping[str(k)] for k, v in eid.items()}
        sfreq = raw.info["sfreq"]
        return {
            "data": raw,
            "events": {e[0] / sfreq: eid[e[2]] for e in events}
        }

    @staticmethod
    def _mapping(path: Path) -> dict:
        """Create an event mapping for each run.

        Returns:
            dict: event mapping of the run.
        """
        run_number = int(path.stem[-2:])
        if run_number == 1 or run_number == 2:
            return {"T0": "X", "T1": "X", "T2": "X"}
        match (run_number - 3) % 4:
            case 0:
                return {"T0": "R", "T1": "EL", "T2": "ER"}
            case 1:
                return {"T0": "R", "T1": "IL", "T2": "IR"}
            case 2:
                return {"T0": "R", "T1": "EU", "T2": "ED"}
            case 3:
                return {"T0": "R", "T1": "IU", "T2": "ID"}

    @staticmethod
    def _get_file_callback(data: RawEDF) -> Callable:
        """Create a file button's callback to plot the data.

        Args:
            data (RawEDF): object managing the file's data.
        """
        return lambda _: data.plot()

    # def filter(self: Self, key: str) -> RawEDF:
    #     """Filter a dataset.

    #     Args:
    #         key (str): The key path to access the data.
    #     """
    #     eegbci.standardize(self._raw)
    #     self._raw.set_montage("standard_1005")
    #     self._raw.pick("eeg")
    #     raw.load_data()
    #     self._raw.set_eeg_reference("average")
    #     self._psd = self._raw.compute_psd()
    #     self._raw.filter(7, 30)
    #     self._psd = self._raw.compute_psd()
    #     self._psd.plot()
