"""Manage data preprocessing point of the pipeline."""
import os
from pathlib import Path
from typing import Any, Callable, Generator, Self

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from mne.datasets import eegbci
from mne.io.edf.edf import RawEDF

import json


class Preprocessing:
    """Data preprocessing object."""

    EVENT_MAP = {
        "IE": {"T0": 'R', "T1": 'R', "T2": 'R'},
        "II": {"T0": 'R', "T1": 'R', "T2": 'R'},
        "UE": {"T0": "R", "T1": "L", "T2": "R"},
        "UI": {"T0": "R", "T1": "L", "T2": "R"},
        "BE": {"T0": "R", "T1": "U", "T2": "D"},
        "BI": {"T0": "R", "T1": "U", "T2": "D"}
    }

    def __init__(self: Self, directory: str) -> None:
        """Initialize the object.

        Args:
            directory (str): directory holding all files.
        """
        mne.set_log_level("ERROR")
        self._path = Path(directory)
        self._data = dict(self._create_data_structure(self._path))
        self._fig = plt.figure(figsize=(16, 9))
        self._buttons = set()

    def menu(self: Self) -> None:
        """Open the menu"""
        self._menu()
        plt.show()

    def _menu(self: Self) -> None:
        """Creation of the menu."""
        self._clear()
        axes = self._fig.subplots(2, 3)
        ie = Button(axes[0, 0], "Idle Eyes Opened")
        ie.on_clicked(lambda _: self._file_menu("IE"))
        ii = Button(axes[1, 0], "Idle Eyes Closed")
        ii.on_clicked(lambda _: self._file_menu("II"))
        ue = Button(axes[0, 1], "Unilateral Executed")
        ue.on_clicked(lambda _: self._file_menu("UE"))
        ui = Button(axes[1, 1], "Unilateral Imagined")
        ui.on_clicked(lambda _: self._file_menu("UI"))
        be = Button(axes[0, 2], "Bilateral Exectued")
        be.on_clicked(lambda _: self._file_menu("BE"))
        bi = Button(axes[1, 2], "Bilateral Imagined")
        bi.on_clicked(lambda _: self._file_menu("BI"))
        self._buttons.update({ie, ii, ue, ui, be, bi})
        self._fig.canvas.draw()

    def _clear(self: Self) -> None:
        """Clear the figure and buttons."""
        self._fig.clear()
        self._buttons.clear()

    def _file_menu(self: Self, category: str) -> None:
        """Open category menu.

        Args:
            categroy (str): chosen category.
        """
        self._clear()
        col = int(np.ceil(np.sqrt(len(self._data[category]))))
        lin = int(np.ceil(len(self._data[category]) / col))
        if col * lin < len(self._data[category]) + 2:
            lin += 1
        axes = self._fig.subplots(lin, col)
        for i in range(lin):
            for j in range(col):
                self._file_button(axes[i, j], category, i * col + j)
        self._fig.canvas.draw()

    def _file_button(self: Self, ax: Axes, category: str, i: int) -> None:
        """Create the raw file button.

        Args:
            ax (Axes): Axes of the subplot holding the button.
            category (list): category of the files.
            i (int): index of the current file int its catetgory.
        """
        files = self._data[category]
        if i < len(files):
            button = Button(ax, files[i].stem)
            button.on_clicked(lambda _: mne.io.read_raw_edf(files[i]).plot())
            self._buttons.add(button)
        elif i == len(files):
            button = Button(ax, "Filter")
            button.on_clicked(lambda _: self._channels_menu(category))
            self._buttons.add(button)
        elif i == len(files) + 1:
            button = Button(ax, "Back")
            button.on_clicked(lambda _: self._menu())
            self._buttons.add(button)
        else:
            ax.set_axis_off()

    def _channels_menu(self: Self, category: str) -> None:
        """Open channels menu.

        Args:
            category (str): the current category.
        """
        self._clear()
        axes = self._fig.subplots(9, 8)
        for i in range(8):
            for j in range(8):
                channel = i * 8 + j
                button = Button(axes[i, j], "CH" + str(channel))
                button.on_clicked(self._channel_callback(category, channel))
                self._buttons.add(button)
        button = Button(axes[8, 0], "Back")
        button.on_clicked(lambda _: self._file_menu(category))
        self._buttons.add(button)
        for j in range(1, 8):
            axes[8, j].set_axis_off()
        self._fig.canvas.draw()

    def _channel_callback(self: Self, category: str, channel: int) -> Callable:
        """Create the channel callback function.

        Args:
            category (str): current file category.
            channel (int): current channel.
        Returns:
            Callable: the button's callback.
        """

        def callback(*_: Any) -> None:
            """Group channel by event, Fourier transform and plot."""
            data = [mne.io.read_raw_edf(f) for f in self._data[category]]
            if category[-1] == 'E':
                data += [mne.io.read_raw_edf(f) for f in self._data["IE"]]
            else:
                data += [mne.io.read_raw_edf(f) for f in self._data["II"]]
            groups = {"T0": [], "T1": [], "T2": []}
            for d in data:
                d.pick(channel)
                events, eid = mne.events_from_annotations(d)
                epochs = mne.Epochs(d, events, eid, 0, 4.1, None, preload=True)
                spectrum = epochs["T0"].compute_psd()
                groups["T0"].add(spectrum.get_data(return_freqs=True))
                if "T1" in epochs:
                    spectrum = epochs["T1"].compute_psd()
                    groups["T1"].add(spectrum.get_data(return_freqs=True))
                if "T2" in epochs:
                    spectrum = epochs["T2"].compute_psd()
                    groups["T2"].add(spectrum.get_data(return_freqs=True))
            self._plot_groups(groups, category, channel)

        return callback

    def _plot_groups(
        self: Self, groups: dict, category: str, channel: int
    ) -> None:
        """Plot grouped transformed data.

        Args:
            groups (dict): groups of same event.
            category (str): current event category.
            channel (int): current displayed channel.
        """
        pass

    def _create_data_structure(self: Self, directory: Path) -> Generator:
        """Create a data structure for easy navigation.

        Args:
            directory (Path): Directory containing brain waves data.
        Returns:
            dict: Data structure.
        """
        files = list(self._find_files(directory))
        data = {"IE": [], "II": [], "UE": [], "UI": [], "BE": [], "BI": []}
        for file in files:
            match int(file.stem[-2:]):
                case 1:
                    data["IE"].append(file)
                case 2:
                    data["II"].append(file)
                case 3 | 7 | 11:
                    data["UE"].append(file)
                case 4 | 8 | 12:
                    data["UI"].append(file)
                case 5 | 9 | 13:
                    data["BE"].append(file)
                case 6 | 10 | 14:
                    data["BI"].append(file)
        return data

    def _find_files(self: Self, directory: Path) -> Generator:
        """Create a list of edf files present in the directory.

        Args:
            directory (Path): Directory containing brain waves data.
        Yields:
            Path: path of the edf file.
        """
        for item in sorted(os.listdir(directory)):
            path = directory / item
            if path.is_dir():
                yield from self._find_files(path)
            elif path.suffix == ".edf":
                yield path

    def _get_psd_callback(self: Self, path: str) -> Callable:
        """PSD filter and plot.

        Args:
            path (str): path of the data file.
        """

        def callback(_: Any) -> None:
            """PSD callback function."""
            data = mne.io.read_raw_edf(path)
            eegbci.standardize(data)
            data.set_montage("standard_1005")
            data.pick("eeg")
            data.load_data()
            data.set_eeg_reference("average")
            fig = data.compute_psd().plot(show=False)
            fig.canvas.manager.set_window_title("bite")
            fig.show()

        return callback

    @staticmethod
    def _get_events(raw: RawEDF, mapping: dict) -> dict:
        """Extract an EDF events.

        Args:
            data (RawEDF): the data object.
            mapping (dict): event mapping depending on the path.
        """
        events, eid = mne.events_from_annotations(raw)
        eid = {v: mapping[str(k)] for k, v in eid.items()}
        sfreq = raw.info["sfreq"]
        return {e[0] / sfreq: eid[e[2]] for e in events}

    @staticmethod
    def _event_mapping(path: Path) -> dict:
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
