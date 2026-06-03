"""Manage data preprocessing point of the pipeline."""
import os
from pathlib import Path
from typing import Callable, Generator, Self

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from mne.datasets import eegbci

# https://www.researchgate.net/publication/286691705_Selection_of_proper_frequency_band_and_compatible_features_for_left_and_right_hand_movement_from_EEG_signal_analysis


class Preprocessing:
    """Data preprocessing object."""

    EVENT_MAP = {
        "IE": {"T0": 'Rest', "T1": 'Rest', "T2": 'Rest'},
        "II": {"T0": 'Rest', "T1": 'Rest', "T2": 'Rest'},
        "UE": {"T0": "Rest", "T1": "Left", "T2": "Right"},
        "UI": {"T0": "Rest", "T1": "Left", "T2": "Right"},
        "BE": {"T0": "Rest", "T1": "Up", "T2": "Down"},
        "BI": {"T0": "Rest", "T1": "Up", "T2": "Down"}
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
        name = "Idle Eyes Opened"
        self._button(axes[0, 0], name, lambda _: self._file_menu("IE"))
        name = "Idle Eyes Closed"
        self._button(axes[1, 0], name, lambda _: self._file_menu("II"))
        name = "Unilateral Executed"
        self._button(axes[0, 1], name, lambda _: self._file_menu("UE"))
        name = "Unilateral Imagined"
        self._button(axes[1, 1], name, lambda _: self._file_menu("UI"))
        name = "Bilateral Exectued"
        self._button(axes[0, 2], name, lambda _: self._file_menu("BE"))
        name = "Bilateral Imagined"
        self._button(axes[1, 2], name, lambda _: self._file_menu("BI"))
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
        if col * lin < len(self._data[category]) + 1:
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
            self._button(
                ax, files[i].stem,
                lambda _: self._filter_menu(files[i], category)
            )
        elif i == len(files):
            self._button(ax, "Back", lambda _: self._menu())
        else:
            ax.set_axis_off()

    def _filter_menu(self: Self, file: Path, category: str) -> None:
        """Display the filtering menu for a file.

        Args:
            file (Path): path of the data file.
            category (str): category of the file.
        """
        self._clear()
        axes = self._fig.subplots(3, 2)
        raw = mne.io.read_raw_edf(file, preload=True)
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        self._button(axes[0, 0], "Raw", lambda _: raw.plot())
        filtered = raw.copy()
        filtered.filter(7, 30)
        self._button(axes[1, 0], "Filtered", lambda _: filtered.plot())
        raw_psd = raw.compute_psd()
        self._button(axes[0, 1], "PSD", lambda _: raw_psd.plot())
        filtered_psd = filtered.compute_psd()
        self._button(axes[1, 1], "Filtered PSD", lambda _: filtered_psd.plot())
        self._button(axes[2, 0], "Back", lambda _: self._file_menu(category))
        axes[2, 1].set_axis_off()
        self._fig.canvas.draw()

    def _button(self: Self, axis: Axes, name: str, callback: Callable) -> None:
        """Create a button in the current display.

        Args:
            axis (Axes): axis of the button.
            name (str): name of the button.
            callback (Callable): buttons callback.
        """
        button = Button(axis, name)
        button.on_clicked(callback)
        self._buttons.add(button)

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
