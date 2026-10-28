"""Common Spatial Pattern module for brainwave deep learning."""

from typing import Self

import numpy as np
from numpy import ndarray

from tpv.diagonalize import Diagonalize


class CSP:
    """Common Spatial Pattern class for EEG signals."""

    def __init__(self: Self, a: ndarray, b: ndarray, filter_range: int = 2) -> None:
        """Initialize the object with two signals sets.

        Args:
            a (ndarray): First dataset (n_epochs, n_channels, n_times).
            b (ndarray): Second dataset (n_epochs, n_channels, n_times).
            pick (tuple): Number of (top, bottom) filters to keep.
        """
        if len(a.shape) != 3 or len(b.shape) != 3 or a.shape[1] != b.shape[1]:
            raise ValueError(f"invalid data shape a:{a.shape} and b:{b.shape}")
        self._a, self._b = a, b
        a = a - a.mean(axis=-1, keepdims=True)
        a = a @ a.transpose(0, 2, 1) / (a.shape[-1] - 1)
        self._cova = a.mean(axis=0)
        b = b - b.mean(axis=-1, keepdims=True)
        b = b @ b.transpose(0, 2, 1) / (b.shape[-1] - 1)
        self._covb = b.mean(axis=0)
        cov = Diagonalize(self._cova + self._covb)
        whitening = np.diag(cov.dia.diagonal() ** -0.5) @ cov.dia_basis.T
        diag = Diagonalize(whitening @ self._cova @ whitening.T)
        self._eigenvalues = diag.dia.diagonal()
        indices = np.argsort(self._eigenvalues)[::-1]
        self._eigenvalues = self._eigenvalues[indices]
        self._projection = diag.dia_basis[:, indices].T @ whitening
        self.set_filter_range(filter_range)

    def set_filter_range(self: Self, filter_range: int) -> None:
        """Set spatial filters range.

        Args:
            n (int): Number of channels to pick from top and from bottom.
        """
        if filter_range <= 0:
            raise ValueError(f"filter range can't be negative: {filter_range}")
        if filter_range <= self._projection.shape[0] // 2:
            self._filtered_projection = np.concat([
                self._projection[:filter_range],
                self._projection[-filter_range:]
            ])
        else:
            self._filtered_projection = self._projection

    def filter(self: Self, signal: ndarray):
        """Filter signals with the projection matrix and the selection range.

        Args:
            signal (ndarra): signal or array of signals.
        """
        return self._filtered_projection @ signal

    def assert_csp(self: Self) -> bool:
        """Test validity of the CSP algorithm with given data.

        Raise an AssertionError if failed.
        """
        np.testing.assert_allclose(
            self._projection @ (self._cova + self._covb) @ self._projection.T,
            np.eye(self._cova.shape[0]),
            atol=1e-8
        )
        np.testing.assert_allclose(
            self._projection @ self._cova @ self._projection.T,
            np.diag(self._eigenvalues),
            atol=1e-8
        )
        return True
