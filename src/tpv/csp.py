"""Common Spatial Pattern module for brainwave deep learning."""

from typing import Self

import numpy as np
from numpy import ndarray

from tpv.diagonalize import Diagonalize


class CSP:
    """Common Spatial Pattern class for EEG signals."""

    def __init__(self: Self, a: ndarray, b: ndarray) -> None:
        """Initialize the object with two signals.

        Args:
            a (ndarray): First dataset.
            b (ndarray): Second dataset.
        """
        self._a, self._b = a, b
        a = a - a.mean(axis=-1, keepdims=True)
        a = a @ a.transpose(0, 2, 1) / (a.shape[-1] - 1)
        self._cova = a.mean(axis=0)
        b = b - b.mean(axis=-1, keepdims=True)
        b = b @ b.transpose(0, 2, 1) / (b.shape[-1] - 1)
        self._covb = b.mean(axis=0)
        cov = Diagonalize(self._cova + self._covb)
        whitening = np.diag(cov.dia.diagonal() ** -0.5) @ cov.dia_basis.T
        self._diag = Diagonalize(whitening @ self._cova @ whitening.T)
        eigenvalues = self._diag.dia.diagonal()
        self._eigenvalues = eigenvalues / (1 - eigenvalues)
        self._projection = self._diag.dia_basis.T @ whitening
        self._projected_a = self._projection @ self._a
        self._projected_b = self._projection @ self._b

    @property
    def projection(self: Self) -> ndarray:
        """Get the signal projection matrix

        Returns:
            ndarray: Signal projection.
        """
        return self._projection

    @property
    def variance_ratios(self: Self) -> ndarray:
        """Get eigenvalues associated to each column of the projection matrix.

        Returns:
            ndarray: Eigenvalues.
        """
        return self._eigenvalues

    @property
    def projected_a(self: Self) -> ndarray:
        """Get filtered a signal.

        Returns:
            ndarray: projected a signal.
        """
        return self._projected_a

    @property
    def projected_b(self: Self) -> ndarray:
        """Get filtered b signal.

        Returns:
            ndarray: projected a signal.
        """
        return self._projected_b

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
            np.diag(self._eigenvalues / (1 + self._eigenvalues)),
            atol=1e-8
        )
        return True
