"""Common Spatial Pattern module for brainwave deep learning."""

from typing import Self

from numpy import ndarray

from tpv.diagonalize import Diagonalize


class CSP:
    """Common Spatial Pattern class for EEG signals."""

    def __init__(self: Self, a: ndarray, b: ndarray) -> None:
        """Initialize the object with two signals."""
        self._a, self._b = a, b
        a = a - a.mean(axis=-1, keepdims=True)
        a = a @ a.transpose(0, 2, 1) / (a.shape[-1] - 1)
        b = b - b.mean(axis=-1, keepdims=True)
        b = b @ b.transpose(0, 2, 1) / (b.shape[-1] - 1)
        c = Diagonalize(a + b)
        w = (c ** -0.5) @ c.dia_basis.T
        diag = Diagonalize(w @ a @ w.T)
        eigenvalues = diag.dia.diagonal()
        self._eigenvalues = eigenvalues / (1 - eigenvalues)
        self._projection = w.T @ diag.dia_basis
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
    def eigenvalues(self: Self) -> ndarray:
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
