"""Trigonalized object file"""

from typing import Self

import numpy as np
from numpy import ndarray
from numpy import random as rng


class Trigonalized:
    """Trigonalized class for symetric matrices."""

    _EPSILON = 1e-12

    def __init__(self: Self, matrix: ndarray) -> None:
        """Diagonalize a symetric matrix.

        Args:
            matrix (ndarray): symetric matrix.
        """
        if len(matrix.shape) != 2 or matrix.shape[0] < 2:
            raise RuntimeError("Not a matrix")
        if not np.array_equal(matrix.T, matrix):
            raise RuntimeError("Not symetric")
        self._m = matrix
        self._dim = len(matrix)
        self._o_basis = self._lanczos()
        self._trigonalised = self._o_basis.T @ matrix @ self._o_basis

    @property
    def trigonal(self: Self) -> ndarray:
        """Get the trigonal matrix.

        Returns:
            ndarray: trigonal matrix.
        """
        return self._trigonalised

    @property
    def basis(self: Self) -> ndarray:
        """Get the orthogonal change-of-basis matrix.

        Returns:
            ndarray: change-of-basis matrix.
        """
        return self._o_basis

    def _lanczos(self: Self) -> ndarray:
        """Perform Lanczos algorithm.

        Returns:
            ndarray: orthogonal diagonalisation matrix.
        """
        basis = np.array([[]] * self._dim)
        q = self._rng_orthonormal(basis)
        while basis.shape[1] != self._dim:
            basis = np.concat([basis, self._krylov(q)], axis=1)
            if basis.shape[1] < self._dim:
                q = self._rng_orthonormal(basis)
        return basis

    def _krylov(self: Self, q: ndarray) -> ndarray:
        """Compute a Krylov space orthonormal basis from self._m and q.

        Args:
            q (ndarray): starting normalized vector.
        Returns:
            ndarray: Krylov space's orthonormal basis.
        """
        basis = np.array([[]] * self._dim)
        norm = 1
        pre_norm = 1
        while norm > Trigonalized._EPSILON * pre_norm:
            basis = np.concat([basis, q / norm], axis=1)
            q = self._m @ basis[:, -1][:, None]
            pre_norm = np.linalg.norm(q)
            q = self._gram_schmidt(basis, q)
            norm = np.linalg.norm(q)
        return basis

    def _rng_orthonormal(self: Self, basis: ndarray) -> ndarray:
        """Find a random normalized vector orthogonal to given basis.

        Args:
            basis (ndarray): current basis.
        Returns:
            ndarray: random vector orthogonal to basis.
        """
        norm = 0
        if basis.shape[1] == 0:
            while norm == 0:
                q = rng.standard_normal((self._dim, 1))
                norm = np.linalg.norm(q)
        else:
            while norm == 0:
                q = rng.standard_normal((self._dim, 1))
                q = self._gram_schmidt(basis, q)
                norm = np.linalg.norm(q)
        return q / norm

    def _gram_schmidt(self: Self, basis: ndarray, q: ndarray) -> ndarray:
        """Gram-Schmidt orthonormalization process.

        Args:
            basis (ndarray): current orthonormal basis matrix (n x m).
            q (ndarray): vector to process (n x 1).
        Returns:
            ndarray: orthonormalized vector or 0.
        """
        for b in basis.T[:, :, None]:
            q -= q.T @ b * b
        return q
