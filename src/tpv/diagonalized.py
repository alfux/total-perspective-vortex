"""Diagonalized object file"""

from typing import Self

import numpy as np
from numpy import ndarray
from numpy import random as rng


class Diagonalized:
    """Diagonalized class for symetric matrices."""

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
        self._m = matrix.copy()
        self._dim = len(self._m)
        self._ot_basis = self._lanczos()
        self._tridiagonalised = self._ot_basis.T @ self._m @ self._ot_basis

    @property
    def tridiagonal(self: Self) -> tuple[ndarray, ndarray]:
        """Get the tridiagonalisation matrix and tridiagonal matrix.

        Returns:
            ndarray: orthogonal matrix, tridiagonal matrix.
        """
        return self._ot_basis, self._tridiagonalised

    def qr_algorithm(self: Self, a: ndarray) -> tuple[ndarray, ndarray]:
        """QR algorithm for matrix diagonalisation.

        Args:
            a (ndarray): a matrix.
        Retrurns:
            tuple[ndarray, ndarray]: Q, R
        """
        for _ in range(10):
            q, r = self._qr_householder(a)
            a = r @ q
        return q, r

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
        while norm > Diagonalized._EPSILON * pre_norm:
            basis = np.concat([basis, q / norm], axis=1)
            q = self._m @ basis[:, -1][:, None]
            pre_norm = np.linalg.norm(q)
            q = self._gram_schmidt_step(basis, q)
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
                q = self._gram_schmidt_step(basis, q)
                norm = np.linalg.norm(q)
        return q / norm

    def _gram_schmidt_step(self: Self, basis: ndarray, q: ndarray) -> ndarray:
        """Gram-Schmidt orthonormalization process on one vector.

        Args:
            basis (ndarray): current orthonormal basis matrix (n x m).
            q (ndarray): vector to process (n x 1).
        Returns:
            ndarray: orthonormalized vector or 0.
        """
        for b in basis.T[:, :, None]:
            q -= q.T @ b * b
        return q

    def _qr_decomposition(self: Self, a: ndarray) -> tuple[ndarray, ndarray]:
        """Perform Gram-Schmidt on a free vector family for QR decomposition.

        Args:
            a (ndarray): matrix with free vector family as columns.
        Returns:
            tuple[ndarray, ndarray]: orthogonal matrix, sup trigonal matrix.
        """
        q = a[:, 0, None]
        norm = np.linalg.norm(q)
        q /= norm
        r = np.zeros_like(q)
        r[0, 0] = norm
        for c in a.T[1:]:
            rn = []
            for v in q.T:
                rn.append(np.dot(v, c))
                c -= rn[-1] * v
            norm = np.linalg.norm(c)
            rn.append(norm)
            rn += [0] * (len(a.T) - len(rn))
            r = np.concat([r, np.array(rn)[:, None]], axis=1)
            q = np.concat([q, c[:, None] / norm], axis=1)
        return q, r

    def _qr_householder(self: Self, a: ndarray) -> tuple[ndarray, ndarray]:
        """Perform Householder QR decomposition.

        Args:
            a (ndarray): a matrix.
        Returns:
            tuple[ndarray, ndarray]: orthogonal matrix, sup trigonal matrix.
        """
        r = a.copy()
        q = np.eye(len(a), len(a))
        n = len(a)
        for i in range(len(a)):
            r_sub = r[i:, i:]
            n_sub = len(r_sub)
            x = r_sub[:, 0, None]
            norm = np.linalg.norm(x)
            y = np.zeros(n_sub)
            y[0] = -np.sign(x[0, 0]) * norm
            v = x - y[:, None]
            h = np.eye(n, n)
            h[i:, i:] = np.eye(n_sub, n_sub) - 2 * (v @ v.T) / (v.T @ v)
            r = h @ r
            q = q @ h
        return q, r
