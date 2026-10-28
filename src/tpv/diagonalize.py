"""Diagonalize module for symetric matrices.

Further optimizations can be implemented later.
"""

import sys
from typing import Self

import numpy as np
from numpy import ndarray
from numpy import random as rng


class Diagonalize:
    """Diagonalize class for symetric matrices."""

    _EPSILON = 1e-12
    _MAX_ITERATIONS = int(1e6)

    def __init__(self: Self, matrix: ndarray) -> None:
        """Diagonalize a symetric matrix.

        Args:
            matrix (ndarray): symetric matrix.
        """
        if len(matrix.shape) != 2 or matrix.shape[0] < 2:
            raise RuntimeError(f"Not a matrix:\n\n{matrix}")
        if not np.allclose(matrix.T, matrix):
            raise RuntimeError(f"Not symetric:\n\n{matrix}")
        self._m = matrix.copy()
        self._dim = len(self._m)
        self._tri_basis = self._lanczos()
        self._tri = self._tri_basis.T @ self._m @ self._tri_basis
        self._dia_basis, self._dia = self._qr_algorithm(self._tri)
        self._dia_basis = self._tri_basis @ self._dia_basis

    @property
    def raw(self: Self) -> ndarray:
        """Original matrix getter.

        Returns:
            ndarray: Original matrix.
        """
        return self._m

    @property
    def dia(self: Self) -> ndarray:
        """Diagonalized matrix getter.

        Returns:
            ndarray: Diagonalized matrix.
        """
        return self._dia

    @property
    def dia_basis(self: Self) -> ndarray:
        """Diagonilization orthogonal basis matrix getter.

        Returns:
            ndarray: Diagonilization orthogonal basis.
        """
        return self._dia_basis

    @property
    def tri(self: Self) -> tuple[ndarray, ndarray]:
        """Get the tridiagonalized.

        Returns:
            ndarray: Tridiagonalized matrix.
        """
        return self._tri

    @property
    def tri_basis(self: Self) -> ndarray:
        """Tridiagonilization orthogonal basis matrix getter.

        Returns:
            ndarray: Tridiagonilization orthogonal basis.
        """
        return self._tri_basis

    def assert_dia(self: Self) -> bool:
        """Test validity of diagonalization.

        Raise an AssertionError if failed.
        """
        np.testing.assert_allclose(
            self.tri_basis.T @ self._m @ self.tri_basis,
            self._tri,
            atol=1e-8
        )
        np.testing.assert_allclose(
            self._dia_basis.T @ self._m @ self._dia_basis,
            self._dia,
            atol=1e-8
        )
        return True

    def _qr_algorithm(self: Self, a: ndarray) -> tuple[ndarray, ndarray]:
        """QR algorithm for matrix diagonalization.

        Args:
            a (ndarray): a matrix.
        Retrurns:
            tuple[ndarray, ndarray]: Q, D
        """
        if a.shape[0] == 1:
            return np.identity(1), a
        q_final = np.eye(a.shape[0])
        for _ in range(self._MAX_ITERATIONS):
            q, r = self._qr_householder(a)
            a = r @ q
            q_final = q_final @ q
            for i in range(a.shape[0] - 1):
                scale = np.abs(a[i, i]) + np.abs(a[i + 1, i + 1])
                if np.abs(a[i + 1, i]) <= self._EPSILON * scale:
                    b, c = a[0: i + 1, 0: i + 1], a[i + 1:, i + 1:]
                    q, r = np.zeros_like(q_final), np.zeros_like(q_final)
                    qb, rb = self._qr_algorithm(b)
                    q[0: i + 1, 0: i + 1] = qb
                    r[0: i + 1, 0: i + 1] = rb
                    qc, rc = self._qr_algorithm(c)
                    q[i + 1:, i + 1:] = qc
                    r[i + 1:, i + 1:] = rc
                    return q_final @ q, r
        print(f"Warning: {self._MAX_ITERATIONS} reached.", file=sys.stderr)
        return q_final, a

    def _diagless_norm(self: Self, a: ndarray) -> float:
        """Compute Frobenius norm of a matrix without it's diagonal.

        Args:
            a (ndarray): The matrix.
        Returns:
            float: The norm.
        """
        return np.sqrt(
            np.linalg.norm(a) ** 2 - np.linalg.norm(a.diagonal()) ** 2
        )

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
        while basis.shape[1] < self._dim and norm > self._EPSILON * pre_norm:
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
            if norm == 0:
                continue
            y = np.zeros(n_sub)
            sign = 1 if x[0, 0] >= 0 else -1
            y[0] = -sign * norm
            v = x - y[:, None]
            h = np.eye(n, n)
            h[i:, i:] = np.eye(n_sub, n_sub) - 2 * (v @ v.T) / (v.T @ v)
            r = h @ r
            q = q @ h
        return q, r
