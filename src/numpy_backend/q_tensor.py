"""Q-tensor operations for the NumPy backend.

Provides construction, decomposition, energy computation and time-stepping
routines for the symmetric traceless Q-tensor on a 2D lattice.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.numpy_backend.diff import laplacian_2D_9_point_isotropic

_rng = np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def make_traceless_symmetric(arr: NDArray) -> None:
    """Enforce traceless symmetry on a ``(..., 2, 2)`` tensor field *in-place*.

    Parameters
    ----------
    arr : NDArray
        Tensor field of shape ``(..., 2, 2)``.
    """
    tmp = 0.5 * (arr[..., 0, 1] + arr[..., 1, 0])
    arr[..., 0, 1] = arr[..., 1, 0] = tmp
    tmp = 0.5 * (arr[..., 0, 0] + arr[..., 1, 1])
    arr[..., 0, 0] -= tmp
    arr[..., 1, 1] -= tmp


# ---------------------------------------------------------------------------
# LdG physics
# ---------------------------------------------------------------------------

def functional_derivative_LdG(
    Q: NDArray, a: float, b: float, K: float,
) -> NDArray:
    """Compute the functional derivative of the Landau-de Gennes free energy.

    Parameters
    ----------
    Q : NDArray
        Q-tensor field of shape ``(L, L, 2, 2)``.
    a, b, K : float
        Material parameters.

    Returns
    -------
    NDArray
        Functional derivative, same shape as *Q*.
    """
    Q_sq = Q @ Q
    tr_Q_sq = Q_sq[..., 0, 0] + Q_sq[..., 1, 1]
    answer = Q * (a + 0.5 * b * tr_Q_sq)[..., None, None]
    answer -= K * laplacian_2D_9_point_isotropic(Q)
    make_traceless_symmetric(answer)
    return answer


def compute_free_energy(
    Q: NDArray, a: float, b: float, K: float,
) -> NDArray:
    """Compute the LdG free-energy density on every lattice site.

    Parameters
    ----------
    Q : NDArray
        Q-tensor field of shape ``(L, L, 2, 2)``.
    a, b, K : float
        Material parameters.

    Returns
    -------
    NDArray
        Scalar field of shape ``(L, L)``.
    """
    Q_sq = Q @ Q
    tr_Q_sq = Q_sq[..., 0, 0] + Q_sq[..., 1, 1]
    f_local = 0.5 * a * tr_Q_sq + 0.125 * b * tr_Q_sq**2

    f_grad = np.zeros_like(f_local)
    for i in range(2):
        for j in range(2):
            grad_Q_ij = np.gradient(Q[..., i, j])
            grad_squared = grad_Q_ij[0] ** 2 + grad_Q_ij[1] ** 2
            f_grad += 0.5 * K * grad_squared

    return f_local + f_grad


def model_A_LdG(
    Q: NDArray, a: float, b: float, K: float, dt: float, gamma: float,
) -> NDArray:
    """One explicit-Euler step of Model-A relaxational dynamics.

    Parameters
    ----------
    Q : NDArray
        Current Q-tensor field.
    a, b, K : float
        Material parameters.
    dt : float
        Time step.
    gamma : float
        Kinetic coefficient.

    Returns
    -------
    NDArray
        Updated Q-tensor field.
    """
    return Q - dt * gamma * functional_derivative_LdG(Q, a, b, K)


# ---------------------------------------------------------------------------
# Q  <-->  (n, S) conversions
# ---------------------------------------------------------------------------

def get_n_S_from_Q(Q: NDArray) -> tuple[NDArray, NDArray]:
    """Extract the director and scalar order parameter from the Q-tensor.

    Parameters
    ----------
    Q : NDArray
        Q-tensor field of shape ``(L, L, 2, 2)``.

    Returns
    -------
    n : NDArray
        Director field of shape ``(L, L, 2)``.
    S : NDArray
        Scalar order-parameter field of shape ``(L, L)``.
    """
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    S = eigenvals[..., -1]
    n = eigenvecs[..., -1]
    return n, S


def get_Q_from_n_S(n: NDArray, S: NDArray | float) -> NDArray:
    """Construct a Q-tensor field from a director and order parameter.

    Parameters
    ----------
    n : NDArray
        Director field of shape ``(L, L, 2)``.
    S : NDArray or float
        Scalar order parameter (field or constant).

    Returns
    -------
    NDArray
        Q-tensor field of shape ``(L, L, 2, 2)``.
    """
    outer = n[..., :, None] * n[..., None, :]
    I = np.eye(2)
    S_arr = np.asarray(S)
    Q = 2 * S_arr[..., None, None] * (outer - 0.5 * I)
    return Q


# ---------------------------------------------------------------------------
# Global diagnostics
# ---------------------------------------------------------------------------

def compute_global_order_parameter(Q: NDArray) -> float:
    """Compute the global scalar order parameter from the spatially-averaged Q.

    Parameters
    ----------
    Q : NDArray
        Q-tensor field of shape ``(L, L, 2, 2)`` (may contain NaNs).

    Returns
    -------
    float
        Largest eigenvalue of the lattice-averaged Q-tensor.
    """
    Q_avg = np.nanmean(Q, axis=(0, 1))
    eigenvalues, _ = np.linalg.eigh(Q_avg)
    return float(eigenvalues[-1])


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------

def make_random_Q(L: int, S: float) -> NDArray:
    """Generate a random Q-tensor field with uniform random director angles.

    Parameters
    ----------
    L : int
        Side length of the square lattice.
    S : float
        Scalar order parameter used everywhere.

    Returns
    -------
    NDArray
        Q-tensor field of shape ``(L, L, 2, 2)``.
    """
    theta = 2 * np.pi * _rng.random((L, L))
    n = np.empty((L, L, 2))
    n[..., 0] = np.cos(theta)
    n[..., 1] = np.sin(theta)

    Q = np.empty((L, L, 2, 2))
    for i in [0, 1]:
        for j in [0, 1]:
            Q[..., i, j] = 2 * S * (n[..., i] * n[..., j] - 0.5 * (1 if i == j else 0))
    return Q
