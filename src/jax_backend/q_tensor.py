"""Q-tensor operations for the JAX backend.

Mirrors the NumPy backend API but uses JAX arrays and ``@jit`` for
GPU acceleration.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, Array

from src.jax_backend.diff import laplacian_2D_9_point_isotropic


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

@jit
def make_traceless_symmetric(arr: Array) -> Array:
    """Enforce traceless symmetry on a ``(..., 2, 2)`` tensor field.

    Unlike the NumPy variant this returns a *new* array (JAX arrays are
    immutable).

    Parameters
    ----------
    arr : jax.Array
        Tensor field of shape ``(..., 2, 2)``.

    Returns
    -------
    jax.Array
        Traceless symmetric tensor field.
    """
    tmp = 0.5 * (arr[..., 0, 1] + arr[..., 1, 0])
    arr = arr.at[..., 0, 1].set(tmp)
    arr = arr.at[..., 1, 0].set(tmp)
    trace = 0.5 * (arr[..., 0, 0] + arr[..., 1, 1])
    arr = arr.at[..., 0, 0].add(-trace)
    arr = arr.at[..., 1, 1].add(-trace)
    return arr


# ---------------------------------------------------------------------------
# LdG physics
# ---------------------------------------------------------------------------

@jit
def functional_derivative_LdG(
    Q: Array, a: float, b: float, K: float,
) -> Array:
    """Compute the functional derivative of the LdG free energy (JAX).

    Parameters
    ----------
    Q : jax.Array
        Q-tensor field of shape ``(L, L, 2, 2)``.
    a, b, K : float
        Material parameters.

    Returns
    -------
    jax.Array
        Functional derivative, same shape as *Q*.
    """
    Q_sq = jnp.matmul(Q, Q)
    tr_Q_sq = Q_sq[..., 0, 0] + Q_sq[..., 1, 1]
    answer = Q * (a + 0.5 * b * tr_Q_sq)[..., None, None]
    answer = answer - K * laplacian_2D_9_point_isotropic(Q)
    answer = make_traceless_symmetric(answer)
    return answer


@jit
def compute_free_energy(
    Q: Array, a: float, b: float, K: float,
) -> Array:
    """Compute the LdG free-energy density on every lattice site (JAX).

    Parameters
    ----------
    Q : jax.Array
        Q-tensor field of shape ``(L, L, 2, 2)``.
    a, b, K : float
        Material parameters.

    Returns
    -------
    jax.Array
        Scalar field of shape ``(L, L)``.
    """
    Q_sq = jnp.matmul(Q, Q)
    tr_Q_sq = Q_sq[..., 0, 0] + Q_sq[..., 1, 1]
    f_local = 0.5 * a * tr_Q_sq + 0.125 * b * tr_Q_sq**2

    def _grad_sq(arr: Array) -> Array:
        grad_x = (jnp.roll(arr, -1, axis=0) - jnp.roll(arr, 1, axis=0)) / 2.0
        grad_y = (jnp.roll(arr, -1, axis=1) - jnp.roll(arr, 1, axis=1)) / 2.0
        return grad_x**2 + grad_y**2

    f_grad = 0.0
    for i in range(2):
        for j in range(2):
            f_grad = f_grad + 0.5 * K * _grad_sq(Q[..., i, j])
    return f_local + f_grad


@jit
def model_A_LdG(
    Q: Array, a: float, b: float, K: float, dt: float, gamma: float,
) -> Array:
    """One explicit-Euler step of Model-A relaxational dynamics (JAX).

    Parameters
    ----------
    Q : jax.Array
        Current Q-tensor field.
    a, b, K : float
        Material parameters.
    dt : float
        Time step.
    gamma : float
        Kinetic coefficient.

    Returns
    -------
    jax.Array
        Updated Q-tensor field.
    """
    return Q - dt * gamma * functional_derivative_LdG(Q, a, b, K)


# ---------------------------------------------------------------------------
# Q  <-->  (n, S) conversions
# ---------------------------------------------------------------------------

@jit
def get_n_S_from_Q(Q: Array) -> tuple[Array, Array]:
    """Extract the director and scalar order parameter from Q (JAX).

    Parameters
    ----------
    Q : jax.Array
        Q-tensor field of shape ``(L, L, 2, 2)``.

    Returns
    -------
    n : jax.Array
        Director field of shape ``(L, L, 2)``.
    S : jax.Array
        Scalar order-parameter field of shape ``(L, L)``.
    """
    eigenvals, eigenvecs = jnp.linalg.eigh(Q)
    S = eigenvals[..., -1]
    n = eigenvecs[..., -1]
    return n, S


@jit
def get_Q_from_n_S(n: Array, S: Array) -> Array:
    """Construct a Q-tensor field from a director and order parameter (JAX).

    Parameters
    ----------
    n : jax.Array
        Director field of shape ``(L, L, 2)``.
    S : jax.Array or float
        Scalar order parameter (field or constant).

    Returns
    -------
    jax.Array
        Q-tensor field of shape ``(L, L, 2, 2)``.
    """
    outer = n[..., :, None] * n[..., None, :]
    I = jnp.eye(2)
    Q = 2 * S[..., None, None] * (outer - 0.5 * I)
    return Q


# ---------------------------------------------------------------------------
# Global diagnostics
# ---------------------------------------------------------------------------

@jit
def compute_global_order_parameter(Q: Array) -> Array:
    """Compute the global scalar order parameter (JAX).

    Parameters
    ----------
    Q : jax.Array
        Q-tensor field of shape ``(L, L, 2, 2)`` (may contain NaNs).

    Returns
    -------
    jax.Array
        Scalar: largest eigenvalue of the lattice-averaged Q-tensor.
    """
    Q_avg = jnp.nanmean(Q, axis=(0, 1))
    eigenvalues, _ = jnp.linalg.eigh(Q_avg)
    return eigenvalues[-1]


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------

def make_random_Q(L: int, S: float) -> Array:
    """Generate a random Q-tensor field with uniform random director angles.

    Parameters
    ----------
    L : int
        Side length of the square lattice.
    S : float
        Scalar order parameter used everywhere.

    Returns
    -------
    jax.Array
        Q-tensor field of shape ``(L, L, 2, 2)``.
    """
    key = random.PRNGKey(12345)
    _, subkey = random.split(key)
    theta = 2 * jnp.pi * random.uniform(subkey, (L, L))
    n0 = jnp.cos(theta)
    n1 = jnp.sin(theta)
    n = jnp.stack((n0, n1), axis=-1)

    I = jnp.eye(2)
    outer = n[..., :, None] * n[..., None, :]
    Q = 2 * S * (outer - 0.5 * I)
    return Q
