"""Finite-difference operators for the JAX backend."""
from __future__ import annotations

import jax.numpy as jnp
from jax import jit, Array


@jit
def laplacian_2D_9_point_isotropic(c: Array, h: float = 1.0) -> Array:
    """Isotropic 9-point Laplacian stencil on a periodic 2D grid (JAX).

    Parameters
    ----------
    c : jax.Array
        Scalar or tensor field with spatial dimensions along axes 0 and 1.
    h : float
        Grid spacing (default ``1.0``).

    Returns
    -------
    jax.Array
        Discrete Laplacian of *c*.
    """
    return (
        jnp.roll(c, -1, axis=0)
        + jnp.roll(c, 1, axis=0)
        + jnp.roll(c, -1, axis=1)
        + jnp.roll(c, 1, axis=1)
        + jnp.roll(jnp.roll(c, -1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(c, -1, axis=0), 1, axis=1)
        + jnp.roll(jnp.roll(c, 1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(c, 1, axis=0), 1, axis=1)
        - 8 * c
    ) / (3 * h**2)
