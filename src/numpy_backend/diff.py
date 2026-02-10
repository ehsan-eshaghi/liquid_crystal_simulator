"""Finite-difference operators for the NumPy backend."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def laplacian_2D(c: NDArray) -> NDArray:
    """Standard 5-point Laplacian stencil on a periodic 2D grid.

    Parameters
    ----------
    c : NDArray
        Scalar or tensor field with spatial dimensions along axes 0 and 1.

    Returns
    -------
    NDArray
        Discrete Laplacian of *c*.
    """
    return (
        np.roll(c, -1, axis=0)
        + np.roll(c, 1, axis=0)
        + np.roll(c, -1, axis=1)
        + np.roll(c, 1, axis=1)
        - 4 * c
    )


def laplacian_2D_9_point_isotropic(c: NDArray, h: float = 1.0) -> NDArray:
    """Isotropic 9-point Laplacian stencil on a periodic 2D grid.

    Parameters
    ----------
    c : NDArray
        Scalar or tensor field with spatial dimensions along axes 0 and 1.
    h : float
        Grid spacing (default ``1.0``).

    Returns
    -------
    NDArray
        Discrete Laplacian of *c*.
    """
    return (
        np.roll(c, -1, axis=0)
        + np.roll(c, 1, axis=0)
        + np.roll(c, -1, axis=1)
        + np.roll(c, 1, axis=1)
        + np.roll(np.roll(c, -1, axis=0), -1, axis=1)
        + np.roll(np.roll(c, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(c, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(c, 1, axis=0), 1, axis=1)
        - 8 * c
    ) / (3 * h**2)


def diffuse_2D(c: NDArray, D: float, dt: float) -> NDArray:
    """One explicit-Euler diffusion step.

    Parameters
    ----------
    c : NDArray
        Concentration / field to diffuse.
    D : float
        Diffusion coefficient.
    dt : float
        Time step.

    Returns
    -------
    NDArray
        Field after one diffusion step.
    """
    return c + dt * D * laplacian_2D_9_point_isotropic(c)
