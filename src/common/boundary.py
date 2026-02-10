"""Shared boundary-condition helpers (pure NumPy).

These functions create masks and initialise director fields on the lattice.
They are backend-agnostic: JAX arrays can be constructed from the returned
NumPy arrays with ``jnp.asarray()``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def create_circle_bc_mask(radius: int, L: int) -> NDArray[np.int_]:
    """Create a binary mask selecting the boundary ring of a centred circle.

    Parameters
    ----------
    radius : int
        Radius of the circle (in lattice units).
    L : int
        Side length of the square lattice.

    Returns
    -------
    NDArray
        Integer array of shape ``(L, L)`` with ``1`` on the boundary ring and
        ``0`` elsewhere.
    """
    x = np.linspace(-1, 1, L) * L / 2
    y = np.linspace(-1, 1, L) * L / 2
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    mask = (distance >= radius - 1) & (distance <= radius)
    return mask.astype(int)


def create_circle_lattice_mask(radius: int, L: int) -> NDArray[np.int_]:
    """Create a binary mask selecting sites *outside* a centred circle.

    Parameters
    ----------
    radius : int
        Radius of the circle (in lattice units).
    L : int
        Side length of the square lattice.

    Returns
    -------
    NDArray
        Integer array of shape ``(L, L)`` with ``1`` outside the circle and
        ``0`` inside.
    """
    x = np.linspace(-1, 1, L) * L / 2
    y = np.linspace(-1, 1, L) * L / 2
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    mask = distance > radius
    return mask.astype(int)


# ---------------------------------------------------------------------------
# Director-field initialisation
# ---------------------------------------------------------------------------

def initialize_radial_n(L: int) -> NDArray[np.floating]:
    """Return a radial nematic director field on an ``L x L`` lattice.

    Each vector points away from the centre of the domain.

    Parameters
    ----------
    L : int
        Side length of the square lattice.

    Returns
    -------
    NDArray
        Array of shape ``(L, L, 2)`` with unit directors.
    """
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)

    n = np.zeros((L, L, 2))
    n[..., 0] = np.cos(theta)
    n[..., 1] = np.sin(theta)
    return n


def initialize_tangential_n(L: int) -> NDArray[np.floating]:
    """Return a tangential nematic director field on an ``L x L`` lattice.

    Each vector is perpendicular to the radial direction (shifted by pi/2).

    Parameters
    ----------
    L : int
        Side length of the square lattice.

    Returns
    -------
    NDArray
        Array of shape ``(L, L, 2)`` with unit directors.
    """
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    X, Y = np.meshgrid(x, y)
    theta = np.pi / 2 + np.arctan2(Y, X)

    n = np.zeros((L, L, 2))
    n[..., 0] = np.cos(theta)
    n[..., 1] = np.sin(theta)
    return n


# ---------------------------------------------------------------------------
# Mask application (NumPy)
# ---------------------------------------------------------------------------

def apply_mask(
    not_masked_input: NDArray,
    mask: NDArray[np.int_],
    masked_input: NDArray,
) -> NDArray:
    """Element-wise selection controlled by a binary mask.

    Where ``mask == 1`` the value from *masked_input* is used; elsewhere the
    value from *not_masked_input* is kept.  Broadcasting is handled
    automatically.

    Parameters
    ----------
    not_masked_input : NDArray
        Values to keep where the mask is **0**.
    mask : NDArray
        Binary integer mask.
    masked_input : NDArray
        Values to insert where the mask is **1**.

    Returns
    -------
    NDArray
        Combined array.
    """
    target_shape = np.broadcast(not_masked_input, masked_input).shape
    if mask.ndim < len(target_shape):
        new_axes = len(target_shape) - mask.ndim
        mask = mask[..., *(None,) * new_axes]
    return np.where(mask == 1, masked_input, not_masked_input)


def apply_no_flux_boundary_conditions(Q: NDArray) -> NDArray:
    """Apply no-flux (Neumann) boundary conditions to a Q-tensor field.

    Parameters
    ----------
    Q : NDArray
        Q-tensor field of shape ``(L, L, 2, 2)``.

    Returns
    -------
    NDArray
        Updated Q-tensor field (modified in-place and returned).
    """
    Q[0, :, :, :] = Q[1, :, :, :]
    Q[-1, :, :, :] = Q[-2, :, :, :]
    Q[:, 0, :, :] = Q[:, 1, :, :]
    Q[:, -1, :, :] = Q[:, -2, :, :]
    return Q
