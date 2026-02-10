"""JAX-specific boundary-condition helpers.

Only the JIT-compiled ``apply_mask`` lives here.  Mask *creation* and
director-field initialisation are in :mod:`src.common.boundary` (pure NumPy)
and should be converted with ``jnp.asarray()`` before use.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def apply_mask(
    not_masked_input: jax.Array,
    mask: jax.Array,
    masked_input: jax.Array,
) -> jax.Array:
    """Element-wise selection controlled by a binary mask (JAX / JIT).

    Where ``mask == 1`` the value from *masked_input* is used; elsewhere the
    value from *not_masked_input* is kept.

    Parameters
    ----------
    not_masked_input : jax.Array
        Values to keep where the mask is **0**.
    mask : jax.Array
        Binary integer mask.
    masked_input : jax.Array
        Values to insert where the mask is **1**.

    Returns
    -------
    jax.Array
        Combined array.
    """
    target_shape = jnp.broadcast_shapes(not_masked_input.shape, masked_input.shape)
    if mask.ndim < len(target_shape):
        new_axes = len(target_shape) - mask.ndim
        mask = mask[..., *(None,) * new_axes]
    return jnp.where(mask == 1, masked_input, not_masked_input)
