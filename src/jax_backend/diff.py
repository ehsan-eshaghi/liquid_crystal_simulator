import jax.numpy as jnp
from jax import jit

@jit
def laplacian_2D_9_point_isotropic(c, h=1.0):
    lap = (
          jnp.roll(c, -1, axis=0) 
        + jnp.roll(c,  1, axis=0)
        + jnp.roll(c, -1, axis=1)
        + jnp.roll(c,  1, axis=1)
        + jnp.roll(jnp.roll(c, -1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(c, -1, axis=0),  1, axis=1)
        + jnp.roll(jnp.roll(c,  1, axis=0), -1, axis=1)
        + jnp.roll(jnp.roll(c,  1, axis=0),  1, axis=1)
        - 8 * c
    ) / (3 * h**2)
    return lap