from jax import jit
import jax
import jax.numpy as jnp

@jax.jit
def apply_mask(not_masked_input, mask, masked_input):
    # Get broadcasted shape
    target_shape = jnp.broadcast_shapes(not_masked_input.shape, masked_input.shape)
    
    # Expand mask dimensions if needed to match broadcasted shape
    if mask.ndim < len(target_shape):
        new_axes = len(target_shape) - mask.ndim
        mask = mask[..., *(None,) * new_axes]
    
    # Apply the mask
    result = jnp.where(mask == 1, masked_input, not_masked_input)
    return result


@jax.jit
def create_circle_mask(radius, L):
    """
    Create a binary bc_mask of a circle centered at the lattice with a given radius.

    Parameters:
    - radius: float, the radius of the circle.
    - L: int, the lattice dimension (L x L).

    Returns:
    - bc_mask: jax.numpy array of shape (L, L), binary bc_mask with a circle of radius `r` at the center.
    """
    # Generate grid of coordinates (scaled from -1 to 1)
    x = jnp.linspace(-1, 1, L) * L / 2
    y = jnp.linspace(-1, 1, L) * L / 2
    X, Y = jnp.meshgrid(x, y)

    # Calculate the distance from the center
    distance_from_center = jnp.sqrt(X**2 + Y**2)

    # Create a binary bc_mask where values inside the circle are 1, others are 0
    bc_mask = (distance_from_center >= radius - 1) & (distance_from_center <= radius)

    return bc_mask.astype(jnp.int32)  # Ensure bc_mask is integer (0 or 1)


@jax.jit
def initialize_radial_n(L):
    """
    Initializes a radial nematic director field with unit vectors pointing radially outward.

    Parameters:
    - L: int, the lattice dimension (L x L).

    Returns:
    - n: jax.numpy array of shape (L, L, 2), the radial director field.
    """
    x = jnp.linspace(-1, 1, L)
    y = jnp.linspace(-1, 1, L)
    X, Y = jnp.meshgrid(x, y)
    theta = jnp.arctan2(Y, X)  # Angle of alignment (radial)

    n = jnp.zeros((L, L, 2))  # Nematic director field
    n = n.at[..., 0].set(jnp.cos(theta))  # n_x
    n = n.at[..., 1].set(jnp.sin(theta))  # n_y

    return n

@jax.jit
def initialize_tangential_n(L):
    """
    Initializes a tangential nematic director field with unit vectors tangential to the radius.

    Parameters:
    - L: int, the lattice dimension (L x L).

    Returns:
    - n: jax.numpy array of shape (L, L, 2), the tangential director field.
    """
    x = jnp.linspace(-1, 1, L)
    y = jnp.linspace(-1, 1, L)
    X, Y = jnp.meshgrid(x, y)
    theta = jnp.arctan2(Y, X)  # Angle of alignment (radial)
    theta = jnp.pi / 2 + jnp.arctan2(Y, X)  # Shift by pi/2 to make the director tangential

    n = jnp.zeros((L, L, 2))  # Nematic director field
    n = n.at[..., 0].set(jnp.cos(theta))  # n_x
    n = n.at[..., 1].set(jnp.sin(theta))  # n_y

    return n