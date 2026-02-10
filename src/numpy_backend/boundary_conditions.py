import numpy as np

def apply_no_flux_boundary_conditions(Q):
    """
    Applies no-flux (Neumann) boundary conditions to the Q-tensor field.

    Parameters:
    - Q: numpy array of shape (L, L, 2, 2), the Q-tensor field.

    Returns:
    - Q: updated Q-tensor field with no-flux boundary conditions.
    """
    L = Q.shape[0]

    # Top and bottom boundaries
    Q[0, :, :, :] = Q[1, :, :, :]       # Top row matches the second row
    Q[-1, :, :, :] = Q[-2, :, :, :]     # Bottom row matches the second-to-last row

    # Left and right boundaries
    Q[:, 0, :, :] = Q[:, 1, :, :]       # Left column matches the second column
    Q[:, -1, :, :] = Q[:, -2, :, :]     # Right column matches the second-to-last column

    return Q

def apply_mask(not_masked_input, mask, masked_input):
    target_shape = np.broadcast(not_masked_input, masked_input).shape
    if mask.ndim < len(target_shape):
        # Calculate how many new axes to add
        new_axes = len(target_shape) - mask.ndim
        mask = mask[..., *(None,)*new_axes]
    result = np.where(mask == 1, masked_input, not_masked_input)
    return result



def create_circle_bc_mask(radius, L):
    # Generate grid of coordinates
    x = np.linspace(-1, 1, L)*L/2
    y = np.linspace(-1, 1, L)*L/2
    X, Y = np.meshgrid(x, y)

    # Calculate the distance from the center
    distance_from_center = np.sqrt(X**2 + Y**2)

    # Create a binary bc_mask where values inside the circle are 1
    bc_mask = (distance_from_center >= radius - 1) & (distance_from_center <= radius)

    return bc_mask.astype(int)  # Ensure bc_mask is integer (0 or 1)

def create_circle_lattice_mask(radius, L):
    # Generate grid of coordinates
    x = np.linspace(-1, 1, L)*L/2
    y = np.linspace(-1, 1, L)*L/2
    X, Y = np.meshgrid(x, y)

    # Calculate the distance from the center
    distance_from_center = np.sqrt(X**2 + Y**2)

    lattice_mask = (distance_from_center > radius)

    return lattice_mask.astype(int)  # Ensure bc_mask is integer (0 or 1)


def initialize_radial_n(L):
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)  # Angle of alignment (radial)

    n = np.zeros((L, L, 2))  # Nematic director field
    n[..., 0] = np.cos(theta)  # n_x
    n[..., 1] = np.sin(theta)  # n_y
    return n

def initialize_tangential_n(L):
    x = np.linspace(-1, 1, L)
    y = np.linspace(-1, 1, L)
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)  # Angle of alignment (radial)
    theta = np.pi / 2 + np.arctan2(Y, X)  # Shift by pi/2 to make the director tangential


    n = np.zeros((L, L, 2))  # Nematic director field
    n[..., 0] = np.cos(theta)  # n_x
    n[..., 1] = np.sin(theta)  # n_y
    return n