import numpy as np

def laplacian_2D(c):
    return (
          np.roll(c, -1, axis=0)  # look 1 ahead in x
        + np.roll(c,  1, axis=0)  # look 1 behind in x
        + np.roll(c, -1, axis=1)  # look 1 ahead in y
        + np.roll(c,  1, axis=1)  # look 1 behind in y
        - 4 * c
    )

def laplacian_2D_9_point_isotropic(c, h=1.0):
    """
    Compute the Laplacian of a 2D array using the isotropic 9-point stencil.
    
    Parameters:
        c (ndarray): 2D array representing the scalar field.
        h (float): Grid spacing (default is 1.0).
    
    Returns:
        ndarray: The Laplacian of the input array.
    """
    laplacian = (
          np.roll(c, -1, axis=0)  # look 1 ahead in x
        + np.roll(c,  1, axis=0)  # look 1 behind in x
        + np.roll(c, -1, axis=1)  # look 1 ahead in y
        + np.roll(c,  1, axis=1)  # look 1 behind in y
        + np.roll(np.roll(c, -1, axis=0), -1, axis=1)  # top-right diagonal
        + np.roll(np.roll(c, -1, axis=0),  1, axis=1)  # top-left diagonal
        + np.roll(np.roll(c,  1, axis=0), -1, axis=1)  # bottom-right diagonal
        + np.roll(np.roll(c,  1, axis=0),  1, axis=1)  # bottom-left diagonal
        - 8 * c  # center point
    ) / (3 * h**2)
    
    return laplacian


def diffuse_2D(c, D, dt):
    return c + dt * D * laplacian_2D_9_point_isotropic(c)