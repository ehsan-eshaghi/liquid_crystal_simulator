import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

# Replace the NumPy RNG with JAX's PRNGKey.
import jax.random as random

from src.jax_backend.diff import laplacian_2D_9_point_isotropic

@jit
def make_traceless_symmetric(arr):
    # Force symmetry
    tmp = 0.5 * (arr[..., 0, 1] + arr[..., 1, 0])
    arr = arr.at[..., 0, 1].set(tmp)
    arr = arr.at[..., 1, 0].set(tmp)
    # Make traceless
    trace = 0.5 * (arr[..., 0, 0] + arr[..., 1, 1])
    arr = arr.at[..., 0, 0].add(-trace)
    arr = arr.at[..., 1, 1].add(-trace)
    return arr



@jit
def functional_derivative_LdG(Q, a, b, K):
    Q_sq = jnp.matmul(Q, Q)
    tr_Q_sq = Q_sq[..., 0, 0] + Q_sq[..., 1, 1]
    # Broadcast the scalar factor to the matrix
    answer = Q * (a + 0.5 * b * tr_Q_sq)[..., None, None]
    answer = answer - K * laplacian_2D_9_point_isotropic(Q)
    answer = make_traceless_symmetric(answer)
    return answer

@jit
def compute_free_energy(Q, a, b, K):
    Q_sq = jnp.matmul(Q, Q)
    tr_Q_sq = Q_sq[..., 0, 0] + Q_sq[..., 1, 1]
    f_local = 0.5 * a * tr_Q_sq + 0.125 * b * tr_Q_sq**2

    # Instead of np.gradient, we use central finite differences.
    def grad_sq(arr):
        grad_x = (jnp.roll(arr, -1, axis=0) - jnp.roll(arr, 1, axis=0)) / 2.0
        grad_y = (jnp.roll(arr, -1, axis=1) - jnp.roll(arr, 1, axis=1)) / 2.0
        return grad_x**2 + grad_y**2

    f_grad = 0.0
    for i in range(2):
        for j in range(2):
            f_grad += 0.5 * K * grad_sq(Q[..., i, j])
    return f_local + f_grad

@jit
def model_A_LdG(Q, a, b, K, dt, gamma):
    return Q - dt * gamma * functional_derivative_LdG(Q, a, b, K)

@jit
def get_n_S_from_Q(Q):
    eigenvals, eigenvecs = jnp.linalg.eigh(Q)
    S = eigenvals[..., -1]      # greatest eigenvalue
    n = eigenvecs[..., -1]      # corresponding eigenvector
    return n, S

@jit
def get_Q_from_n_S(n, S):
    # Compute outer product for each vector
    outer = n[..., :, None] * n[..., None, :]
    I = jnp.eye(2)
    Q = 2 * S[..., None, None] * (outer - 0.5 * I)
    return Q

def make_random_Q(L, S):
    key = random.PRNGKey(12345)
    # Generate random angles uniformly in [0, 2*pi)
    key, subkey = random.split(key)
    theta = 2 * jnp.pi * random.uniform(subkey, (L, L))
    n0 = jnp.cos(theta)
    n1 = jnp.sin(theta)
    n = jnp.stack((n0, n1), axis=-1)  # shape: (L, L, 2)
    
    # Build Q-tensor field from director field
    I = jnp.eye(2)
    outer = n[..., :, None] * n[..., None, :]
    Q = 2 * S * (outer - 0.5 * I)
    return Q

@jit
def compute_global_order_parameter(Q):
    Q_avg = jnp.nanmean(Q, axis=(0, 1))  # Compute the average Q-tensor over the lattice
    eigenvalues, _ = jnp.linalg.eigh(Q_avg)  # Compute the eigenvalues of the average Q-tensor
    order_parameter = eigenvalues[-1]  # The largest eigenvalue represents the order parameter
    return order_parameter