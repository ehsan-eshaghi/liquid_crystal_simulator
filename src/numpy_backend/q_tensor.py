import numpy as np

from src.numpy_backend.diff import  laplacian_2D_9_point_isotropic

rng = np.random.default_rng(12345)

def make_traceless_symmetric(arr):
    # symmetric
    tmp = 0.5 * (arr[...,0,1] + arr[...,1,0])
    arr[...,0,1] = arr[...,1,0] = tmp
    # traceless
    tmp = 0.5 * (arr[...,0,0] + arr[...,1,1])
    arr[...,0,0] -= tmp
    arr[...,1,1] -= tmp


def functional_derivative_LdG(Q, a,b, K):
    Q_sq = Q @ Q  # "@" is matrix multiplication
    tr_Q_sq = Q_sq[...,0,0] + Q_sq[...,1,1]  # Tr[Q^2]
    answer = np.empty_like(Q)
    for i in [0,1]:
        for j in [0,1]:
            answer[...,i,j] = (
                  Q[...,i,j] * (a + 0.5* b * tr_Q_sq)
            )
    answer -= K * laplacian_2D_9_point_isotropic(Q)
    make_traceless_symmetric(answer)
    return answer

def compute_free_energy(Q, a,b, K):
    Q_sq = Q @ Q
    tr_Q_sq = Q_sq[..., 0, 0] + Q_sq[..., 1, 1]
    f_local = 0.5 * a * tr_Q_sq + 0.125 * b * tr_Q_sq**2

    f_grad = np.zeros_like(f_local)
    for i in range(2):
        for j in range(2):
            grad_Q_ij = np.gradient(Q[..., i, j])
            grad_squared = grad_Q_ij[0]**2 + grad_Q_ij[1]**2
            f_grad += 0.5 * K * grad_squared

    f_total = f_local + f_grad
    return f_total

def model_A_LdG(Q, a,b, K, dt, gamma):
    return Q - dt * gamma * functional_derivative_LdG(Q, a,b, K)

#To plot the results, we'll need the degree of order $S$ and the director $\hat n$ as the greatest eigenvalue of $Q$ and its corresponding eigenvector.

def get_n_S_from_Q(Q):
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    S = eigenvals[..., -1]  # greatest eigenvalue
    n = eigenvecs[..., -1]  # corresponding eigenvector
    return n, S

def get_Q_from_n_S(n,S):
    L = S.shape[0]  # Assuming S and n are square arrays with shape (L, L)
    Q = np.zeros((L, L, 2, 2))
    for i in range(L):
        for j in range(L):
            ni = n[i, j]  # Director vector at site (i,j)
            outer = np.outer(ni, ni)  # Outer product of n with itself
            Q[i, j] = 2 * S[i, j] * (outer - 0.5 * np.eye(2))  # Apply the Q tensor formula
            
    return Q

def get_Q_from_n_S(n, S):
    # Compute the outer product for each vector. The new axes allow broadcasting
    # so that each element becomes a 2x2 matrix.
    outer = n[..., :, None] * n[..., None, :]
    
    # Create a 2x2 identity matrix
    I = np.eye(2)
    
    # Reshape S for broadcasting and compute the Q tensor
    Q = 2 * S[..., None, None] * (outer - 0.5 * I)
    return Q

def compute_global_order_parameter(Q):
    Q_avg = np.nanmean(Q, axis=(0, 1))
    eigenvalues, _ = np.linalg.eigh(Q_avg)
    global_order_parameter = eigenvalues[-1]
    return global_order_parameter

def make_random_Q(L,S):
    theta = 2 * np.pi * rng.random((L, L))  # random orientations
    n = np.empty((L, L, 2))
    n[..., 0] = np.cos(theta)
    n[..., 1] = np.sin(theta)

    Q = np.empty((L,L,2,2))
    for i in [0, 1]:
        for j in [0, 1]:
            Q[..., i,j]  = 2 * S * (n[...,i] * n[...,j] - 0.5 * (1 if i==j else 0))
    return Q
