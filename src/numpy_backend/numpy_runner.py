import numpy as np
import pandas as pd
from itertools import product
import time
import configparser

# Import functions from your simulation modules
from src.numpy_backend.boundary_conditions import apply_mask
from src.numpy_backend.q_tensor import (
    compute_global_order_parameter,
    model_A_LdG,
    get_Q_from_n_S,
    functional_derivative_LdG,
    compute_free_energy,
)
from src.numpy_backend.boundary_conditions import (
    create_circle_bc_mask,
    create_circle_lattice_mask,
    initialize_tangential_n,
)
from src.numpy_backend.q_tensor import make_random_Q, get_n_S_from_Q



# Simulation runner
def run_simulation(params,config,integrator):
    a, K, S = params
    # Simulation constants
    L = config.getint('simulation', 'L')
    radius = int(L / 2)
    dt = config.getfloat('fixed_step_solver', 'dt')
    b = config.getfloat('simulation', 'b')
    gamma = config.getfloat('simulation', 'gamma')
    t_final = config.getint('simulation', 't_final')
    sample_rate = config.getint('simulation', 'sample_rate')
    update_freq = config.getint('simulation', 'update_freq')
    
    simulation_id = f"a_{a}_b_{b}_K_{K}_r_{radius}_S_{S}"
    print(f"Starting simulation: {simulation_id}")
    Q = make_random_Q(L, S)
    tangential_n = initialize_tangential_n(L)
    bc = get_Q_from_n_S(tangential_n, S)
    bc_mask = create_circle_bc_mask(radius, L)
    lattice_mask = create_circle_lattice_mask(radius, L)
    t = 0

    Q = apply_mask(Q, bc_mask, bc)
    
    simulation_data = []
    start_time = time.time()
    while t < t_final:
        for _ in range(update_freq):
            Q = model_A_LdG(Q, a, b, K, dt, gamma)
            Q = apply_mask(Q, bc_mask, bc)
            t += dt

        fe = compute_free_energy(Q, a, b, K)
        fd = functional_derivative_LdG(Q, a, b, K)
        fe = apply_mask(fe, lattice_mask, np.full_like(fe, np.nan, dtype=np.float32))
        fd = apply_mask(fd, lattice_mask, np.full_like(fd, np.nan, dtype=np.float32))
        Q_masked = apply_mask(Q, lattice_mask, np.full_like(Q, np.nan, dtype=np.float32))
        gop = compute_global_order_parameter(Q_masked)

        if t % sample_rate == 0:
            n, S = get_n_S_from_Q(Q_masked)
            simulation_data.append({
                "id": simulation_id,
                "fe": fe,
                "fe_mean": np.nanmean(fe),
                "fd_norm": np.linalg.norm(fd[~np.isnan(fd)]),
                "gop": gop,
                "Q": Q_masked,
                "bc": bc,
                "bc_mask": bc_mask,
                "A": a,
                "B": b,
                "K": K,
                "t": t,
                "r": radius,
                "n": n,
                "S": S
            })

    runtime = time.time() - start_time
    print(f"Completed simulation: {simulation_id} in {runtime:.2f} seconds")
    return simulation_data, runtime