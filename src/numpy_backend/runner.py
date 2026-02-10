"""NumPy-backend simulation runner.

Runs an explicit-Euler integration of Model-A relaxational dynamics for the
Landau-de Gennes Q-tensor on a 2D lattice with circular boundary conditions.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.config import AppConfig
from src.common.boundary import (
    apply_mask,
    create_circle_bc_mask,
    create_circle_lattice_mask,
    initialize_tangential_n,
)
from src.numpy_backend.q_tensor import (
    compute_free_energy,
    compute_global_order_parameter,
    functional_derivative_LdG,
    get_n_S_from_Q,
    get_Q_from_n_S,
    make_random_Q,
    model_A_LdG,
)

logger = logging.getLogger(__name__)


def run_simulation(
    params: tuple[float, float, float],
    config: AppConfig,
    integrator: str,
) -> tuple[list[dict[str, Any]], float]:
    """Run a single NumPy-backend simulation.

    Parameters
    ----------
    params : tuple
        Material parameters ``(a, K, S)``.
    config : AppConfig
        Parsed application configuration.
    integrator : str
        Integrator name (only ``"euler"`` is supported for the NumPy backend).

    Returns
    -------
    simulation_data : list[dict]
        Snapshot records collected at the sample rate.
    runtime : float
        Wall-clock time in seconds.
    """
    a, K, S = params
    sim = config.simulation
    L = sim.L
    radius = L // 2
    dt = config.fixed_step.dt
    b = sim.b
    gamma = sim.gamma
    t_final = sim.t_final
    sample_rate = sim.sample_rate
    update_freq = sim.update_freq

    simulation_id = f"a_{a}_b_{b}_K_{K}_r_{radius}_S_{S}"
    logger.info("Starting simulation: %s", simulation_id)

    # Initialise fields
    Q = make_random_Q(L, S)
    tangential_n = initialize_tangential_n(L)
    bc = get_Q_from_n_S(tangential_n, S)
    bc_mask = create_circle_bc_mask(radius, L)
    lattice_mask = create_circle_lattice_mask(radius, L)

    Q = apply_mask(Q, bc_mask, bc)

    simulation_data: list[dict[str, Any]] = []
    t = 0.0
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
            n, S_field = get_n_S_from_Q(Q_masked)
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
                "S": S_field,
            })

    runtime = time.time() - start_time
    logger.info("Completed simulation: %s in %.2f seconds", simulation_id, runtime)
    return simulation_data, runtime
