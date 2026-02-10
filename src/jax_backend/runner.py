"""JAX-backend simulation runner.

Supports two integrators:

* **euler** -- explicit forward-Euler (fixed step).
* **dopri5** -- adaptive Dormand-Prince via *diffrax*.

Both integrate Model-A relaxational dynamics of the Landau-de Gennes
Q-tensor on a 2D lattice with circular boundary conditions.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp

try:
    import diffrax  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    diffrax = None

from src.config import AppConfig
from src.common.boundary import (
    create_circle_bc_mask,
    create_circle_lattice_mask,
    initialize_tangential_n,
)
from src.jax_backend.boundary import apply_mask
from src.jax_backend.q_tensor import (
    compute_free_energy,
    compute_global_order_parameter,
    functional_derivative_LdG,
    get_n_S_from_Q,
    get_Q_from_n_S,
    make_random_Q,
    model_A_LdG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    params: tuple[float, float, float],
    config: AppConfig,
    integrator: str,
) -> tuple[list[dict[str, Any]], float]:
    """Run a single JAX-backend simulation.

    Parameters
    ----------
    params : tuple
        Material parameters ``(a, K, S)``.
    config : AppConfig
        Parsed application configuration.
    integrator : str
        One of ``"euler"``, ``"forward_euler"``, ``"dopri5"``,
        ``"dormand_prince"``, or ``"rk45"``.

    Returns
    -------
    simulation_data : list[dict]
        Snapshot records collected at the sample rate.
    runtime : float
        Wall-clock time in seconds.
    """
    # -- Unpack config -----------------------------------------------------
    a, K, S = params
    sim = config.simulation
    L = sim.L
    radius = L // 2
    b = sim.b
    gamma = sim.gamma
    t_final = sim.t_final
    sample_rate = sim.sample_rate
    update_freq = sim.update_freq

    dt_euler = config.fixed_step.dt
    dt0 = config.adaptive_step.dt0
    rtol = config.adaptive_step.rtol
    atol = config.adaptive_step.atol

    simulation_id = f"a_{a}_b_{b}_K_{K}_r_{radius}_S_{S}"
    logger.info("Starting simulation: %s (integrator=%s)", simulation_id, integrator)

    # -- Initialise fields -------------------------------------------------
    Q_init = make_random_Q(L, S)
    tangential_n = initialize_tangential_n(L)

    bc = get_Q_from_n_S(jnp.asarray(tangential_n), jnp.asarray(S))
    bc_mask = jnp.asarray(create_circle_bc_mask(radius, L))
    lattice_mask = jnp.asarray(create_circle_lattice_mask(radius, L))

    Q_init = apply_mask(Q_init, bc_mask, bc)

    # -- Snapshot recorder -------------------------------------------------
    simulation_data: list[dict[str, Any]] = []

    def _record_state(Q_snap: jax.Array, t_snap: float) -> None:
        """Compute diagnostics and append a snapshot to *simulation_data*."""
        fe = compute_free_energy(Q_snap, a, b, K)
        fd = functional_derivative_LdG(Q_snap, a, b, K)

        nan_fe = jnp.full_like(fe, jnp.nan, dtype=jnp.float32)
        nan_fd = jnp.full_like(fd, jnp.nan, dtype=jnp.float32)
        nan_Q = jnp.full_like(Q_snap, jnp.nan, dtype=jnp.float32)

        fe_masked = apply_mask(fe, lattice_mask, nan_fe)
        fd_masked = apply_mask(fd, lattice_mask, nan_fd)
        Q_masked = apply_mask(Q_snap, lattice_mask, nan_Q)

        gop = float(compute_global_order_parameter(Q_masked))
        n, S_field = get_n_S_from_Q(Q_masked)
        n_masked = apply_mask(
            n, lattice_mask, jnp.full_like(n, jnp.nan, dtype=jnp.float32),
        )
        S_masked = apply_mask(
            S_field, lattice_mask, jnp.full_like(S_field, jnp.nan, dtype=jnp.float32),
        )

        simulation_data.append({
            "id": simulation_id,
            "fe": fe_masked,
            "fe_mean": float(jnp.nanmean(fe_masked)),
            "fd_norm": float(jnp.linalg.norm(fd_masked[~jnp.isnan(fd_masked)])),
            "gop": gop,
            "Q": Q_masked,
            "bc": bc,
            "bc_mask": bc_mask,
            "A": a,
            "B": b,
            "K": K,
            "t": float(t_snap),
            "r": radius,
            "n": n_masked,
            "S": S_masked,
        })

    # -- Run dynamics ------------------------------------------------------
    start_time = time.time()

    if integrator in {"euler", "forward_euler"}:
        Q = Q_init
        t = 0.0
        while t < t_final:
            Q = model_A_LdG(Q, a, b, K, dt_euler, gamma)
            Q = apply_mask(Q, bc_mask, bc)
            t += dt_euler
            if t % sample_rate == 0:
                _record_state(Q, t)

    elif integrator in {"dopri5", "dormand_prince", "rk45"}:
        if diffrax is None:
            raise ImportError(
                "Integrator 'dopri5' selected but diffrax is not installed.  "
                "Install with:  pip install diffrax"
            )

        def rhs(t: float, Q_state: jax.Array, _args: None) -> jax.Array:
            """RHS of dQ/dt = -gamma * dF/dQ with zero rate on boundary."""
            fd = functional_derivative_LdG(Q_state, a, b, K)
            dQdt = -gamma * fd
            return apply_mask(dQdt, bc_mask, jnp.zeros_like(dQdt))

        term = diffrax.ODETerm(rhs)
        solver = diffrax.Dopri5()
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
        save_ts = jnp.arange(0, t_final + sample_rate, sample_rate)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=float(t_final),
            dt0=dt0,
            y0=Q_init,
            saveat=diffrax.SaveAt(ts=save_ts),
            stepsize_controller=controller,
            max_steps=10_000,
        )

        # Skip the initial state at t = 0
        for t_snap, Q_snap in zip(sol.ts[1:], sol.ys[1:]):
            _record_state(Q_snap, float(t_snap))

    else:
        raise ValueError(f"Unknown integrator '{integrator}'.")

    runtime = time.time() - start_time
    logger.info("Completed simulation: %s in %.2f s", simulation_id, runtime)
    return simulation_data, runtime
