# SPDX-License-Identifier: MIT
"""JAX backend runner with two integrators: explicit Euler and adaptive Dormand‑Prince (Dopri5).

The integrator is chosen in *config.ini*:

[simulation]
integrator = euler          # or dopri5 / dormand_prince / rk45
# Euler params
dt          = 1e-3
update_freq = 10
# Dopri5 params
dt0         = 1e-2          # initial step size
rtol        = 1e-5
atol        = 1e-7

The rest of the interface stays identical to the previous version so nothing
outside this file has to change.
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

try:
    import diffrax  # type: ignore
except ImportError:  # pragma: no cover – handled at runtime
    diffrax = None

# › Project imports
from src.jax_backend.boundary_conditions import apply_mask
from src.jax_backend.q_tensor import (
    compute_global_order_parameter,
    model_A_LdG,
    functional_derivative_LdG,
    compute_free_energy,
    get_Q_from_n_S,
)
from src.numpy_backend.boundary_conditions import (
    create_circle_bc_mask,
    create_circle_lattice_mask,
    initialize_tangential_n,
)
from src.jax_backend.q_tensor import make_random_Q,get_n_S_from_Q# -----------------------------------------------------------------------------
# 🏃‍♂️  Main entry point
# -----------------------------------------------------------------------------

def run_simulation(params, config,integrator):
    """Run a single simulation and return a list of records and runtime."""

    # -------------------- 1️⃣  Unpack config --------------------
    a, K, S = params

    L = config.getint("simulation", "L")
    radius = int(L / 2)

    # Common coefficients
    b = config.getfloat("simulation", "b")
    gamma = config.getfloat("simulation", "gamma")
    t_final = config.getint("simulation", "t_final")
    sample_rate = config.getint("simulation", "sample_rate")
    update_freq = config.getint("simulation", "update_freq")

    # Euler‑specific
    dt_euler = config.getfloat("fixed_step_solver", "dt")

    # Dopri‑specific
    dt0 = config.getfloat("adaptive_step_solver", "dt0", fallback=1e-2)
    rtol = config.getfloat("adaptive_step_solver", "rtol", fallback=1e-5)
    atol = config.getfloat("adaptive_step_solver", "atol", fallback=1e-5)

    # Identify the run
    simulation_id = f"a_{a}_b_{b}_K_{K}_r_{radius}_S_{S}"
    print(f"Starting simulation: {simulation_id} (integrator={integrator})")

    # -------------------- 2️⃣  Initialise fields --------------------
    Q_init = make_random_Q(L, S)
    tangential_n = initialize_tangential_n(L)

    bc = get_Q_from_n_S(tangential_n, S)  # (L,L,2,2)
    bc_mask = create_circle_bc_mask(radius, L)  # (L,L)
    lattice_mask = create_circle_lattice_mask(radius, L)  # (L,L)

    # Cast to JAX arrays
    Q_init = jnp.asarray(Q_init)
    bc = jnp.asarray(bc)
    bc_mask = jnp.asarray(bc_mask)
    lattice_mask = jnp.asarray(lattice_mask)

    # Apply boundary conditions to the initial state
    Q_init = apply_mask(Q_init, bc_mask, bc)

    # Container for snapshots
    simulation_data: list[dict] = []

    # -------------------- 3️⃣  Helper to record state --------------------
    def _record_state(Q_snap: jax.Array, t_snap: float):
        """Add one entry to `simulation_data` for the given snapshot."""
        fe = compute_free_energy(Q_snap, a, b, K)         # (L,L)
        fd = functional_derivative_LdG(Q_snap, a, b, K)    # (L,L,2,2)

        fe_masked = apply_mask(
            fe,
            lattice_mask,
            jnp.full_like(fe, jnp.nan, dtype=jnp.float32),
        )
        fd_masked = apply_mask(
            fd,
            lattice_mask,
            jnp.full_like(fd, jnp.nan, dtype=jnp.float32),
        )
        Q_masked = apply_mask(
            Q_snap,
            lattice_mask,
            jnp.full_like(Q_snap, jnp.nan, dtype=jnp.float32),
        )

        gop = float(compute_global_order_parameter(Q_masked))
        n, S_field = get_n_S_from_Q(Q_masked)
        n_masked = apply_mask(
            n,
            lattice_mask,
            jnp.full_like(n, jnp.nan, dtype=jnp.float32),
        )
        S_masked = apply_mask(
            S_field,
            lattice_mask,
            jnp.full_like(S_field, jnp.nan, dtype=jnp.float32),
        )

        simulation_data.append(
            dict(
                id=simulation_id,
                fe=fe_masked,
                fe_mean=float(jnp.nanmean(fe_masked)),
                fd_norm=float(jnp.linalg.norm(fd_masked[~jnp.isnan(fd_masked)])),
                gop=gop,
                Q=Q_masked,
                bc=bc,
                bc_mask=bc_mask,
                A=a,
                B=b,
                K=K,
                t=float(t_snap),
                r=radius,
                n=n_masked,
                S=S_masked,
            )
        )

    # ------------------------------------------------------------------
    # 4️⃣  Run dynamics
    # ------------------------------------------------------------------
    start_time = time.time()

    if integrator in {"euler", "forward_euler"}:
        # ✏️ Explicit Euler (legacy)
        Q = Q_init
        t = 0.0
        while t < t_final:
            Q = model_A_LdG(Q, a, b, K, dt_euler, gamma)
            Q = apply_mask(Q, bc_mask, bc)  # keep boundary fixed
            t += dt_euler
            if t % sample_rate == 0:
                _record_state(Q, t)

    elif integrator in {"dopri5", "dormand_prince", "rk45"}:
        # 🛩️  Adaptive Dormand–Prince via Diffrax
        if diffrax is None:
            raise ImportError(
                "Integrator 'dopri5' selected but diffrax is not installed. "
                "Install with 'pip install diffrax'.",
            )

        def rhs(t, Q_state, _args):  # noqa: D401 – diffrax signature
            """dQ/dt = -γ δF/δQ with zero rate on boundary."""
            fd = functional_derivative_LdG(Q_state, a, b, K)
            dQdt = -gamma * fd
            # Zero derivative on boundary nodes
            dQdt = apply_mask(dQdt, bc_mask, jnp.zeros_like(dQdt))
            return dQdt

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
            max_steps=10000
        )

        # Skip the initial state at t = 0 (already masked above)
        for t_snap, Q_snap in zip(sol.ts[1:], sol.ys[1:]):
            _record_state(Q_snap, float(t_snap))

    else:
        raise ValueError(f"Unknown integrator '{integrator}'.")

    # -------------------- 5️⃣  Done --------------------
    runtime = time.time() - start_time
    print(f"Completed simulation: {simulation_id} in {runtime:.2f} s")
    return simulation_data, runtime
