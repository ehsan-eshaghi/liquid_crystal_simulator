#!/usr/bin/env python
"""Find the maximum constant dt for explicit Euler integration.

Uses step-doubling error estimation with the relative and absolute tolerances
specified in ``config.ini``.

Usage::

    python -m scripts.dt_euler_finder [--a -0.75] [--b 1] [--K 1] [--S0 1]
"""
from __future__ import annotations

import argparse
import logging

import jax
import jax.numpy as jnp

from src.config import load_config
from src.common.boundary import create_circle_bc_mask, initialize_tangential_n
from src.jax_backend.boundary import apply_mask
from src.jax_backend.q_tensor import (
    functional_derivative_LdG,
    get_Q_from_n_S,
    make_random_Q,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find max dt for explicit Euler given rtol/atol",
    )
    parser.add_argument("--a", type=float, help="LdG 'a' parameter (override INI)")
    parser.add_argument("--b", type=float, help="LdG 'b' parameter (override INI)")
    parser.add_argument("--K", type=float, help="LdG 'K' parameter (override INI)")
    parser.add_argument("--S0", type=float, help="Initial scalar order S (override INI)")
    parser.add_argument("--config", default="config.ini", help="Path to config file")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)
    sim = config.simulation
    params = config.parameters

    L = sim.L
    gamma = sim.gamma
    t_final = sim.t_final
    rtol = sim.rtol
    atol = sim.atol
    dt_cfl = sim.dt_cfl

    a = args.a if args.a is not None else params.low_a
    b = args.b if args.b is not None else sim.b
    K = args.K if args.K is not None else params.high_K
    S0 = args.S0 if args.S0 is not None else params.low_S

    # Build initial Q-field and boundary masks
    Q0 = make_random_Q(L, S0)
    n_bc = initialize_tangential_n(L)
    bc = get_Q_from_n_S(jnp.asarray(n_bc), jnp.asarray(S0))
    bc_mask = jnp.asarray(create_circle_bc_mask(L // 2, L))

    Q_init = apply_mask(Q0, bc_mask, bc)

    @jax.jit
    def rhs(Q: jax.Array) -> jax.Array:
        dQ = -gamma * functional_derivative_LdG(Q, a, b, K)
        return apply_mask(dQ, bc_mask, jnp.zeros_like(dQ))

    @jax.jit
    def euler_step(Q: jax.Array, dt: float) -> tuple[jax.Array, jax.Array]:
        F0 = rhs(Q)
        Q1 = Q + dt * F0
        Qh = Q + 0.5 * dt * F0
        F1 = rhs(Qh)
        Q2 = Qh + 0.5 * dt * F1
        err = jnp.linalg.norm(Q2 - Q1)
        scale = rtol * jnp.linalg.norm(Q1) + atol
        return Q1, err / scale

    # Binary search for max dt in [0, dt_cfl]
    dt_lo, dt_hi = 0.0, dt_cfl
    for _ in range(10):
        dt_mid = 0.5 * (dt_lo + dt_hi)
        Q = Q_init
        t = 0.0
        valid = True
        while t < t_final:
            Q, rel_err = euler_step(Q, dt_mid)
            if rel_err > 1.0:
                valid = False
                break
            t += dt_mid
        if valid:
            dt_lo = dt_mid
        else:
            dt_hi = dt_mid
        if (dt_hi - dt_lo) < 1e-1 * dt_hi:
            break
        logger.info("dt_mid = %.6e", dt_mid)

    logger.info("Max dt_euler = %.3e  (rtol=%s, atol=%s)", dt_lo, rtol, atol)


if __name__ == "__main__":
    main()
