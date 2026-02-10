#!/usr/bin/env python
"""
dt_euler_finder.py

Finds the maximum constant dt for explicit Euler such that every
step's step-doubling error (relative+absolute) stays within the
rtol & atol specified in config.ini.
"""
import argparse
import configparser
import numpy as np
import jax
import jax.numpy as jnp

# JAX-backend functions for dynamics
from src.jax_backend.boundary_conditions import apply_mask
from src.jax_backend.q_tensor import functional_derivative_LdG, make_random_Q, get_Q_from_n_S

# NumPy-backend for boundary condition initialization and mask
from src.numpy_backend.boundary_conditions import initialize_tangential_n, create_circle_bc_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Find max dt for explicit Euler given rtol/atol")
    parser.add_argument("--a", type=float, help="LdG 'a' parameter (override INI)")
    parser.add_argument("--b", type=float, help="LdG 'b' parameter (override INI)")
    parser.add_argument("--K", type=float, help="LdG 'K' parameter (override INI)")
    parser.add_argument("--S0", type=float, help="Initial scalar order S (override INI)")
    return parser.parse_args()


def main():
    # 1) Load config.ini
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")
    sim = cfg["simulation"]
    params = cfg["parameters"]

    # Simulation settings
    L = sim.getint("L")
    gamma = sim.getfloat("gamma")
    t_final = sim.getfloat("t_final")
    rtol = sim.getfloat("rtol")
    atol = sim.getfloat("atol")
    dt_cfl = sim.getfloat("dt_cfl", fallback=1.0)

    # Fetch parameters (override INI if provided)
    args = parse_args()
    a = args.a if args.a is not None else float(params["low_a"])
    b = args.b if args.b is not None else float(sim.get("b"))
    K = args.K if args.K is not None else float(params["high_K"])
    S0 = args.S0 if args.S0 is not None else float(params["low_S"])

    # 2) Build initial Q-field and BC masks
    # 2a) Random Q initial state via JAX
    Q0 = make_random_Q(L, S0)

    # 2b) Pure-Python tangential BC (NumPy)
    n_bc_np = initialize_tangential_n(L)  # (L, L, 2)
    bc_np = get_Q_from_n_S(n_bc_np, S0)  # (L, L, 2, 2)
    bc = jnp.array(bc_np)

    # 2c) Circular boundary mask
    bc_mask = create_circle_bc_mask(L // 2, L)

    # Apply BC on GPU
    Q_init = apply_mask(Q0, bc_mask, bc)

    # 3) RHS of gradient flow with BC
    @jax.jit
    def rhs(Q):
        dQ = -gamma * functional_derivative_LdG(Q, a, b, K)
        return apply_mask(dQ, bc_mask, jnp.zeros_like(dQ))

    # 4) One explicit Euler step + step-doubling error estimate
    @jax.jit
    def euler_step(Q, dt):
        F0 = rhs(Q)
        Q1 = Q + dt * F0
        Qh = Q + 0.5 * dt * F0
        F1 = rhs(Qh)
        Q2 = Qh + 0.5 * dt * F1
        err = jnp.linalg.norm(Q2 - Q1)
        scale = rtol * jnp.linalg.norm(Q1) + atol
        return Q1, err / scale

    # 5) Binary search for max dt in [0, dt_cfl]
    dt_lo, dt_hi = 0.0, dt_cfl
    for _ in range(10):  # ~1% precision
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

        print(dt_mid)

    print(f"→ max dt_euler = {dt_lo:.3e}  (rtol={rtol}, atol={atol})")


if __name__ == "__main__":
    main()
