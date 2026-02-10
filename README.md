# Simulating Nematic Liquid Crystal Dynamics using Landau-de Gennes Theory

![simulation preview](./a_-1.0_b_1.0_K_1.0_r_30_S_1.0_Q.gif)

This project implements a 2D simulation framework for nematic liquid crystals governed by the Landau-de Gennes (LdG) theory. Two solvers are provided:

- A **NumPy-based explicit Euler integrator** for pedagogical clarity.
- A **JAX-accelerated solver** with GPU support and adaptive time stepping using the **Dormand-Prince method** (under development).

Both solvers simulate the relaxational dynamics of the tensor order parameter \( Q_{ij} \), capturing defect evolution and energy minimisation on a discretised lattice.

---

## Project Structure

```
.
├── pyproject.toml                  # Project metadata & dependencies
├── config.ini                      # Simulation parameters
├── scripts/
│   ├── run_simulations.py          # CLI to run parameter sweeps
│   ├── visualize.py                # Generate plots, GIFs and videos
│   └── dt_euler_finder.py          # Find max stable Euler time step
├── src/
│   ├── config.py                   # Centralised configuration (dataclasses)
│   ├── visualization.py            # Rendering & export helpers
│   ├── common/
│   │   └── boundary.py             # Shared masks & director initialisation
│   ├── numpy_backend/
│   │   ├── diff.py                 # Finite-difference operators
│   │   ├── q_tensor.py             # Q-tensor operations
│   │   └── runner.py               # Simulation runner (Euler)
│   └── jax_backend/
│       ├── boundary.py             # JIT-compiled apply_mask
│       ├── diff.py                 # Finite-difference operators (JAX)
│       ├── q_tensor.py             # Q-tensor operations (JAX)
│       └── runner.py               # Simulation runner (Euler / Dopri5)
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Core (NumPy backend only)
pip install .

# With JAX + GPU support
pip install ".[jax]"
```

---

## Usage

Run a parameter sweep:

```bash
python -m scripts.run_simulations --backend jax --integrator dopri5
```

Generate visualisations from saved data:

```bash
python -m scripts.visualize --backend jax
```

Find the maximum stable Euler time step:

```bash
python -m scripts.dt_euler_finder
```

---

## Thesis Overview

This codebase supports the experiments and figures in the thesis:

**Title:** *Simulating Nematic Liquid Crystal Dynamics Using Landau-de Gennes Theory: From Euler Integration to GPU-Accelerated Adaptive Solvers*
**Author:** Ehsan Es'haghi
**Date:** March 2025 (in progress, not yet published)

The simulations model relaxational (Model A) dynamics of nematic order on a 2D lattice. The code allows:

- Custom Dirichlet boundary conditions on user-defined subregions
- Parametric control over material constants (e.g., \( a, b, K \))
- Comparative analysis of fixed vs adaptive time-stepping
- GPU acceleration with JAX's JIT compilation and vectorization

---

## Roadmap

Planned features include:

- [ ] Support for arbitrary boundary geometries (e.g., annulus, stripes)
- [ ] External field coupling (e.g., electric or magnetic)
- [ ] Anchoring angle control for boundary conditions
- [ ] Inhomogeneous material parameters
- [ ] Multi-domain Q-tensor initialisation strategies
- [ ] Improved visualisation and energy diagnostics
- [ ] Finalise adaptive Dormand-Prince solver for production use

---

## Contributing

Contributions are welcome! Whether it's fixing a bug, optimising performance, or adding new physics modules, feel free to fork the repo and submit a pull request.

For major changes or suggestions, open an issue to discuss them first.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
