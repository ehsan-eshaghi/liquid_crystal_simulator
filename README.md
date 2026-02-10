# Simulating Nematic Liquid Crystal Dynamics using Landau-de Gennes Theory

![simulation preview](./a_-1.0_b_1.0_K_1.0_r_30_S_1.0_Q.gif)

This project implements a 2D simulation framework for nematic liquid crystals governed by the Landau-de Gennes (LdG) theory. Two solvers are provided:

- A **NumPy-based explicit Euler integrator** for pedagogical clarity.
- A **JAX-accelerated solver** with GPU support and adaptive time stepping using the **Dormand–Prince method** (under development).

Both solvers simulate the relaxational dynamics of the tensor order parameter \( Q_{ij} \), capturing defect evolution and energy minimization on a discretized lattice.

---

## 📁 Project Structure

```
.
├── a_-1.0_b_1.0_K_1.0_r_30_S_1.0_Q.gif       # Sample simulation output (Q-tensor visualization)
├── config.ini                                # Simulation parameters
├── jax_simulation_data.pkl                   # Output from JAX solver
├── numpy_simulation_data.pkl                 # Output from NumPy solver
├── run_simulations.py                        # CLI to run and compare both solvers
├── LICENSE
├── src/
│   ├── jax_backend/                          # GPU-accelerated backend
│   └── numpy_backend/                        # CPU-only backend
└── visualize.py                              # Plotting and animation utilities
```

---

## 🧪 Thesis Overview

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

## 🚧 Roadmap

Planned features include:

- [ ] Support for arbitrary boundary geometries (e.g., annulus, stripes)
- [ ] External field coupling (e.g., electric or magnetic)
- [ ] Anchoring angle control for boundary conditions
- [ ] Inhomogeneous material parameters
- [ ] Multi-domain Q-tensor initialization strategies
- [ ] Improved visualization and energy diagnostics
- [ ] Finalize adaptive Dormand–Prince solver for production use

---

## 🤝 Contributing

Contributions are welcome! Whether it’s fixing a bug, optimizing performance, or adding new physics modules, feel free to fork the repo and submit a pull request.

For major changes or suggestions, open an issue to discuss them first. Let’s build something useful for the liquid crystal simulation community 🚀

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
