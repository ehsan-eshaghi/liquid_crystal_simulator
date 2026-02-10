#!/usr/bin/env python
"""Run Q-tensor simulations across a parameter sweep.

Usage::

    python -m scripts.run_simulations --backend jax --integrator dopri5
"""
from __future__ import annotations

import argparse
import importlib
import logging

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Q-tensor simulations.")
    parser.add_argument(
        "--backend",
        choices=["numpy", "jax"],
        default="jax",
        help="Choose backend for simulation (default: jax)",
    )
    parser.add_argument(
        "--integrator",
        choices=["euler", "dopri5"],
        default="dopri5",
        help="Choose integrator for simulation (default: dopri5)",
    )
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to configuration INI file (default: config.ini)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    param_combinations = config.parameters.combinations()

    if args.backend == "numpy":
        runner = importlib.import_module("src.numpy_backend.runner")
    else:
        runner = importlib.import_module("src.jax_backend.runner")

    all_simulation_records: list[dict] = []
    runtimes: list[float] = []

    for params in param_combinations:
        simulation_result, runtime = runner.run_simulation(params, config, args.integrator)
        all_simulation_records.extend(simulation_result)
        runtimes.append(runtime)

    output_filename = f"{args.backend}_simulation_data.pkl"
    all_simulations_df = pd.DataFrame(all_simulation_records)
    all_simulations_df.to_pickle(output_filename)

    avg_runtime = np.mean(runtimes)
    logger.info("All simulations complete. Data saved to %s", output_filename)
    logger.info("Average runtime per simulation: %.2f seconds", avg_runtime)


if __name__ == "__main__":
    main()
