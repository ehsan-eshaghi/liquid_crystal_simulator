import argparse
import configparser
import numpy as np
import pandas as pd
from itertools import product
import importlib


def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


def load_parameters(config):
    low_a = config.getfloat("parameters", "low_a")
    high_a = config.getfloat("parameters", "high_a")
    size_a = config.getint("parameters", "size_a")
    low_K = config.getfloat("parameters", "low_K")
    high_K = config.getfloat("parameters", "high_K")
    size_K = config.getint("parameters", "size_K")
    low_S = config.getfloat("parameters", "low_S")
    high_S = config.getfloat("parameters", "high_S")
    size_S = config.getint("parameters", "size_S")

    a_values = np.round(np.linspace(low_a, high_a, size_a), 5)
    K_values = np.round(np.linspace(low_K, high_K, size_K), 5)
    S_values = np.round(np.linspace(low_S, high_S, size_S), 5)

    return list(product(a_values, K_values, S_values))


def main():
    parser = argparse.ArgumentParser(description="Run Q-tensor simulations.")
    parser.add_argument("--backend", choices=["numpy", "jax"], default="jax", help="Choose backend for simulation")
    parser.add_argument(
        "--integrator", choices=["euler", "dopri5"], default="dopri5", help="Choose integrator for simulation"
    )
    args = parser.parse_args()

    config = load_config()
    param_combinations = load_parameters(config)

    if args.backend == "numpy":
        runner = importlib.import_module("src.numpy_backend.numpy_runner")
    else:
        runner = importlib.import_module("src.jax_backend.jax_runner")

    all_simulation_records = []
    runtimes = []

    for params in param_combinations:
        simulation_result, runtime = runner.run_simulation(params, config, args.integrator)
        all_simulation_records.extend(simulation_result)
        runtimes.append(runtime)

    output_filename = f"{args.backend}_simulation_data.pkl"
    all_simulations_df = pd.DataFrame(all_simulation_records)
    all_simulations_df.to_pickle(output_filename)

    avg_runtime = np.mean(runtimes)
    print(f"\nAll simulations complete and data saved to {output_filename}")
    print(f"Average runtime per simulation: {avg_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
