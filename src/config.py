"""Centralized configuration for the liquid crystal simulator.

Parses ``config.ini`` once into typed dataclasses so that every module receives
the same validated settings without redundant ``config.getfloat(...)`` calls.
"""
from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default path – can be overridden by callers.
_DEFAULT_CONFIG_PATH = Path("config.ini")


@dataclass(frozen=True)
class SimulationConfig:
    """Core simulation settings shared by all backends."""

    L: int
    b: float
    gamma: float
    t_final: float
    sample_rate: int
    update_freq: int

    # Tolerance / CFL settings (used by dt-finder, etc.)
    rtol: float = 1e-5
    atol: float = 1e-5
    dt_cfl: float = 1.0


@dataclass(frozen=True)
class FixedStepConfig:
    """Settings for the explicit Euler (fixed-step) integrator."""

    dt: float


@dataclass(frozen=True)
class AdaptiveStepConfig:
    """Settings for the adaptive Dormand-Prince integrator."""

    dt0: float
    rtol: float
    atol: float


@dataclass(frozen=True)
class ParameterSweep:
    """Ranges for the material-parameter sweep."""

    low_a: float
    high_a: float
    size_a: int
    low_K: float
    high_K: float
    size_K: int
    low_S: float
    high_S: float
    size_S: int

    def combinations(self) -> list[tuple[float, float, float]]:
        """Return the Cartesian product ``(a, K, S)`` of the parameter grid."""
        a_values = np.round(np.linspace(self.low_a, self.high_a, self.size_a), 5)
        K_values = np.round(np.linspace(self.low_K, self.high_K, self.size_K), 5)
        S_values = np.round(np.linspace(self.low_S, self.high_S, self.size_S), 5)
        return list(product(a_values, K_values, S_values))


@dataclass(frozen=True)
class AppConfig:
    """Top-level container holding all configuration sections."""

    simulation: SimulationConfig
    fixed_step: FixedStepConfig
    adaptive_step: AdaptiveStepConfig
    parameters: ParameterSweep


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read *config.ini* and return a fully-typed :class:`AppConfig`.

    Parameters
    ----------
    path : str or Path
        Path to the INI file.  Defaults to ``config.ini`` in the working
        directory.

    Returns
    -------
    AppConfig
        Parsed and validated configuration.
    """
    cfg = configparser.ConfigParser()
    files_read = cfg.read(str(path))
    if not files_read:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    logger.info("Loaded configuration from %s", path)

    sim = cfg["simulation"]
    simulation = SimulationConfig(
        L=sim.getint("L"),
        b=sim.getfloat("b"),
        gamma=sim.getfloat("gamma"),
        t_final=sim.getfloat("t_final"),
        sample_rate=sim.getint("sample_rate"),
        update_freq=sim.getint("update_freq"),
        rtol=sim.getfloat("rtol", fallback=1e-5),
        atol=sim.getfloat("atol", fallback=1e-5),
        dt_cfl=sim.getfloat("dt_cfl", fallback=1.0),
    )

    fs = cfg["fixed_step_solver"]
    fixed_step = FixedStepConfig(dt=fs.getfloat("dt"))

    ads = cfg["adaptive_step_solver"]
    adaptive_step = AdaptiveStepConfig(
        dt0=ads.getfloat("dt0", fallback=1e-2),
        rtol=ads.getfloat("rtol", fallback=1e-5),
        atol=ads.getfloat("atol", fallback=1e-5),
    )

    p = cfg["parameters"]
    parameters = ParameterSweep(
        low_a=p.getfloat("low_a"),
        high_a=p.getfloat("high_a"),
        size_a=p.getint("size_a"),
        low_K=p.getfloat("low_K"),
        high_K=p.getfloat("high_K"),
        size_K=p.getint("size_K"),
        low_S=p.getfloat("low_S"),
        high_S=p.getfloat("high_S"),
        size_S=p.getint("size_S"),
    )

    return AppConfig(
        simulation=simulation,
        fixed_step=fixed_step,
        adaptive_step=adaptive_step,
        parameters=parameters,
    )
