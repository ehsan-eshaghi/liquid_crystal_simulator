#!/usr/bin/env python
"""Generate visualisations from saved simulation data.

Usage::

    python -m scripts.visualize --backend jax
"""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src.visualization import (
    capture_director_field_figures,
    capture_director_field_frame,
    capture_heatmap_figures,
    save_as_gif,
    save_as_video_ffmpeg,
    save_figure,
    save_figures,
    scatter_plot,
)

logger = logging.getLogger(__name__)


def load_simulation_data(pickle_filename: str) -> pd.DataFrame:
    """Load simulation data from a pickle file."""
    return pd.read_pickle(pickle_filename)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Visualise simulation results.")
    parser.add_argument(
        "--backend",
        choices=["numpy", "jax"],
        default="jax",
        help="Backend whose output to visualise (default: jax)",
    )
    args = parser.parse_args()

    pickle_filename = f"{args.backend}_simulation_data.pkl"
    logger.info("Loading simulation data from %s", pickle_filename)
    df = load_simulation_data(pickle_filename)

    grouped = df.groupby("id")
    for sim_id, group_df in grouped:
        output_dir = os.path.join(f"visualization_{args.backend}", sim_id)
        os.makedirs(output_dir, exist_ok=True)

        # Director-field animation
        frames = capture_director_field_frame(
            group_df["t"].values,
            group_df["Q"].values,
            group_df["n"].values,
            group_df["S"].values,
        )
        save_as_video_ffmpeg(frames, filename=f"{sim_id}_Q.mp4", output_dir=output_dir)
        save_as_gif(frames, filename=f"{sim_id}_Q.gif", output_dir=output_dir)

        # Snapshot figures (every 10th sample)
        figures = capture_director_field_figures(
            group_df["t"].values[::10],
            group_df["Q"].values[::10],
            group_df["n"].values[::10],
            group_df["S"].values[::10],
        )
        save_figures(figures, base_name=f"df_{sim_id}_", output_dir=output_dir)

        # Diagnostic scatter plots
        figure = scatter_plot(
            group_df["t"].values,
            group_df["fd_norm"].values,
            y_label="functional derivative L2 norm",
            label="",
        )
        save_figure(figure, output_dir, f"{sim_id}_fd_norm")

        figure = scatter_plot(
            group_df["t"].values,
            group_df["fe_mean"].values,
            y_label="free energy mean",
            label="",
        )
        save_figure(figure, output_dir, f"{sim_id}_fe_mean")

        figure = scatter_plot(
            group_df["t"].values,
            group_df["gop"].values,
            y_label="global order parameter",
            label="",
        )
        save_figure(figure, output_dir, f"{sim_id}_gop")

        # Free-energy heatmaps (every 10th sample)
        figures = capture_heatmap_figures(
            group_df["fe"].values[::10],
            group_df["t"].values[::10],
        )
        save_figures(figures, base_name=f"fe_{sim_id}_", output_dir=output_dir)

        logger.info("Saved visualisations for simulation %s in %s", sim_id, output_dir)


if __name__ == "__main__":
    main()
