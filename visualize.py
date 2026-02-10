import os
import pandas as pd
import numpy as np

# Import your visualization functions from the simulation module.
from src.numpy_backend.q_tensor import get_n_S_from_Q
from src.visualization import (
    capture_director_field_frame,
    capture_heatmap_figures,
    save_as_gif,
    save_as_video_ffmpeg,
    save_figures,
    save_figure,
    scatter_plot,
    capture_director_field_figures,
)


def load_simulation_data(pickle_filename):
    """Load simulation data from a pickle file as a pandas DataFrame."""
    return pd.read_pickle(pickle_filename)


backend = "jax"


def main():
    # File containing the saved simulation log DataFrame.
    pickle_filename = f"jax_simulation_data.pkl"
    print(f"Loading simulation data from {pickle_filename}")
    # Load the DataFrame.
    df = load_simulation_data(pickle_filename)

    # Group the data by simulation id.
    # df = df[df["t"] % 10 == 0]
    df["t"] = df["t"]*10
    grouped = df.groupby("id")
    for sim_id, group_df in grouped:
        # Convert the group DataFrame to a list of dictionaries (records).

        # Create an output directory for this simulation's results.
        output_dir = os.path.join(f"visualization_{backend}/", sim_id)
        os.makedirs(output_dir, exist_ok=True)

        frames = capture_director_field_frame(
            group_df["t"].values, group_df["Q"].values, group_df["n"].values, group_df["S"].values
        )
        filename = f"{sim_id}_Q.mp4"
        save_as_video_ffmpeg(frames, filename=filename, output_dir=output_dir)
        filename = f"{sim_id}_Q.gif"
        save_as_gif(frames, filename=filename, output_dir=output_dir)
        figures = capture_director_field_figures(
            group_df["t"].values[::10],
            group_df["Q"].values[::10],
            group_df["n"].values[::10],
            group_df["S"].values[::10],
        )
        save_figures(figures, base_name=f"df_{sim_id}_", output_dir=output_dir)
        figure = scatter_plot(
            group_df["t"].values,
            group_df["fd_norm"].values,
            y_label="functional derivative L2 norm",
            label="",
            x_label="time",
        )
        save_figure(figure, output_dir, f"{sim_id}_fd_norm")
        figure = scatter_plot(
            group_df["t"].values, group_df["fe_mean"].values, y_label="free energy mean", label="", x_label="time"
        )
        save_figure(figure, output_dir, f"{sim_id}_fe_mean")
        figure = scatter_plot(
            group_df["t"].values, group_df["gop"].values, y_label="global order parameter", label="", x_label="time"
        )
        save_figure(figure, output_dir, f"{sim_id}_gop")

        figures = capture_heatmap_figures(group_df["fe"].values[::10], group_df["t"].values[::10])
        save_figures(figures, base_name=f"fe_{sim_id}_", output_dir=output_dir)
        print(f"Saved visualizations for simulation {sim_id} in {output_dir}")


if __name__ == "__main__":
    main()
