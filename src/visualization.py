"""Visualization utilities for the liquid crystal simulator.

Provides helpers to render director-field quiver plots, free-energy heatmaps,
scatter plots, and export them as GIF / MP4 / PNG files.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Director-field rendering
# ---------------------------------------------------------------------------

def create_simulation_canvas(
    n: NDArray, S: NDArray,
) -> tuple[Figure, plt.Axes, FigureCanvas, object, object]:
    """Create a reusable matplotlib canvas for director-field visualisation.

    Parameters
    ----------
    n : NDArray
        Director field of shape ``(L, L, 2)``.
    S : NDArray
        Scalar order-parameter field of shape ``(L, L)``.

    Returns
    -------
    fig, ax, canvas, director_lines, order_heatmap
    """
    L = n.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    canvas = FigureCanvas(fig)

    director_lines = ax.quiver(
        n[..., 0], n[..., 1],
        headwidth=0, scale=L / 2, pivot="middle", angles="xy",
    )
    order_heatmap = ax.imshow(S, vmin=0, vmax=1)
    plt.colorbar(order_heatmap, fraction=0.045, label="$S$")
    ax.set_title("t=0")

    return fig, ax, canvas, director_lines, order_heatmap


def capture_director_field_frame(
    t_list: Sequence[float],
    Q_list: Sequence[NDArray],
    n_list: Sequence[NDArray],
    S_list: Sequence[NDArray],
) -> list[Image.Image]:
    """Render an animation-ready list of PIL frames for the director field.

    Parameters
    ----------
    t_list, Q_list, n_list, S_list
        Parallel sequences of time values, Q-tensors, directors, and order
        parameters.

    Returns
    -------
    list[Image.Image]
        RGB frames suitable for GIF / video export.
    """
    fig, ax, canvas, director_lines, order_heatmap = create_simulation_canvas(
        n_list[0], S_list[0],
    )

    frames: list[Image.Image] = []
    for t, _Q, n, S in zip(t_list, Q_list, n_list, S_list):
        director_lines.set_UVC(n[..., 0], n[..., 1])
        order_heatmap.set_data(S)
        ax.set_title(f"t = {t}")
        canvas.draw()

        buf, (width, height) = canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype="uint8").reshape((height, width, 4))[:, :, :3]
        frames.append(Image.fromarray(frame))

    plt.close(fig)
    return frames


def capture_director_field_figures(
    t_list: Sequence[float],
    Q_list: Sequence[NDArray],
    n_list: Sequence[NDArray],
    S_list: Sequence[NDArray],
) -> list[Figure]:
    """Create individual matplotlib figures for selected time snapshots.

    Parameters
    ----------
    t_list, Q_list, n_list, S_list
        Parallel sequences of time values, Q-tensors, directors, and order
        parameters.

    Returns
    -------
    list[Figure]
        One figure per snapshot.
    """
    figures: list[Figure] = []
    for t, _Q, n, S in zip(t_list, Q_list, n_list, S_list):
        fig, ax, _, director_lines, order_heatmap = create_simulation_canvas(n, S)
        director_lines.set_UVC(n[..., 0], n[..., 1])
        order_heatmap.set_data(S)
        ax.set_title(f"t = {t}")
        figures.append(fig)
    return figures


# ---------------------------------------------------------------------------
# Free-energy heatmap rendering
# ---------------------------------------------------------------------------

def capture_fe_frames(heatmaps: Sequence[NDArray]) -> list[Image.Image]:
    """Render free-energy heatmaps as PIL frames.

    Parameters
    ----------
    heatmaps : sequence of NDArray
        One 2-D array per time step.

    Returns
    -------
    list[Image.Image]
        RGB frames.
    """
    frames: list[Image.Image] = []
    for heatmap in heatmaps:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        ax.axis("off")
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))
        rgb_array = buf[:, :, 1:4]  # ARGB -> RGB

        frames.append(Image.fromarray(rgb_array, "RGB"))
        plt.close(fig)

        logger.debug("Free-energy sum: %s", heatmap.sum())
    return frames


def capture_heatmap_figures(
    heatmaps: Sequence[NDArray], times: Sequence[float],
) -> list[Figure]:
    """Create individual free-energy heatmap figures.

    Parameters
    ----------
    heatmaps : sequence of NDArray
        One 2-D free-energy array per time step.
    times : sequence of float
        Corresponding time values.

    Returns
    -------
    list[Figure]
    """
    figs: list[Figure] = []
    for heatmap, t in zip(heatmaps, times):
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        ax.set_title(f"t = {t}")
        fig.colorbar(im, ax=ax)
        fig.canvas.draw()
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Scatter / line plots
# ---------------------------------------------------------------------------

def scatter_plot(
    x: NDArray,
    y: NDArray,
    y_label: str,
    label: str,
    x_label: str = "time",
) -> Figure:
    """Create a scatter plot of *y* vs *x*.

    Parameters
    ----------
    x, y : NDArray
        Data arrays.
    y_label : str
        Label for the y-axis.
    label : str
        Legend label for the data series.
    x_label : str
        Label for the x-axis (default ``"time"``).

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, marker="o", linestyle="-", color="b", label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if label:
        ax.legend()
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_as_gif(
    frames: list[Image.Image],
    filename: str = "director_lines_evolution.gif",
    output_dir: str = ".",
    duration: int = 500,
) -> None:
    """Save a list of PIL frames as an animated GIF.

    Parameters
    ----------
    frames : list[Image.Image]
        Frames to save.
    filename : str
        Output filename.
    output_dir : str
        Target directory (created if necessary).
    duration : int
        Milliseconds between frames (default ``500``).
    """
    os.makedirs(output_dir, exist_ok=True)
    if not frames:
        logger.warning("No frames to save for %s.", filename)
        return
    path = os.path.join(output_dir, filename)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    logger.info("GIF saved as '%s'.", path)


def save_as_video_ffmpeg(
    frames: list[Image.Image],
    filename: str = "output.mp4",
    output_dir: str = ".",
    fps: int = 5,
    crf: int = 23,
    codec: str = "libx264",
) -> None:
    """Save frames as an MP4 video using FFmpeg.

    Parameters
    ----------
    frames : list[Image.Image]
        Frames to encode.
    filename, output_dir, fps, crf, codec
        FFmpeg encoding settings.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, filename)

    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths: list[str] = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        frame.save(frame_path)
        frame_paths.append(frame_path)

    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} "
        f"-i {temp_dir}/frame_%04d.png "
        f"-c:v {codec} -crf {crf} -preset slow {video_path}"
    )
    logger.info("Running FFmpeg: %s", ffmpeg_cmd)
    os.system(ffmpeg_cmd)

    for fp in frame_paths:
        os.remove(fp)
    os.rmdir(temp_dir)

    logger.info("Video saved as '%s'.", video_path)


def save_figures(
    figs: list[Figure], base_name: str, output_dir: str,
) -> None:
    """Save a list of figures as numbered PNGs.

    Parameters
    ----------
    figs : list[Figure]
        Figures to save.
    base_name : str
        Filename prefix (index is appended).
    output_dir : str
        Target directory (created if necessary).
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, fig in enumerate(figs):
        file_name = f"{base_name}{idx}.png"
        fig.savefig(os.path.join(output_dir, file_name), bbox_inches="tight")
        plt.close(fig)


def save_figure(figure: Figure, directory: str, file_name: str) -> None:
    """Save a single figure as a PNG.

    Parameters
    ----------
    figure : Figure
        The figure to save.
    directory : str
        Target directory (created if necessary).
    file_name : str
        Base name (without extension).
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{file_name}.png")
    figure.savefig(file_path)
    plt.close(figure)
    logger.info("Figure saved as %s", file_path)
