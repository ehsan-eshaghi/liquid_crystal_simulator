import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from PIL import Image

def create_simulation_canvas(n, S):

    L = n.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5))
    canvas = FigureCanvas(fig)

    director_lines = ax.quiver(
        n[..., 0], n[..., 1],
        headwidth=0, scale=L / 2, pivot="middle", angles="xy"
    )
    order_heatmap = ax.imshow(S, vmin=0, vmax=1)
    plt.colorbar(order_heatmap, fraction=0.045, label="$S$")
    ax.set_title('t=0')

    return fig, ax, canvas, director_lines, order_heatmap

def capture_director_field_frame(t_list,Q_list,n_list,S_list):

    fig, ax, canvas, director_lines, order_heatmap = create_simulation_canvas(n_list[0], S_list[0])

    frames = []
    # Loop through each record and capture a frame for the current state.
    for t,Q,n,S in zip(t_list,Q_list,n_list,S_list):
        director_lines.set_UVC(n[..., 0], n[..., 1])
        order_heatmap.set_data(S)
        ax.set_title(f"t = {t}")
        canvas.draw()

        buf, (width, height) = canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype='uint8').reshape((height, width, 4))[:, :, :3]
        frames.append(Image.fromarray(frame))

    return frames

def capture_director_field_figures(t_list, Q_list, n_list, S_list):
    figures = []
    
    for t, Q, n, S in zip(t_list, Q_list, n_list, S_list):
        fig, ax, _, director_lines, order_heatmap = create_simulation_canvas(n, S)

        director_lines.set_UVC(n[..., 0], n[..., 1])
        order_heatmap.set_data(S)
        ax.set_title(f"t = {t}")
        
        figures.append(fig)  # Store the full figure to save later

    return figures

def capture_fe_frames(heatmaps):
    frames = []
    for heatmap in heatmaps:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        ax.axis("off")
        fig.canvas.draw()
        
        # Get the width and height of the canvas
        w, h = fig.canvas.get_width_height()
        
        # Retrieve ARGB data as a 1D array of bytes
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # Reshape the buffer to (height, width, 4) for ARGB
        buf.shape = (h, w, 4)
        
        # Convert ARGB to RGB by dropping the alpha channel
        # The ARGB order is [A, R, G, B]. We take channels 1 to 4.
        rgb_array = buf[:, :, 1:4]
        
        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(rgb_array, 'RGB')
        frames.append(image)
        
        plt.close(fig)

        print("sum fe:",heatmap.sum())
    return frames

def save_as_gif(frames, filename="director_lines_evolution.gif",output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if frames:
        frames[0].save(
            f"{output_dir}/{filename}",
            save_all=True,
            append_images=frames[1:],
            duration=500,  # Duration (ms) between frames
            loop=0         # Loop forever
        )
        print(f"GIF saved as '{filename}'.")
    else:
        print("No frames were captured.")

def save_as_video_ffmpeg(frames, filename="output.mp4", output_dir=".", fps=5, crf=23, codec="libx264"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_path = os.path.join(output_dir, filename)

    # Save frames as temporary images
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        frame.save(frame_path)
        frame_paths.append(frame_path)

    # Build FFmpeg command
    ffmpeg_cmd = f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v {codec} -crf {crf} -preset slow {video_path}"

    print(f"Running FFmpeg: {ffmpeg_cmd}")
    os.system(ffmpeg_cmd)

    # Cleanup temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)

    print(f"Video saved as '{video_path}'.")


def scatter_plot(x, y, y_label, label, x_label='time'):
    # Prepare the figure
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, marker='o', linestyle='-', color='b', label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    
    return fig  # Return the figure instead of saving it

def save_figures(figs, base_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, fig in enumerate(figs):
        file_name = f"{base_name}{idx}.png"
        fig.savefig(os.path.join(output_dir, file_name), bbox_inches="tight")
        plt.close(fig)

def save_figure(figure, directory, file_name):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construct the full file path
    file_name = f"{file_name}.png"
    file_path = os.path.join(directory, file_name)
    
    # Save the figure
    figure.savefig(file_path)
    
    print(f"Figure saved as {file_path}")

def capture_heatmap_figures(heatmaps,times):
    figs = []
    for heatmap,t in zip(heatmaps,times):
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        ax.set_title(f"t = {t}")  # Add title here
        # If you want the axis labels, do not turn off the axis:
        # ax.axis("off")
        # Instead, you can customize ticks or labels if needed.
        fig.colorbar(im, ax=ax)  # Add colorbar

        # Draw the canvas to make sure all elements are rendered
        fig.canvas.draw()
        
        figs.append(fig)
    return figs