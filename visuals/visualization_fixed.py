"""
Same as visualization.py but with y-axis fixed to (-0.3, 0.3) for both ai0 and ai1 subplots.
Use for consistent comparison across samples.
"""
import os
import os.path as path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so script doesn't hang when saving figures
import matplotlib.pyplot as plt

Y_LIM = (-0.3, 0.3)


def visualize_csv_data(csv_path: str):
    """Visualize data from a CSV file with fixed y-axis (-0.3, 0.3)."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    t = data['Time']
    a0 = data['ai0']
    a1 = data['ai1']

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    ax0.plot(t, a0)
    ax0.set_title("ai0")
    ax0.set_ylabel("Amplitude")
    ax0.set_ylim(Y_LIM)
    ax0.grid(True)

    ax1.plot(t, a1)
    ax1.set_title("ai1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(Y_LIM)
    ax1.grid(True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    count = 0
    for file in os.listdir("data/CSV"):
        if file.endswith(".csv"):
            csv_path = path.join("data/CSV", file)
            fig = visualize_csv_data(csv_path)
            save_path = path.splitext(csv_path)[0] + ".png"
            # Optionally save to a separate folder, e.g. data/boiling_plots_fixed
            out_dir = path.join("data", "boiling_plots_fixed")
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(path.join(out_dir, path.basename(save_path)))
            plt.close(fig)
            count += 1
    print(f"Done. Saved {count} plot(s) to data/boiling_plots_fixed/")
