from os import path
from charset_normalizer import from_path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def visualize_csv_data(csv_path: str):
    """Visualize data from a CSV file."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)

    t = data['Time']
    a0 = data['ai0']
    a1 = data['ai1']

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

    ax0.plot(t, a0)
    ax0.set_title("ai0")
    ax0.set_ylabel("Amplitude")
    ax0.grid(True)

    ax1.plot(t, a1)
    ax1.set_title("ai1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    plt.tight_layout()
    plt.show()
    
    
visualize_csv_data("data\CSV\MATLAB 1-00 PM Fri, Jun 28, 2024 Run8 .csv")

# Run visualization on all csv files in diretory data/CSVand save them to data/boiling_plots as png files
for file in path.listdir("data/CSV"):
    if file.endswith(".csv"):
        csv_path = path.join("data/CSV", file)
        visualize_csv_data(csv_path)
        save_path = path.splitext(csv_path)[0] + ".png"
        plt.savefig(path.join("data/boiling_plots", path.basename(save_path)))
        plt.close()



# def visualize_matlab_data(path: str):
#     """Visualize data from a MATLAB .mat file."""
#     import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import loadmat

# def visualize_matlab_data(mat_path: str):
#     """Visualize data from a MATLAB .mat file."""
#     d = loadmat(mat_path, squeeze_me=True)

#     t0 = np.asarray(d["UntitledPXI1Slot2_ai0_Time"]).ravel()
#     a0 = np.asarray(d["UntitledPXI1Slot2_ai0"]).ravel()

#     t1 = np.asarray(d["UntitledPXI1Slot2_ai1_Time"]).ravel()
#     a1 = np.asarray(d["UntitledPXI1Slot2_ai1"]).ravel()

#     if np.allclose(t0, t1):
#         t = t0
#         a1_aligned = a1
#     else:
#         t = t0
#         a1_aligned = np.interp(t0, t1, a1)

#     fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

#     ax0.plot(t, a0)
#     ax0.set_title("ai0")
#     ax0.set_ylabel("Amplitude")
#     ax0.grid(True)

#     ax1.plot(t, a1_aligned)
#     ax1.set_title("ai1")
#     ax1.set_xlabel("Time (s)")
#     ax1.set_ylabel("Amplitude")
#     ax1.grid(True)

#     plt.tight_layout()
#     plt.show()

# visualize_matlab_data("data/MATLAB/MATLAB 1-00 PM Fri, Jun 28, 2024 Run8 .mat")
