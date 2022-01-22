import numpy as np
import h5py
import glob
import re
import matplotlib.pyplot as plt
from utils.plot_utils import plot_streamplot
from utils.plot_utils import plot_contour

# -- Read hd5 file
filename = "data/grad_adjoint.h5"
with h5py.File(filename, "r") as f:
    x = np.array(f["temp/x"])
    y = np.array(f["temp/y"])
    t = np.array(f["temp/v"])
    u = np.array(f["ux/v"])
    v = np.array(f["uy/v"])
fig, ax = plot_streamplot(x, y, t, u, v, return_fig=True)
fig.savefig("fig.png", bbox_inches="tight", dpi=200)
plt.show()

# print(np.min(t), np.max(t))

# -- Read hd5 file
filename = "data/grad_fd.h5"
with h5py.File(filename, "r") as f:
    x = np.array(f["temp/x"])
    y = np.array(f["temp/y"])
    t = np.array(f["temp/v"])
    u = np.array(f["ux/v"])
    v = np.array(f["uy/v"])
fig, ax = plot_streamplot(x, y, t, u, v, return_fig=True)
fig.savefig("fig.png", bbox_inches="tight", dpi=200)
plt.show()

# print(np.min(t), np.max(t))
