import numpy as np
import h5py
import glob
import re
import matplotlib.pyplot as plt
from utils.plot_utils import plot_streamplot
from utils.plot_utils import plot_contour

# -- Get list of files
fname, time = [], []
for file in [*glob.glob("*.h5"), *glob.glob("data/*.h5")]:
    try:
        time.append(float(re.findall("\d+\.\d+", file)[0]))
        fname.append(file)
    except:
        print("No number found in {:}".format(file))
idx = np.argsort(time)
fname = np.array(fname)[idx]
time = np.array(time)[idx]

for i, f in enumerate(fname):
    print("# {:3d}: {:}".format(i, f))
print("Enter number:")
i = int(input())

# -- Read hd5 file
filename = fname[i]
with h5py.File(filename, "r") as f:
    x = np.array(f["x"])
    y = np.array(f["y"])
    t = np.array(f["temp/v"])
    try:
        u = np.array(f["ux/v"])
        v = np.array(f["uy/v"])
        p = np.array(f["pres/v"])
    except:
        u = v = p = None

    try:
        vorticity = np.array(f["vorticity/v"])
    except:
        vorticity = None
        
    try:
        s = np.array(f["solid/mask"])
    except:
        s = None

print("Plot {:}".format(filename))
if u is not None:
    fig, ax = plot_streamplot(x, y, t, u, v, return_fig=True)
else:
    fig, ax = plot_contour(x, y, t, return_fig=True)

fig.savefig("fig.png", bbox_inches="tight", dpi=200)
plt.show()

# Plot vorticity
if vorticity is not None:
    fig, ax = plot_streamplot(x, y, vorticity, u, v, return_fig=True)
    plt.show()

# Contour line for obstacle
if s is not None:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ax.contour(xx, yy, s, levels=[0.5], colors="k")
    plt.show()
