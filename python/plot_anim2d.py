import numpy as np
import h5py
import glob
import re
import matplotlib.pyplot as plt
from utils.plot_utils import plot_contour
from utils.plot_utils import plot_streamplot
import os.path
import ffmpeg

settings = {
    "duration": None,  # time in seconds; determines fps
    "filename": "data/out.mp4",
}

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
print("From number:")
i0 = int(input())
print("To number:")
i9 = int(input())
print("Step:")
step = int(input())

for i, f in enumerate(fname[i0:i9:step]):
    # -- Read hd5 file
    filename = f
    figname = f.replace(".h5", ".png")

    if os.path.isfile(figname):
        print("{} already exists".format(figname))
        continue

    with h5py.File(filename, "r") as f:
        t = np.array(f["temp/v"])
        u = np.array(f["ux/v"])
        v = np.array(f["uy/v"])
        x = np.array(f["temp/x"])
        y = np.array(f["temp/y"])
        try:
            tbc = np.array(f["tempbc/v"])
        except:
            tbc = np.zeros(t.shape)

    print("Plot {:}".format(filename))
    # fig, ax = plot_contour(x, y, t, return_fig=True)
    fig, ax = plot_streamplot(x, y, t + tbc, u, v, return_fig=True)
    fig.savefig(figname, dpi=200, bbox_inches="tight")
    plt.close("all")
