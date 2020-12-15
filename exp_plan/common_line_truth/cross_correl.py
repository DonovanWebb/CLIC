import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.transform import radon
import mrcfile as mrc 


def plot_all():
    return True
        

def sino(image):
    theta = np.linspace(0., 360., 120, endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)

    if plot_all():
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

        ax1.set_title("Original")
        ax1.imshow(image, cmap=plt.cm.Greys_r)
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
                extent=(0, 360, 0, sinogram.shape[0]), aspect='auto')

        fig1.tight_layout()
        return sinogram.T, fig1
    return sinogram.T, 0


def compare(sin1, sin2):
    mat = np.zeros((sin1.shape[0], sin2.shape[0]))
    for x in range(sin1.shape[0]):
        for y in range(sin2.shape[0]):
            correl = np.cbrt(((sin1[x]-sin2[y])**2).mean(axis=0))
            mat[x,y] = correl
            if correl < 7:
                plt.figure(100)
                plt.plot(sin1[x]) 
                plt.plot(sin2[y]) 
                print(x, y)
                plt.show()
    # mat = np.reshape(mat, (int(np.sqrt(mat.shape[0])),int(np.sqrt(mat.shape[0]))))
    return mat


def find_min(mat, ax):
    n_lines = 120
    x, y = mat.shape
    min_arg = np.argmin(mat)
    row = min_arg // x
    col = min_arg % x
    loc = np.array((row,col))
    loc2 = loc + 180/360 * n_lines
    if loc2[0] > n_lines:  loc2[0] -= n_lines
    if loc2[1] > n_lines:  loc2[1] -= n_lines

    locabs = loc / n_lines * 360
    loc2abs = loc2 / n_lines * 360
    if plot_all():
        ax.scatter(loc[1], loc[0])
        ax.scatter(loc2[1], loc2[0])
        ax.set_title(f'{locabs} , {loc2abs}')
        return ax, loc, loc2

    return 0, locabs, loc2abs



def get_sin(path):
    with mrc.open(path) as f:
        img = f.data
    sin, fig = sino(img)
    return sin, fig


def main(i, j):
    p1 = f'/dls/ebic/data/staff-scratch/Donovan/CLIC/dsets/exp_plan/7_5_projs/{i}.mrc'
    p2 = f'/dls/ebic/data/staff-scratch/Donovan/CLIC/dsets/exp_plan/7_5_projs/{j}.mrc'
    sin1, fig1 = get_sin(p1)
    sin2, fig2 = get_sin(p2)
    mat = compare(sin1, sin2)
    if plot_all():
        fig3 = plt.figure()
        ax = fig3.add_subplot(1, 1, 1)
        ax.imshow(mat)
        ax, loc1, loc2 = find_min(mat, ax)

        plt.figure(100)
        plt.plot(sin1[loc1[0]]) 
        plt.plot(sin2[loc1[1]]) 
        plt.show()
    else:
        ax, loc1, loc2 = find_min(mat, 0)
    print(f'{[i,j]}: {loc1}, {loc2}')


for i in range(10):
    for j in range(80,90):
        if j > i:
            main(i, j)


