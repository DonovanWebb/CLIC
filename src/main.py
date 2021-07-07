#!/usr/bin/env python
'''
Main file for CLIC algorithm.
Config options can be set at the top and the jobs to be run can be seen at the
bottom of the file.

Results are clustering of sinograms - grouping of sinograms can be seen
iteratively in the output stream. At the end the large merges will be
displayed.
'''
import numpy as np
from sinogram_input import sinogram_main
from dim_red import fitmodel
from clustering import clustering_main
import star_writer
import plt_truth
import discrete
import sin_guesser
import argparse

# Other dependencies
from skimage.transform import radon, resize
import mrcfile
from sklearn.metrics.pairwise import euclidean_distances as eucl_dist
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import time
import gemmi
import os

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


parser = argparse.ArgumentParser()

t = ''' Dataset to be considered for clustering. Input path to mrcs
stack, to individual mrc particles, or to particle starfile with "/PATH/TO/PARTS/*.mrc"
(Notes: 1. Don't forget "", 2. if star file run from relion home dir)
'''
parser.add_argument("-i", "--data_set", help=t, required=True, type=str)

t = ''' Number of projections to consider. Defaults to 1000 '''
parser.add_argument("-n", "--num", help=t, default=1000, type=int)

t = ''' Signal to noise ratio of projection before making sinograms '''
parser.add_argument("-r", "--snr", help=t, default=-1, type=float)

t = ''' Downscaling of image prior to making sinograms '''
parser.add_argument("-d", "--down_scale", help=t, default=2, type=int)

t = ''' Number of components of dimensional reduction technique '''
parser.add_argument("-c", "--num_comps", help=t, default=10, type=int)

t = ''' Dimensional reduction technique.
options are: PCA, UMAP, TSNE, LLE, ISOMAP, MDS, TRIMAP '''
parser.add_argument("-m", "--model", help=t, default='UMAP', type=str)

t = ''' Number of lines in one sinogram (shouldn't need to change
recommended=120)'''
parser.add_argument("-l", "--nlines", help=t, default=120, type=int)

t = ''' Run on gpu with CUDA '''
parser.add_argument("-g", "--gpu", help=t, default=False, action='store_true')

args = parser.parse_args()



def plot(lines_reddim, num, clic_dir, ids):
    ''' Simple plotting of dimensionally reduced lines. Set up for dset with
    two classes (alternating) '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    per_sino = lines_reddim.shape[0] // num
    plt.figure(1)
    for x in range(num):
        if x < 16:
            plt.scatter(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, 1])
    plt.axis('off')
    plt.savefig(f"{clic_dir}/2d_plot_1.pdf")
    plt.figure(2)
    for x in range(num):
        if x < 160:
            if x % 2 == 0:
                plt.scatter(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                            lines_reddim[x*per_sino:(x+1)*per_sino:, 1], c='r', alpha=0.4)
            else:
                plt.scatter(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                            lines_reddim[x*per_sino:(x+1)*per_sino:, 1], c='b', alpha=0.4)
    plt.axis('off')
    plt.savefig(f"{clic_dir}/2d_plot_binary.pdf")

    fig = plt.figure(3)
    ax = fig.gca(projection='3d')

    ints = [os.path.basename(x) for x in ids]
    ints = [os.path.splitext(x)[0] for x in ints]
    ids_ints = [int(x) for x in ints]
    gt_ids_bin = [x % 2 for x in ids_ints]
    for x in range(len(gt_ids_bin)):
        if gt_ids_bin[x] == 0:
            ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                    lines_reddim[x*per_sino:(x+1)*per_sino:, 1],
                    lines_reddim[x*per_sino:(x+1)*per_sino:, 2], c='r', alpha=0.4)
        else:
            ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                    lines_reddim[x*per_sino:(x+1)*per_sino:, 1],
                    lines_reddim[x*per_sino:(x+1)*per_sino:, 2], c='b', alpha=0.4)
    plt.savefig(f"{clic_dir}/3d_plot_binary.pdf")

    fig = plt.figure(5)
    ax = fig.gca(projection='3d')

    for x in range(5):
        ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                lines_reddim[x*per_sino:(x+1)*per_sino:, 1],
                lines_reddim[x*per_sino:(x+1)*per_sino:, 2], alpha=0.8)
    plt.savefig(f"{clic_dir}/3d_plot_binary.pdf")


if __name__ == '__main__':
    start = time.time()

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    clic_dir = f'CLIC_Job_{time_stamp}'
    os.makedirs(clic_dir, exist_ok = True)
    all_ims, num, ids = sinogram_main(args)
    star_file  = star_writer.create(ids, clic_dir)
    
    args.num = num  # Update with lowest num
    lines_reddim, model = fitmodel(all_ims, args.model, args.num_comps)
    """ for generating figures for pca recon and eigenfilters """
    import pca_recon
    pca_recon.recon_sino(all_ims[-1], model, args)
    pca_recon.plt_comps(model)
    plt.show()

    plot(lines_reddim, args.num, clic_dir, ids)

    '''
    """
    Optional code - for experiment looking at how common line group spreads with noise
    """
    r_lines = discrete.rand_lines(lines_reddim, n_rand=100)
    discrete.get_stats(r_lines)
    # r = 17.5
    all_groups = sin_guesser.main(args.num, r)
    theta = 3
    th_lines = sin_guesser.choose_rand_group(all_groups)

    group_lines = discrete.get_discrete_lines(lines_reddim, th_lines, r, theta)
    discrete.get_stats(group_lines)
    discrete.plot(group_lines, lines_reddim)
    '''

    clustering_main(lines_reddim, args, clic_dir, ids)
    print(time.time() - start)
    plt.show()
