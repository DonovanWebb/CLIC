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
from sinogram_input import get_part_locs
from dim_red import fitmodel
from clustering import clustering_main
import clustering
import star_writer
import plt_truth
import discrete
import min_matrix
import sin_guesser
import argparse

# Other dependencies
from skimage.transform import radon, resize
import mrcfile
from sklearn.metrics.pairwise import euclidean_distances as eucl_dist
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
import time
import gemmi
import os
import random
from itertools import permutations
import numba
from numba import cuda
import math
import collections

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

t = ''' Batchsize '''
parser.add_argument("-b", "--batch_size", help=t, default=-1, type=int)

t = ''' Number of clusters '''
parser.add_argument("-k", "--num_clusters", help=t, default=2, type=int)

args = parser.parse_args()



def plot(lines_reddim, num, clic_dir, ids):
    ''' Simple plotting of dimensionally reduced lines. Set up for dset with
    two classes (alternating) '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from itertools import combinations

    ints = [os.path.basename(x) for x in ids]
    ints = [os.path.splitext(x)[0] for x in ints]
    ids_ints = [int(x) for x in ints]
    gt_ids_bin = [x % 2 for x in ids_ints]

    per_sino = lines_reddim.shape[0] // num
    ############## 2D 
    # plt.figure(1)
    # for x in range(num):
    #     if x < 16:
    #         plt.scatter(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
    #                     lines_reddim[x*per_sino:(x+1)*per_sino:, 1])
    # plt.axis('off')
    # plt.savefig(f"{clic_dir}/2d_plot_1.pdf")
    ############## 2D binary
    all_comb = combinations(range(lines_reddim.shape[1]), 2)
    all_comb = [x for x in all_comb]
    fig, axs = plt.subplots(5, int(np.ceil(len(all_comb)/5)))
    for j in range(len(all_comb)):
        ax = axs[j % 5, int(np.ceil(j//5))]
        comb = all_comb[j]
        for x in range(len(gt_ids_bin)):
            if gt_ids_bin[x] == 0:
                ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, comb[0]],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, comb[1]], c='r', alpha=0.4)
            else:
                ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, comb[0]],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, comb[1]], c='b', alpha=0.4)
        ax.axis('off')
        ax.set_title(f"{comb}")
    fig.tight_layout()

    plt.savefig(f"{clic_dir}/2d_plot_binary.pdf")
    ############## 3D
    fig = plt.figure('3D')
    ax = fig.gca(projection='3d')

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
    ############## 3D small
    # fig = plt.figure('3d_small')
    # ax = fig.gca(projection='3d')

    # for x in range(5):
    #     ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
    #             lines_reddim[x*per_sino:(x+1)*per_sino:, 1],
    #             lines_reddim[x*per_sino:(x+1)*per_sino:, 2], alpha=0.8)
    plt.savefig(f"{clic_dir}/3d_plot_binary.pdf")


def batching(n, b_size):
    all_n = range(n)
    if b_size >= n or b_size == -1:
        return [all_n]
    n_batches = int(np.ceil(2*n/b_size) - 1)
    num_half = int(np.floor(b_size/2))
    batches = np.array([np.concatenate((random.sample(range(0, x), num_half), np.array(range(x, x+num_half)))) for x in range(num_half*2, n, num_half)])
    batches = np.concatenate(([range(0, num_half*2)], batches))
    if n % (num_half*2) != 0:
        max_n_arg = int(np.argwhere(batches[-1] == n))
        batches[-1] = np.concatenate((batches[-1, :max_n_arg], random.sample(range(0, b_size), num_half*2 - max_n_arg)))
    return batches


if __name__ == '__main__':
    start = time.time()

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    clic_dir = f'CLIC_Job_{time_stamp}'
    os.makedirs(clic_dir, exist_ok = True)

    part_locs, n = get_part_locs(args)  # arg in this list is g_id
    #batches = np.concatenate((batching(n), batching(n), batching(n), batching(n)))  # batches of g_id
    # batches = np.concatenate((batching(n, args.batch_size),  batching(n, args.batch_size), batching(n, args.batch_size)))  # batches of g_id
    batches = batching(n, args.batch_size)

    all_name_ids = []
    b = 0
    matrix = np.zeros((len(batches), n, args.num_clusters))
    for batch in batches:
        start_batch = time.time()
        batch_dir = f'CLIC_Job_{time_stamp}/batch_{b}'
        os.makedirs(batch_dir, exist_ok = True)
        print(f"### Running batch {b+1} of {len(batches)} with size {len(batch)} particles ###")

        all_ims, num, name_ids = sinogram_main(args, part_locs, batch)
        for name_id in name_ids:
            if name_id not in all_name_ids:
                all_name_ids.append(name_id)
        star_file  = star_writer.create(name_ids, batch_dir)

        args.num = num  # Update with lowest num
        lines_reddim, model = fitmodel(all_ims, args.model, args.num_comps)
        """ for generating figures for pca recon and eigenfilters """
        #import pca_recon
        #pca_recon.recon_sino(all_ims[-1], model, args)
        #pca_recon.plt_comps(model)
        #plt.show()

        # plot(lines_reddim, args.num, clic_dir, name_ids)

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

        batch_classes = clustering_main(lines_reddim, args, batch_dir, name_ids)
        matrix[b] = min_matrix.make_slice(batch_classes, batch, matrix.shape)
        print(f"   Batch time: {time.time() - start_batch:.2f}s")
        b += 1
    np.save("matrix.npy", matrix)
    aligned_matrix = min_matrix.align_batches(matrix)
    all_classes = min_matrix.make_line(aligned_matrix)
    # Score classes
    binary_test = False
    if binary_test == True:
	    ids_ints = clustering.ids_to_int(all_name_ids)
	    gt_ids_bin = [x % 4 for x in ids_ints]
	    np.save("gt_ids_bin.npy", gt_ids_bin)
	    score = clustering.score_bins(gt_ids_bin, all_classes, args)
	    print(f"Total_score: {score}")

    print(f"Total time: {time.time() - start:.2f}s")
    plt.show()
