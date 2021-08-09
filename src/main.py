#!/usr/bin/env python
'''
Main file for CLIC 3D heterogeneity sorting algorithm.  Config options
are passed by arguments and the job pipeline to be run can be seen at
the bottom of the file.

Results: A dendrogram of each batch is made.  Output of star files
with each particle assigned to a cluster.
'''
import numpy as np
from sinogram_input import sinogram_main
from sinogram_input import get_part_locs
from dim_red import fitmodel
from clustering import clustering_main
from CLIC_plot import plot
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

# To silence deprecation warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()

t = ''' Dataset to be considered for clustering. Input path to mrcs
stack, to individual mrc particles, or to particle starfile with "/PATH/TO/PARTS/*.mrc"
(Notes: 1. Don't forget "", 
        2. if star file: run from relion home dir
        3. expects *.mrc or path/to/file.mrcs or path/to/file.star only. 
'''
parser.add_argument("-i", "--data_set", help=t, required=True, type=str)

t = ''' Number of projections to consider total. Defaults to 1000 '''
parser.add_argument("-n", "--num", help=t, default=1000, type=int)

t = ''' Batchsize - runs overlapping batches of provided size. This
speeds up process and requires less memory. Recommended batch size is
750 < b < 2000'''
parser.add_argument("-b", "--batch_size", help=t, default=-1, type=int)

t = ''' Downscaling of image prior to making sinograms '''
parser.add_argument("-d", "--down_scale", help=t, default=2, type=int)

t = ''' Number of components of dimensional reduction technique. This
requires some experimentation '''
parser.add_argument("-c", "--num_comps", help=t, default=10, type=int)

t = ''' Dimensional reduction technique.
options are: PCA, UMAP, TSNE, LLE, ISOMAP, MDS, TRIMAP.
Recommended: UMAP and PCA. '''
parser.add_argument("-m", "--model", help=t, default='UMAP', type=str)

t = ''' Number of lines in one sinogram (shouldn't need to change
recommended=120)'''
parser.add_argument("-l", "--nlines", help=t, default=120, type=int)

t = ''' Run on gpu with CUDA '''
parser.add_argument("-g", "--gpu", help=t, default=False, action='store_true')

t = ''' Number of clusters '''
parser.add_argument("-k", "--num_clusters", help=t, default=2, type=int)

t = ''' For testing: Signal to noise ratio to be applied to projection
before making sinograms '''
parser.add_argument("-r", "--snr", help=t, default=-1, type=float)

args = parser.parse_args()


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

        """
        Optional code 
        """
        """ 
        for generating figures for pca recon and eigenfilters 
        """
        """
        import pca_recon
        pca_recon.recon_sino(all_ims[-1], model, args)
        pca_recon.plt_comps(model)
        plt.show()
        plot(lines_reddim, args.num, clic_dir, name_ids)
        """

        """ 
        for experiment looking at how common line group spreads with noise
        """
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
        """

        batch_classes = clustering_main(lines_reddim, args, batch_dir, name_ids)
        matrix[b] = min_matrix.make_slice(batch_classes, batch, matrix.shape)
        print(f"   Batch time: {time.time() - start_batch:.2f}s")
        b += 1

    np.save(f"{clic_dir}matrix.npy", matrix)
    aligned_matrix = min_matrix.align_batches(matrix)
    all_classes = min_matrix.make_line(aligned_matrix)

    # Score classes in testing. input alternates between classes. i.e. classes  0101010101...
    binary_test = False
    if binary_test == True:
	    ids_ints = clustering.ids_to_int(all_name_ids)
	    gt_ids_bin = [x % args.num_clusters for x in ids_ints]
	    np.save("gt_ids_bin.npy", gt_ids_bin)
	    score = clustering.score_bins(gt_ids_bin, all_classes, args)
	    print(f"Total_score: {score}")

    print(f"Total time: {time.time() - start:.2f}s")
    plt.show()
