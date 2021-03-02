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


parser = argparse.ArgumentParser()

t = ''' Dataset to be considered for clustering. Input path to mrcs
stack or to individual mrc particles with "/PATH/TO/PARTS/*.mrc"
(Don't forget "")
'''
parser.add_argument("-i", "--data_set", help=t, required=True, type=str)

t = ''' Number of projections to consider. A multiple of 16 is used as this
makes displaying output cleaner and for easy scoring of a two class system.
(TODO A few functions will need to change to accomodate arbitary number)'''
parser.add_argument("-n", "--num", help=t, default=1000, type=int)

t = ''' Signal to noise ratio of projection before making sinograms '''
parser.add_argument("-r", "--snr", help=t, default=1, type=float)

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

args = parser.parse_args()



def plot(lines_reddim, num):
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

    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    for x in range(num):
        if x < 160:
            if x % 2 == 0:
                ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, 1],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, 2], c='r', alpha=0.4)
            else:
                ax.plot(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, 1],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, 2], c='b', alpha=0.4)
    plt.show()


if __name__ == '__main__':

    all_ims, num, ids = sinogram_main(args)
    star_file  = star_writer.create(ids)
    
    args.num = num  # Update with lowest num
    lines_reddim = fitmodel(all_ims, args.model, args.num_comps)
    plot(lines_reddim, args.num)

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

    clustering_main(lines_reddim, args, star_file)
