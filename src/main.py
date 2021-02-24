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
import plt_truth
import discrete
import sin_guesser


class Config():
    def __init__(self):

        ''' Dataset to be considered for clustering. To see all options and
        format needed refer to sinogram_input.py script '''
        self.dset = 'testlocal'

        ''' Number of projections to consider. A multiple of 16 is used as this
        makes displaying output cleaner and for easy scoring of a two class system.
        (TODO A few functions will need to change to accomodate arbitary number)'''
        self.num = 100

        ''' Signal to noise ratio of projection before making sinograms '''
        self.snr = 1/(2**1)

        ''' Downscaling of image prior to making sinograms '''
        self.ds = 1

        ''' Number of components of dimensional reduction technique '''
        self.num_comps = 10

        ''' Dimensional reduction technique.
        options are: PCA, UMAP, TSNE, LLE, ISOMAP, MDS, TRIMAP '''
        self.model = 'UMAP'


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


'''
"""
Optional code - for Fourier lines instead of real space sinogram lines
"""
from ft_input import ft_main
all_ims = ft_main(Config())
from ftsino_input import ftsino_main
all_ims = ftsino_main(Config())
'''

all_ims = sinogram_main(Config())
lines_reddim = fitmodel(all_ims, Config().model, Config().num_comps)
plot(lines_reddim, Config().num)

'''
"""
Optional code - for experiment looking at how common line group spreads with noise
"""
r_lines = discrete.rand_lines(lines_reddim, n_rand=100)
discrete.get_stats(r_lines)
# r = 17.5
all_groups = sin_guesser.main(Config().num, r)
theta = 3
th_lines = sin_guesser.choose_rand_group(all_groups)

group_lines = discrete.get_discrete_lines(lines_reddim, th_lines, r, theta)
discrete.get_stats(group_lines)
discrete.plot(group_lines, lines_reddim)
'''

clustering_main(lines_reddim, Config())
