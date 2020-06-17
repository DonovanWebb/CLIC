from sinogram_input import sinogram_main
from dim_red import fitmodel
from clustering import clustering_main
import plt_truth
import discrete
import sin_guesser


class Config():
    def __init__(self):
        self.dset = 'exp_plan'
        self.num = 16*10  #43
        self.snr = 1/(2**0)
        self.ds = 4
        self.num_comps = 3
        self.model = 'UMAP'


def plot(lines_reddim, num):
    import matplotlib.pyplot as plt
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
    plt.show()


'''
def rand_stats(lines):
    import numpy as np
    """ stats from randomly chosen lines """
    n = lines.shape[0]
    n_rand = 100
    rand = np.random.randint(n, size=n_rand)
    r_lines = lines[rand]
    print("random stats")
    plt_truth.get_stats(r_lines)
'''

# from ft_input import ft_main
# all_ims = ft_main(Config())
# from ftsino_input import ftsino_main
# all_ims = ftsino_main(Config())

import numpy as np
all_ims = sinogram_main(Config())
lines_reddim = fitmodel(all_ims, Config().model, Config().num_comps)
plot(lines_reddim, Config().num)

'''
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
