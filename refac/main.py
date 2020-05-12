from sinogram_input import sinogram_main
from dim_red import fitmodel
from clustering import clustering_main


class Config():
    def __init__(self):
        self.dset = 'testlocal'
        self.num = 16*4
        self.snr = 1
        self.ds = 1
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


from ft_input import ft_main
# all_ims = ft_main(Config())
from ftsino_input import ftsino_main
all_ims = ftsino_main(Config())
# all_ims = sinogram_main(Config())
lines_reddim = fitmodel(all_ims, Config().model, Config().num_comps)
plot(lines_reddim, Config().num)
clustering_main(lines_reddim, Config())
