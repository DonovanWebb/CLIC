from sinogram_input import sinogram_main
from dim_red import fitmodel
from clustering import clustering_main


class Config():
    def __init__(self):
        self.dset = 'testlocal'
        self.num = 16
        self.snr = 1
        self.ds = 1
        self.num_comps = 2
        self.model = 'TSNE'


def plot(lines_reddim):
    import matplotlib.pyplot as plt
    plt.scatter(lines_reddim[:, 0], lines_reddim[:, 1])
    plt.show()


sinos = sinogram_main(Config())
lines_reddim = fitmodel(sinos, Config().model, Config().num_comps)
# plot(lines_reddim)
clustering_main(lines_reddim, Config())
