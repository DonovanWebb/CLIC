from sinogram_input import sinogram_main
from dim_red import fitmodel


class Config():
    def __init__(self):
        self.dset = 'testlocal'
        self.num = 10
        self.snr = 1
        self.ds = 1
        self.num_comps = 2
        self.model = 'TSNE'


def plot(sinos2d):
    import matplotlib.pyplot as plt
    plt.scatter(sinos2d[:, 0], sinos2d[:, 1])
    plt.show()


sinos = sinogram_main(Config())
sinos2d = fitmodel(sinos, Config().model, Config().num_comps)
plot(sinos2d)
