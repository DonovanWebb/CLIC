from sinogram_input import sinogram_main
from dim_red import fitmodel
from clustering import clustering_main


class Config():
    def __init__(self):
        self.dset = 'testlocal'
        self.num = 16*40
        self.snr = 1/4
        self.ds = 1
        self.num_comps = 10
        self.model = 'UMAP'


def plot(lines_reddim, num):
    import matplotlib.pyplot as plt
    per_sino = lines_reddim.shape[0] // num
    for x in range(num):
        if x % 2 == 0:
            plt.scatter(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, 1], c='r')
        else:
            plt.scatter(lines_reddim[x*per_sino:(x+1)*per_sino, 0],
                        lines_reddim[x*per_sino:(x+1)*per_sino:, 1], c='b')
    plt.show()


sinos = sinogram_main(Config())
lines_reddim = fitmodel(sinos, Config().model, Config().num_comps)
plot(lines_reddim, Config().num)
clustering_main(lines_reddim, Config())
