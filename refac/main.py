from sinogram_input import sinogram_main


class Config():
    def __init__(self):
        self.dset = 'testlocal'
        self.num = 10
        self.snr = 1
        self.ds = 1
        self.num_comps = 2


sinos = sinogram_main(Config)
print(sinos.shape)
