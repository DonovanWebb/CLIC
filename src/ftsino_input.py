import numpy as np
from skimage.transform import radon, resize
import mrcfile


def dsetpath(dataset):
    ''' return path to dataset '''
    # path_head = '/dls/ebic/data/staff-scratch/Donovan/'
    path_head = ''

    dsets = {'NN': '3Drepro/Radon/NN/proj_5angles/all/',
             'all': 'mvs/protein/all/',
             'noise': 'mvs/protein/all_noise/',
             'no e': 'mvs/protein/no_e/',
             'mixed': 'mvs/protein/mixed/',
             'mixed4': 'mvs/protein/mixed4/',
             'mixedLarge': 'mvs/protein/mixedLarge/',
             'mixedOff1': 'mvs/protein/offset/off1/',
             'mixedOff2': 'mvs/protein/offset/off2/',
             'mixedOff4': 'mvs/protein/offset/off4/',
             'mixedOff8': 'mvs/protein/offset/off8/',
             'ctfcor': 'mvs/protein/CTFYuriy/ctf_corrected/separated/',
             'tempall': 'mvs/recon/temp_for_slides/all/',
             'tempno_e': 'mvs/recon/temp_for_slides/no_e/',
             'tempmixed': 'mvs/recon/temp_for_slides/mixed/',
             'testlocal': '/home/lexi/Documents/Diamond/CLIC_refac/test_data/mixed/'}

    if dataset not in dsets:
        print('ERROR: dset not found')
        exit()

    return path_head + dsets[dataset]


def load_mrc(path):
    with mrcfile.open(path) as f:
        image = f.data.T
    return image


def stand_image(image):
    std = np.std(image)
    if std == 0:
        return image
    else:
        image_stand = (image - np.mean(image))/std
        return image_stand


def norm_image(image):
    minx = np.min(image)
    maxx = np.max(image)
    if maxx - minx == 0:
        return image
    else:
        image_norm = (image - minx)/(maxx - minx)
        return image_norm


def add_noise(image, snr=1):  # Try colored noise and shot noise
    ''' Add gaussian noise to data '''
    dims = tuple(image.shape)
    mean = 0
    sigma = np.sqrt(np.var(image)/snr)
    noise = np.random.normal(mean, sigma, dims)
    noisy_image = image + noise
    return noisy_image


def downscale(image, ds):
    image_resized = resize(image,
                           (image.shape[0] // ds, image.shape[1] // ds),
                           anti_aliasing=True)
    return image_resized


def circular_mask(im):
    '''
    Artefacts occur if do sinogram on unmasked particle in noise
    This mask fixes that
    '''
    h, w = im.shape
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask*im


def make_sinogram(image):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram.T


def find_im_size(path):
    im_path = path + '0.mrc'
    im = load_mrc(im_path)
    im_size = im.shape[0]
    return im_size


def ft_line(sino):
    n = sino.shape[0]
    if n % 2 == 0:
        ft_len = n//2 + 1
    else:
        ft_len = (n+1)//2
    ft_sino = np.zeros((sino.shape[0], ft_len))
    for i in range(sino.shape[0]):
        line = sino[i]
        ft = np.fft.rfft(line)
        ft[:5] = 0
        ft[-5:] = 0
        # phase = np.arctan(np.imag(ft)/np.real(ft))
        # magnitude = 100*np.log(np.abs(ft))
        # magnitude = stand_image(magnitude)
        # import matplotlib.pyplot as plt
        # plt.plot(ft)
        # plt.show()
        ft_sino[i] = np.real(ft)
    # import matplotlib.pyplot as plt
    # plt.imshow(ft_sino)
    # plt.show()

    return ft_sino


def stand_lines(all_ftsinos):
    all_ftsinos = np.reshape(all_ftsinos, (-1, all_ftsinos.shape[-1]))
    all_stand = np.zeros(all_ftsinos.shape)
    for i in range(all_ftsinos.shape[1]):
        col = all_ftsinos[:, i]
        col_stand = norm_image(col)
        all_stand[:, i] = col_stand

    # import matplotlib.pyplot as plt
    # for l in range(all_ftsinos.shape[0]):
    #     line = all_stand[l]
    #     plt.plot(line)
    #     plt.show()

    return all_stand


def ftsino_main(Config):
    dset_path = dsetpath(Config.dset)
    ds_size = find_im_size(dset_path) // Config.ds
    all_ftsinos = np.zeros((Config.num, ds_size, 76))
    for x in range(Config.num):
        im_path = dset_path + f'{x}.mrc'
        im = load_mrc(im_path)
        im = stand_image(im)
        im = add_noise(im, Config.snr)
        im = downscale(im, Config.ds)
        im = circular_mask(im)
        sino = make_sinogram(im)
        ft_sino = ft_line(sino)
        all_ftsinos[x] = ft_sino
    all_ftsinos = stand_lines(all_ftsinos)
    return all_ftsinos
