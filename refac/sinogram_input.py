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
             'testlocal': '~/Documents/Diamond/CLIC_refac/test_data/'}

    if dataset not in dsets:
        print('ERROR: dset not found')
        exit()

    return path_head + dsets[dataset]


def load_mrc(path):
    with mrcfile.open(path) as f:
        image = f.data.T
    return image


def norm_image(image):
    image_norm = (image - np.mean(image))/np.std(image)
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


def sinogram_main(Config):
    dset_path = dsetpath(Config.dset)
    ds_size = find_im_size(dset_path) // Config.ds
    all_sinos = np.zeros((Config.num, ds_size, ds_size))
    for x in range(Config.num):
        im_path = dset_path + f'{x}.mrc'
        im = load_mrc(im_path)
        im = norm_image(im)
        im = add_noise(im, Config.snr)
        im = downscale(im, Config.ds)
        im = circular_mask(im)
        sino = make_sinogram(im)
        all_sinos[x] = sino
    return all_sinos
