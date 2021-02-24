"""
input: Config from main.py
output: matrix containing sinograms from given dataset

This script loads projections, adds noise, masks, makes sinograms
"""
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
             'testlocal': '/home/lexi/Documents/Diamond/CLIC/CLIC_refac/test_data/mixed/',
             'SLICEM': '/home/lexi/Documents/Diamond/CLIC/CLIC_exp/SLICEM_exp/mixture_2D.mrcs',
             'exp_local': '/home/lexi/Documents/Diamond/CLIC/CLIC_refac/exp_plan/7_5_projs/',
             'JFrank': 'CLIC/dsets/simulated/JFrank/particles/',
             'JFrank1': 'CLIC/dsets/simulated/JFrank/particles1/',
             'JFrank2': 'CLIC/dsets/simulated/JFrank/particles2/',
             'JFrank3': 'CLIC/dsets/simulated/JFrank/particles3/',
             'JFrankd23': 'CLIC/dsets/simulated/JFrank/particlesd23/',
             'JFrankd41k': 'CLIC/dsets/simulated/JFrank/particlesd41k/',
             'fact': 'CLIC/dsets/real/fact/',
             'factmixed': 'CLIC/dsets/real/factmixed/',
             'exp_plan': 'CLIC/dsets/exp_plan/7_5_projs/'}

    if dataset not in dsets:
        print('ERROR: dset not found')
        exit()

    return path_head + dsets[dataset]


def load_mrc(path):
    with mrcfile.open(path) as f:
        image = f.data
    return image


def stand_image(image):
    image_stand = (image - np.mean(image))/np.std(image)

    return image_stand


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


def make_sinogram(image, nlines=120):
    theta = np.linspace(0., 360., nlines, endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return sinogram.T


def find_im_size(path):
    im_path = path + '0.mrc'
    im = load_mrc(im_path)
    im_size = im.shape[0]
    return im_size

def gblur(im):
    ''' Adds gaussian blur to projection '''
    import cv2
    kernel = 5
    im = cv2.GaussianBlur(im, (kernel, kernel),0)
    return im


def sinogram_main(Config):
    dset_path = dsetpath(Config.dset)
    nlines = 120  # Number of single lines in sinogram (120 will be 3 degree interval)

    # CLEAN
    if dset_path.endswith('.mrcs'):
        ''' Open mrcs class file '''
        first_pass = True
        with mrcfile.open(dset_path) as f:
            classes = f.data
        for x in range(Config.num):
            im = classes[x]
            if first_pass:
                ds_size = im.shape[0] // Config.ds
                all_sinos = np.zeros((Config.num, nlines, ds_size))
                first_pass = False
            # im = stand_image(im)
            # im = add_noise(im, Config.snr)
            im = downscale(im, Config.ds)
            im = stand_image(im)
            im = circular_mask(im)
            sino = make_sinogram(im, nlines)
            all_sinos[x] = sino
        return all_sinos

    else:
        ''' Open individual particle images (numbered 1->N) '''
        first_pass = True
        for x in range(Config.num):
            im_path = dset_path + f'{x}.mrc'
            im = load_mrc(im_path)
            if first_pass:
                ds_size = im.shape[0] // Config.ds
                all_sinos = np.zeros((Config.num, nlines, ds_size))
                first_pass = False
            im = stand_image(im)
            im = add_noise(im, Config.snr)

            import matplotlib.pyplot as plt
            plt.imshow(im, cmap='gray')
            plt.axis('off')
            plt.show()

            im = downscale(im, Config.ds)
            im = circular_mask(im)
            sino = make_sinogram(im, nlines)

            import matplotlib.pyplot as plt
            plt.imshow(sino, cmap='gray')
            plt.axis('off')
            plt.show()

            all_sinos[x] = sino
        return all_sinos
