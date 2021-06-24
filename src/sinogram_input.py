"""
input: Config from main.py
output: matrix containing sinograms from given dataset

This script loads projections, adds noise, masks, makes sinograms
"""
import numpy as np
from skimage.transform import radon, resize
import mrcfile
import gemmi


'''
Example dsets and locations...
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
            'testlocal': '/home/lexi/Documents/Diamond/CLIC/test_data/mixed/',
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
'''


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


def pre_process(im, config):
    #im = stand_image(im)
    if config.snr != -1:
        im = add_noise(im, config.snr)
    # optional displaying (for debug)
    """
    import matplotlib.pyplot as plt
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.show()
    """
    im = downscale(im, config.down_scale)
    #im = stand_image(im)
    im = circular_mask(im)
    import matplotlib.pyplot as plt  # just for figure
    plt.figure("original image")  # just for figure
    plt.imshow(im, "gray")  # just for figure
    sino = make_sinogram(im, config.nlines)
    '''
    import matplotlib.pyplot as plt
    plt.imshow(sino, cmap='gray')
    plt.axis('off')
    plt.show()
    '''
    return sino


def sinogram_main(config):
    dset_path = config.data_set

    if dset_path.endswith('.mrcs'):
        with mrcfile.open(dset_path) as f:
            classes = f.data
            n_max = classes.shape[0]
    elif dset_path.endswith('mrc'):
        import glob
        all_files = glob.glob(dset_path)
        n_max = len(all_files)
        if all_files == []:
            print(f"Error: No mrc found in: {dset_path}")
            exit()
    elif dset_path.endswith('star'):
        # read star file to extract im locs
        starfile = gemmi.cif.read_file(dset_path)
        block = starfile.find_block('particles')
        locations = [x for x in block.find_values(f'_rlnimagename')]
        n_max = len(locations)
        # need error handling here


    else:
        print(f"Error: Invalid path specification: {dset_path}")
        exit()

    n = min(n_max, config.num)
    #print(f"Will use {n} particles")

    ids = []
    do_subset_test = False
    i = 0  # for subset test
    perc = 0.8  # for subset test
    c0 = 0
    c1 = 0
    for x in range(n):

        if dset_path.endswith('.mrcs'):
            im = classes[x]
            ids.append(f'{x+1}@{dset_path}')
        elif dset_path.endswith('mrc'):
            if do_subset_test == True:
                # subset test:
                import os
                while True:
                    im_path = all_files[x+i]
                    im_class = os.path.basename(im_path)
                    im_class = int(os.path.splitext(im_class)[0])
                    rand = np.random.rand(1)
                    if rand > perc and im_class % 2 == 0:
                        c0 += 1
                        break
                    elif rand < perc and im_class % 2 == 1:
                        c1 += 1
                        break
                    else:
                        i += 1
            else:
                    im_path = all_files[x]
            im = load_mrc(im_path)
            ids.append(f'{im_path}')
        elif dset_path.endswith('star'):
            im_loc = locations[x]
            (ind, stack_loc) = im_loc.split('@')
            stack = load_mrc(stack_loc)
            # if only one im present in stack
            if stack.ndim == 2:
                im = stack 
            else:
                im = stack[int(ind) - 1]  # Rln stack starts at 1!
            ids.append(f'{im_loc}')

        if x == 0:  # first pass makes all_sinos
            ds_size = im.shape[0] // config.down_scale
            all_sinos = np.zeros((n, config.nlines, ds_size))

        sino = pre_process(im, config)
        all_sinos[x] = sino

    print(c0/n, c1/n)
    return all_sinos, n, ids
