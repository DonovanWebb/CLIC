import numpy as np
from skimage.transform import radon, resize
import mrcfile
import cv2


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


def find_im_size(path):
    im_path = path + '0.mrc'
    im = load_mrc(im_path)
    im_size = im.shape[0]
    return im_size


def make_ft(im):
    f = np.fft.fft2(im)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    phase_spectrum = np.arctan(np.imag(fshift)/np.real(fshift))
    ft_im = phase_spectrum
    # import matplotlib.pyplot as plt
    # plt.imshow(ft_im)
    # plt.show()
    return ft_im


def make_lines(im):
    if im.shape[0] % 2 == 0:
        im = resize(im, (im.shape[0] - 1, im.shape[1] - 1), anti_aliasing=True)
    max_pix = im.shape[0]
    lx = max_pix//2
    theta = np.linspace(0, 2*np.pi, max_pix)
    line_im = np.zeros((max_pix, max_pix))
    for i in range(theta.shape[0]):
        th = theta[i]
        # Make a line with "num" points...
        x0, y0 = lx*np.cos(th) + max_pix/2, lx*np.sin(th) + max_pix/2  # These are pixel coordinates
        x1, y1 = -lx*np.cos(th) + max_pix/2, -lx*np.sin(th) + max_pix/2  # These are pixel coordinates
        length = max_pix
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

        # Extract the values along the line
        zi = im[x.astype(np.int), y.astype(np.int)]
        line_im[i] = zi

        #-- Plot...
        '''
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2)
        axes[0].imshow(im)
        axes[0].plot([x0, x1], [y0, y1], 'ro-')
        axes[0].axis('image')

        axes[1].plot(zi)

        plt.show()
        '''
    '''
    plt.imshow(line_im)
    plt.show()
    '''
    return line_im


def ft_main(Config):
    dset_path = dsetpath(Config.dset)
    ds_size = find_im_size(dset_path) // Config.ds - 1 # change
    all_ft = np.zeros((Config.num, ds_size, ds_size))
    for x in range(Config.num):
        im_path = dset_path + f'{x}.mrc'
        im = load_mrc(im_path)
        im = stand_image(im)
        im = add_noise(im, Config.snr)
        im = downscale(im, Config.ds)

        ft_im = make_ft(im)
        lines = make_lines(ft_im)
        all_ft[x] = lines

    return all_ft
