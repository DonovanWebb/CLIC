import os
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import mrcfile as mrc
import cv2
import imageio
from kneed import KneeLocator


def lowpass(img):
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)     # center

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 1), np.uint8)
    lowfilter = 60
    mask[crow-lowfilter:crow+lowfilter, ccol-lowfilter:ccol+lowfilter] = 1
    plt.imshow(mask)

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    return img_back


def knee_thresh(entr_im):
    flat_im = entr_im.ravel()
    ordered_flat = np.sort(flat_im)
    tot_area = len(ordered_flat)
    kn = KneeLocator(ordered_flat, range(tot_area), curve='concave', direction='increasing')
    inv_norm = ordered_flat[-1] - ordered_flat[0]
    norm_thresh = kn.x_difference[np.argmax(kn.y_difference)]
    thresh = ordered_flat[0] + norm_thresh*inv_norm
    plt.figure("knee")
    plt.plot(ordered_flat, range(tot_area))
    plt.xlabel("Entropy")
    plt.ylabel("Cumulative frequency")
    plt.plot([thresh]*2, [0, tot_area], c='r')
    plt.savefig(f'knee.png', bbox_inches='tight')
    return thresh


def calc_thresh(entr_im, mask_perc=0.7):
    flat_im = entr_im.ravel()
    tot_area = len(flat_im)
    mask_area = tot_area * mask_perc
    ordered_flat = np.sort(flat_im)
    thresh = ordered_flat[int(np.ceil(mask_area))]
    print(thresh)
    plt.plot(ordered_flat, range(tot_area))
    return thresh


def add_noise(image, snr=1):  # Try colored noise and shot noise
    ''' Add gaussian noise to data '''
    if snr == -1:
        return image
    dims = tuple(image.shape)
    mean = 0
    sigma = np.sqrt(np.var(image)/snr)
    noise = np.random.normal(mean, sigma, dims)
    noisy_image = image + noise
    return noisy_image


def post_process_mask(thresh_im):
    small = int(thresh_im.shape[0] * 0.025)
    big = int(thresh_im.shape[0] * 0.05)
    kernelSmall = np.ones((small, small), np.uint8)
    kernelBig = np.ones((big, big), np.uint8)
    # opening = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernelSmall)
    # dilation = cv2.dilate(opening, kernelBig, iterations=1)
    # closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernelSmall)
    dilation = cv2.dilate(thresh_im, kernelSmall, iterations=10)
    thresh = dilation.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(thresh, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # remove if cont too large or too small
    # To Do
    cont_size = np.array([cv2.contourArea(c) for c in contours])

    # central contour:
    mid_pnts = np.array([np.array([sum(c[:,:,0]), sum(c[:,:,1])]) / len(c) for c in contours])
    from_cent = np.array([np.sqrt((p[0] - thresh_im.shape[0]/2)**2 + (p[1]-thresh_im.shape[0]/2)**2) for p in mid_pnts])
    cent_cont = contours[np.argmin(from_cent)]
    mask = np.zeros_like(thresh_im)
    #mask = cv2.fillPoly(mask, cent_cont, 1)
    mask = cv2.drawContours(mask, [cent_cont], 0, (255), cv2.FILLED)

    return mask

def watershed_m(image):
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    return labels
    
def main(im_orig, noise=-1):
    dim = im_orig.shape[0]
    #im_orig = cv2.resize(im_orig, (dim // 4, dim // 4))
    #im_orig = lowpass(im_orig)
    image = add_noise(im_orig, noise)
    maxpix = max(image.max(), -1*image.min())
    image = image/maxpix
    disk_size = int(image.shape[0] * 0.175)
    entr_im = entropy(image, disk(disk_size))

    #thresh = calc_thresh(entr_im)
    thresh = knee_thresh(entr_im)
    ret, thresh_im = cv2.threshold(entr_im, thresh, 255, cv2.THRESH_BINARY)

    # fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(10, 4))
    # ax0.imshow(image, cmap='gray')
    # ax0.set_xlabel("Noisy image")

    # ax1.imshow(im_orig, cmap='gray')
    # ax1.set_xlabel("Clean image")

    # ax2.imshow(entr_im, cmap='viridis')
    # ax2.set_xlabel("Local entropy")

    # ax3.imshow(thresh_im, cmap='gray')
    # ax3.set_xlabel("Thresholded entropy")
    # fig.tight_layout()

    mask = post_process_mask(thresh_im)
    # mask_water = watershed_m(mask)
    average = cv2.mean(im_orig)[0]
    masked = np.multiply(im_orig, mask/255)
    # masked[masked == 0] = average

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

    ax0.imshow(im_orig, cmap='gray')
    ax0.set_xlabel("Clean image")

    ax1.imshow(image, cmap='jet')
    ax1.set_xlabel("Noisy image")

    ax2.imshow(entr_im, cmap='viridis')
    ax2.set_xlabel("Local entropy")

    ax3.imshow(masked, cmap='gray')
    ax3.imshow(mask, alpha=0.2, cmap='inferno_r')
    ax3.set_xlabel("Thresholded entropy")
    plt.savefig(f'entr_masking{np.random.randint(10)}.png', bbox_inches='tight')

    # plt.show()
    return masked


if __name__ == "__main__":
    im_dir = '/home/lexi/Documents/Diamond/CLIC/test_data/mixed/'
    im_dir = '/home/lexi/Documents/Diamond/entropy/classes/'
    im_dir = '/home/lexi/Documents/CLIC/entropy_filter/particles/'
    im_dir = '/home/lexi/Documents/CLIC/test_data/mixed/'

    for x in os.listdir(im_dir):
        if x.endswith('.mrc'):
            im_path = os.path.join(im_dir, x)
            with mrc.open(im_path) as f:
                im_orig = f.data
            main(im_orig)
        elif x.endswith('.mrcs'):
            im_path = os.path.join(im_dir, x)
            with mrc.open(im_path) as f:
                ims_orig = f.data
            print(ims_orig.shape)
            for im_orig in ims_orig:
                main(im_orig)
