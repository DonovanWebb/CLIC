from sklearn.decomposition import PCA
import numpy as np
from sklearn.feature_extraction import image
from skimage.transform import radon, resize, iradon
from skimage.transform import iradon_sart
import matplotlib.pyplot as plt
import cv2


def plt_comps(mapper):

    comps = mapper.components_

    plt.figure('Components')
    for i in range(1, comps.shape[0]):
        plt.subplot(comps.shape[0]//10, 10, i)
        plt.plot(comps[i-1])
        plt.xticks([])
        plt.yticks([])
    plt.savefig('components.png', bbox_inches='tight')



def recon_sino(im, mapper, config):
    theta = np.linspace(0., 360., config.nlines, endpoint=False)
    # sino = radon(im, theta=theta).T
    sino = im

    if config.model == "PCA":
        comps = mapper.components_
        sino_pca = mapper.transform(sino)
        sino_recon = np.dot(sino_pca[:,:], comps[:,:])

    elif config.model == "UMAP":
        sino_umap = mapper.transform(sino)
        sino_recon = mapper.inverse_transform(sino_umap)

    im_recon_fbp = iradon(sino_recon.T, theta=theta, filter_name='ramp')
    reconstruction_sart = iradon_sart(sino_recon.T, theta=theta)
    reconstruction_sart2 = iradon_sart(sino_recon.T, theta=theta,
                                    image=reconstruction_sart)
    reconstruction_sart3 = iradon_sart(sino_recon.T, theta=theta,
                                    image=reconstruction_sart2)

    plt.figure("original")
    plt.imshow(sino, "gray")
    plt.axis('off')
    plt.savefig(f'snr{config.snr}_sino.png', bbox_inches='tight')

    plt.figure("recon")
    plt.imshow(sino_recon, "gray")
    plt.axis('off')
    plt.savefig(f'snr{config.snr}_model{config.model}_comp{config.num_comps}_recon_sino.png', bbox_inches='tight')

    # plt.figure("image")
    # plt.imshow(im, "gray")
    plt.figure("recon fbp image")
    plt.imshow(im_recon_fbp, "gray")
    plt.axis('off')
    plt.savefig(f'snr{config.snr}_model{config.model}_comp{config.num_comps}_recon_fbp.png', bbox_inches='tight')
    plt.figure("recon sart1 image")
    plt.imshow(reconstruction_sart, "gray")
    plt.axis('off')
    plt.figure("recon sart3 image")
    plt.imshow(reconstruction_sart3, "gray")
    plt.axis('off')
    plt.savefig('recon_sart3.png', bbox_inches='tight')
    plt.savefig(f'snr{config.snr}_model{config.model}_comp{config.num_comps}_recon_sart3.png', bbox_inches='tight')


    # for i in range(10):
    #     plt.plot(sino[i], "red")
    #     plt.plot(sino_recon[i], "green")
    #     plt.show()


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


if __name__  == "__main__":
    dset = []
    num = 500
    for i in range(num):
        r1 = np.random.randint(1000)
        r2 = np.random.random()
        r3 = np.random.random()
        time = np.arange(r1, r1+240, 1)*0.1*r2
        dset += list(np.sin(time)*r3)
    for i in range(len(dset)):
        if np.random.randint(2) == 0:
            dset[i] += np.random.random(1)/10
        else:
            dset[i] -= np.random.random(1)/10
    dset = np.reshape(dset, (num, -1))


    subset = dset[:10]
    im = plt.imread("/home/lexi/Pictures/ed_edits/ed_wizard.png")[:,:,0]
    im = cv2.resize(im, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    im = circular_mask(im)
    theta = np.linspace(0., 180., max(im.shape), endpoint=False)
    sino = radon(im, theta=theta)

    pca = PCA(random_state = 41)

    dset_pca = pca.fit_transform(sino)

    plt_comps(pca)
    recon_sino(im, pca)
