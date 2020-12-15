import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
# from sklearn.manifold import MDS
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
import mrcfile
from skimage.transform import radon, rescale, resize, downscale_local_mean
import time


def create_circular_mask(h, w, center=None, radius=None, soft_edge=False):
    '''
    Artefacts occur if do sinogram on unmasked particle in noise
    This soft edged mask fixes this
    '''
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    #soft_edge
    if soft_edge:
        radius = radius - 7
        dist_adj = (1/(dist_from_center+0.001-radius))
        dist_adj[dist_adj < 0] = 1
        dist_adj[dist_adj > 1] = 1
        inds = np.where(np.isnan(dist_adj))
        dist_adj[inds] = np.take(1, inds[1])
        return dist_adj**1.5
    else:
        mask = dist_from_center <= radius
        return mask


def norm_image(image):
    ''' Make conv's life easier by normalising input '''
    image_norm = (image - np.mean(image))/np.std(image)
    return image_norm

    
def norm_sinos(s):
    n_s = np.array([norm_image(l) for l in s])
    return n_s


def downscale(image, ds):
    # image_rescaled = rescale(image, 1/ds, anti_aliasing=False)
    image_resized = resize(image, (image.shape[0] // ds, image.shape[1] // ds),
                       anti_aliasing=True)
    # image_downscaled = downscale_local_mean(image, (ds, ds))
    # print(image_rescaled.shape)
    # print(image_resized.shape)
    # print(image_downscaled.shape)
    # exit()
    return image_resized

def sino(image,num=1,sin_ds=1):
    ''' Make sinogram from image '''

    # image = downscale(image, sin_ds)

    mask1 = create_circular_mask(image.shape[0], image.shape[0], soft_edge=True)
    mask2 = create_circular_mask(image.shape[0], image.shape[0])
    mask = mask2
    masked = mask*image
    # theta = np.linspace(90., 450., int(max(image.shape)*(1/sin_ds)), endpoint=False) # CHANGE
    theta = np.linspace(0., 180., max(image.shape)//sin_ds, endpoint=False)
    # theta = np.linspace(90., 450., int(max(image.shape)), endpoint=False)
    sinogram = radon(masked, theta=theta, circle=True)
    sinogram_mask = radon(mask, theta=theta, circle=True)
    plt.figure(num)
    plt.imshow(sinogram, cmap='gray')
    plt.figure(num+10)
    plt.imshow(masked, cmap='gray')
    # plt.show()
    return sinogram.T


def load_mrc(path):
    with mrcfile.open(path) as f:
        image = f.data.T
    return image


def update_annot(ind,sc,annot):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([str(names[n]) for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    if event.inaxes == ax1:
        vis = annot1.get_visible()
        cont, ind = sc1.contains(event)
        if cont:
            update_annot(ind,sc1,annot1)
            annot1.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot1.set_visible(False)
                fig.canvas.draw_idle()

    elif event.inaxes == ax2:
        vis = annot2.get_visible()
        cont, ind = sc2.contains(event)
        if cont:
            update_annot(ind,sc2,annot2)
            annot2.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot2.set_visible(False)
                fig.canvas.draw_idle()

    elif event.inaxes == ax3:
        vis = annot3.get_visible()
        cont, ind = sc3.contains(event)
        if cont:
            update_annot(ind,sc3,annot3)
            annot3.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot3.set_visible(False)
                fig.canvas.draw_idle()

                
def add_sin(mrc_name,i,sinos,snr=4,sin_ds=1):
    mrc = load_mrc(mrc_name)
    mrc = norm_image(mrc)
    noise_mrc = add_noise(mrc,snr)
    sino1 = sino(noise_mrc,i,sin_ds)
    n_sino = norm_sinos(sino1)
    sinos.append(sino1)
    return sinos


def add_noise(image, snr=1):  # Try colored noise and shot noise
    ''' Add gaussian noise to data '''
    dims = tuple(image.shape)
    mean = 0
    sigma = np.sqrt(np.var(image)*snr)
    noise = np.random.normal(mean, sigma, dims)
    noisy_image = image + noise
    norm_noisy = norm_image(noisy_image)
    return norm_noisy

    
def plot_prince(fig,x, i,red_comp,colors):
    ax = fig.add_subplot(2,np.ceil(x/2),i) 
    ax.set_xlabel('Principal Component 0', fontsize = 15)
    ax.set_ylabel(f'Principal Component {i}', fontsize = 15)
    sc = ax.scatter(red_comp[:,0],
                red_comp[:,i],
                s = 50,
                c = colors   )
    ax.grid()

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    return fig,ax,sc,annot
    

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2), min(dist_2)
    

def group_line(nodes, grp_n=3):
    kernel = np.ones(grp_n,dtype=int)
    grpd_nodes = np.ones((nodes.shape[0]-grp_n+1,nodes.shape[1]))
    for l in range(nodes.shape[1]):
        grpd_nodes[:,l] = np.convolve(nodes[:,l],kernel,mode='valid')
    return grpd_nodes
    

def test(sin1,sin2,grp):
    min_dists = []
    ar_mins = []
    for n in range(sin1.shape[0]//2+1):
        ar_min, min_dist = closest_node(sin1[n,:], sin2)
        # print(f'{n}  --> {ar_min},  {min_dist}')
        min_dists.append(min_dist)
        ar_mins.append(ar_min)
    min_arg = np.argmin(min_dists)
    return (min_arg+(grp-1)//2, ar_mins[min_arg]+(grp-1)//2)


def good_match(NN, sin_size):
    if (NN[0] <= 0.6*sin_size and NN[0] >= 0.4*sin_size) or (NN[0] >= 0 and NN[0] <= 0.1*sin_size):
        if (NN[1] <= 0.6*sin_size and NN[1] >= 0.4*sin_size) or (NN[1] >= 0 and NN[1] <= 0.1*sin_size):
            return True
    return False


def add_bg_sins(num,snr,sin_ds,num_pca,dataset='NN'):
    start_adding = time.time()
    if num == 0:
        return []
    sinos = []
    for i in range(100, num + 100):
        if dataset == 'NN':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/all/{i}.mrc',3,sinos,snr,sin_ds)
        elif dataset == 'all':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/all/{i}.mrc',3,sinos,snr,sin_ds)  # full protein
        elif dataset == 'noise':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/all_noise/{i}.mrc',3,sinos,snr,sin_ds)  # full protein
        elif dataset == 'no e':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/no_e/{i}.mrc',3,sinos,snr,sin_ds)  # no e
        elif dataset == 'mixed':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/mixed/{i}.mrc',3,sinos,snr,sin_ds)  # mixed
        elif dataset == 'mixed4':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/mixed4/{i}.mrc',3,sinos,snr,sin_ds)  # mixed4
        elif dataset == 'mixedLarge':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/mixedLarge/{i}.mrc',3,sinos,snr,sin_ds)  # mixed4
        elif dataset == 'mixedOff1':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/offset/off1/{i}.mrc',3,sinos,snr,sin_ds)  # off1
        elif dataset == 'mixedOff2':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/offset/off2/{i}.mrc',3,sinos,snr,sin_ds)  # off2
        elif dataset == 'mixedOff4':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/offset/off4/{i}.mrc',3,sinos,snr,sin_ds)  # off4
        elif dataset == 'mixedOff8':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/offset/off8/{i}.mrc',3,sinos,snr,sin_ds)  # off8
        elif dataset == 'ctfcor':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/protein/CTFYuriy/ctf_corrected/separated/{i}.mrc',3,sinos,snr,sin_ds)  # ctf corrected
        elif dataset == 'tempall':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/recon/temp_for_slides/all/{i}i.mrc',3,sinos,snr,sin_ds)
        elif dataset == 'tempno_e':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/recon/temp_for_slides/no_e/{i}i.mrc',3,sinos,snr,sin_ds)
        elif dataset == 'tempmixed':
            sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/mvs/recon/temp_for_slides/mixed/{i}.mrc',3,sinos,snr,sin_ds)
        else:
            print('Options are "NN" "all" "no e" and "mixed"')
            exit()

    pre_sinos = np.vstack(sinos)

    # #unspplit...
    # pre_sinos = np.array(sinos)
    # pre_sinos = np.reshape(pre_sinos,(num,-1))

    end_adding = time.time()
    print(f'Time to add sinos : {end_adding-start_adding}')

    ### LINEAR ###
    # model = PCA(n_components=num_pca)
    ### MANIFOLDS ###
    # model = Isomap(n_components=num_pca)
    # model = LLE(n_components=num_pca, n_neighbors=5)
    # model = MDS(n_components=num_pca)
    model = TSNE(n_components=num_pca)

    start_train = time.time()
    sinos_trans = model.fit_transform(pre_sinos)
    end_train = time.time()

    comp_vects = []
    for x in sinos_trans.T:
        comp_vects.append(np.var(x))
    plt.figure(20)
    plt.plot(comp_vects)
    # plt.show()

    print(f'Time to Train model : {end_train-start_train}')
    return sinos_trans, model


def main(model,sin_ds=1, snr=1, num_pca=20, grouping=1):
    start_sining = time.time()
    sinos = []
    to_test = np.random.randint(100)
    sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/0/{to_test}.mrc',1,sinos,snr,sin_ds)
    sinos = add_sin(f'/dls/ebic/data/staff-scratch/Donovan/3Drepro/Radon/NN/proj_5angles/1/{to_test}.mrc',2,sinos,snr,sin_ds)

    all_sino = np.vstack((sinos))
    len_sin = sinos[0].shape[0]
    end_sining = time.time()
    # print(f'sining : {end_sining-start_sining}')

    # print('Nearest Neighbour')
    NN = test(all_sino[:len_sin,:],all_sino[len_sin:2*len_sin,:],1)

    # print('Nearest Neighbour and grouped')
    grpd_1 = group_line(all_sino[:len_sin],grouping)
    grpd_2 = group_line(all_sino[len_sin:len_sin*2],grouping)
    NN_G = test(grpd_1, grpd_2,grouping)


    names = np.array(list(range(0,all_sino.shape[0])))
    
    colors = list(names)
    c_opts = ['r','g','b','y','m']
    for c in range(len(names)):
        colors[c] = c_opts[c//(len(names)//len(sinos))]

    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn

    # all_sino = StandardScaler().fit_transform(all_sino)

    no_prince = 1

    red_comp = model.transform(all_sino)
    
    comp_vects = []
    for x in red_comp.T:
        comp_vects.append(np.var(x))

    plt.figure(20)
    plt.plot(comp_vects)
    plt.show()

    # print('Principal')
    PC = test(red_comp[:len_sin],red_comp[len_sin:len_sin*2],1)
    # print('Principal and grouped')
    grpd_sino1 = group_line(red_comp[:len_sin],grouping)
    grpd_sino2 = group_line(red_comp[len_sin:len_sin*2],grouping)
    PC_G = test(grpd_sino1,grpd_sino2,grouping)
    # print(f'{NN} {NN_G} {PC} {PC_G}')

    '''
    fig = plt.figure(figsize = (8,8))
    fig,ax1,sc1,annot1 = plot_prince(fig,no_prince, 1,red_comp,colors)
    # fig,ax2,sc2,annot2 = plot_prince(fig,no_prince, 2,red_comp,colors)
    # fig,ax3,sc3,annot3 = plot_prince(fig,no_prince, 3,red_comp,colors)
    # fig,ax4,sc4,annot4 = plot_prince(fig,no_prince, 4,red_comp,colors)
    # fig,ax5,sc5,annot5 = plot_prince(fig,no_prince, 5,red_comp,colors)

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

    '''

    return(good_match(NN,len_sin),good_match(NN_G,len_sin), good_match(PC,len_sin), good_match(PC_G,len_sin))
    
if __name__ == '__main__':
    # config
    sin_ds = 2
    snr = 32
    num_pca = 10
    grouping = 3
    

    # print('NN','NN_G','PC','PC_G')
    for i in range(5):
        results = np.array([])
        repeats = 500
        pre_sinos,model = add_bg_sins(100,snr,sin_ds,num_pca)
        for x in range(repeats):
            start_main = time.time()
            results = np.append(results,main(model,sin_ds,snr,num_pca,grouping))
            end_main = time.time()
            # print(f'Main : {end_main-start_main}')
        results = np.reshape(results,(-1,4))
        for i in range(4):
            print(np.sum(results[:,i])/repeats,end = '\t')
        print('\n')
