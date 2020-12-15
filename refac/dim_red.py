"""
input: sinograms matrix (N, nlines, line)  and config file
output: reduced dimension sinogram matrix (N, nlines, num_comps)

reduces dimensions of all single lines by some dim red method
"""
import numpy as np
import time


def comp_var(sinos_trans):
    ''' get variance of each component '''
    comp_vects = []
    for x in sinos_trans.T:
        comp_vects.append(np.var(x))
    return comp_vects


def split_sinos(sinos):
    lines = np.reshape(sinos, (-1, sinos.shape[2]))
    return lines


def fitmodel(sinos, model, num_comps):
    # LINEAR
    if model == 'PCA':
        from sklearn.decomposition import PCA
        model = PCA(n_components=num_comps)
    # MANIFOLDS
    elif model == 'ISOMAP':
        from sklearn.manifold import Isomap
        model = Isomap(n_components=num_comps)
    elif model == 'LLE':
        from sklearn.manifold import LocallyLinearEmbedding as LLE
        model = LLE(n_components=num_comps, n_neighbors=5)
    elif model == 'MDS':
        from sklearn.manifold import MDS
        model = MDS(n_components=num_comps)
    elif model == 'TSNE':
        from sklearn.manifold import TSNE
        model = TSNE(n_components=num_comps)
    elif model == 'UMAP':
        import umap
        model = umap.UMAP(n_neighbors=5, min_dist=0.3,
                          n_components=num_comps)
    elif model == 'TRIMAP':
        import trimap
        model = trimap.TRIMAP(n_iters=1000)

    start_train = time.time()
    lines = split_sinos(sinos)
    sinos_trans = model.fit_transform(lines)
    end_train = time.time()
    print(f'Time to Train model : {end_train-start_train}')

    comp_var(sinos_trans)

    return sinos_trans
