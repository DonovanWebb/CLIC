import numpy as np
from timeit import default_timer as timer
from numba import vectorize
from numba import jit, njit

#-------------------------------------------------#
#                   WITH JIT                      #
#-------------------------------------------------#


@jit(nopython=True, parallel=False)
def eucl_dist(node, nodes):
    '''
    input is two (2) and (l_sizex2) arrays
    output is (l_size) array
    '''
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return dist_2


@jit(nopython=True, parallel=False)
def in_one(n, data, ln):
    mins = np.zeros((n*n*ln))
    ind = 0
    for i0 in range(data.shape[0]):
        sin = data[i0]
        for i1 in range(data.shape[0]):
            sc = data[i1]
            for i2 in range(sin.shape[0]):
                l = sin[i2]
                dists = eucl_dist(l, sc)
                min_ = np.argmin(dists)
                mins[ind] = dists[min_]
                ind += 1
    return np.reshape(mins, (n**2, ln))


#-------------------------------------------------#
#                   WITH VEC                      #
#-------------------------------------------------#
'''
Input is n (num samples), data (as array), ln (length of sinogram)
Output is minimum distance array
'''


# 3!
@vectorize(["float64(float64, float64, float64, float64, float64, float64)"],
           target='parallel')
def vec_eucl_dist3(a1, a2, a3, b1, b2, b3):
    dist = (a1 - b1)**2 + (a2 - b2)**2 + (a3-b3)**2
    return dist


@njit
def make_array3(n, x, data, ln):
    data_a1 = data[x, :, 0].flatten()
    data_a2 = data[x, :, 1].flatten()
    data_a3 = data[x, :, 2].flatten()

    data_a1 = np.repeat(data_a1, n*ln)
    data_a2 = np.repeat(data_a2, n*ln)
    data_a3 = np.repeat(data_a3, n*ln)

    return data_a1, data_a2, data_a3


def gpu3(n, data, ln):

    results_min = np.zeros(n**2*ln)
    b1 = data[:, :, 0].flatten()
    b2 = data[:, :, 1].flatten()
    b3 = data[:, :, 2].flatten()
    b1 = np.vstack([b1]*ln).flatten()
    b2 = np.vstack([b2]*ln).flatten()
    b3 = np.vstack([b3]*ln).flatten()

    start = timer()
    for x in range(n):
        a1, a2 ,a3 = make_array3(n, x, data, ln)
        eu_dists = np.zeros(ln*ln*n, dtype="float64")
        eu_dists = vec_eucl_dist3(a1, a2, a3, b1, b2, b3)
        results_min[x*ln*n:(x+1)*(ln*n)] = find_mins_batch(
            eu_dists, n, ln)
    duration = timer() - start
    print("Finding results:", duration)

    return np.reshape(results_min, (n*ln, n))


# 2!
@vectorize(["float64(float64, float64, float64, float64)"],
           target='parallel')
def vec_eucl_dist(a1, a2, b1, b2):
    dist = (a1 - b1)**2 + (a2 - b2)**2
    return dist


@njit
def make_array(n, x, data, ln):
    data_a1 = data[x, :, 0].flatten()
    data_a2 = data[x, :, 1].flatten()

    data_a1 = np.repeat(data_a1, n*ln)
    data_a2 = np.repeat(data_a2, n*ln)

    return data_a1, data_a2


def gpu(n, data, ln):

    results_min = np.zeros(n**2*ln)
    b1 = data[:, :, 0].flatten()
    b2 = data[:, :, 1].flatten()
    b1 = np.vstack([b1]*ln).flatten()
    b2 = np.vstack([b2]*ln).flatten()

    start = timer()
    for x in range(n):
        a1, a2 = make_array(n, x, data, ln)
        eu_dists = np.zeros(ln*ln*n, dtype="float64")
        eu_dists = vec_eucl_dist(a1, a2, b1, b2)
        # results_min[x*ln*n:(x+1)*(ln*n)] = find_mins_batch(
        #     vec_eucl_dist(a1, a2, b1, b2), n, ln)
        results_min[x*ln*n:(x+1)*(ln*n)] = find_mins_batch(
            eu_dists, n, ln)
    duration = timer() - start
    print("Finding results:", duration)

    return np.reshape(results_min, (n*ln, n))


@njit
def find_mins(results, n, ln):
    mins = np.zeros(ln*n**2)
    for x in range(ln*n**2):
        sample = results[ln*x:ln*(x+1)]
        mins_arg = np.argmin(sample)
        mins[x] = sample[mins_arg]
    return mins


@njit
def find_mins_batch(results, n, ln):
    mins = np.zeros(ln*n)
    for x in range(ln*n):
        sample = results[ln*x:ln*(x+1)]
        mins_arg = np.argmin(sample)
        mins[x] = sample[mins_arg]
    return mins


def calc_score(dist):
    sorted_dists = np.argsort(dist)
    n = int(0.1*sorted_dists.shape[0])
    top_n_arg = sorted_dists[:n]
    top_n = [dist[x] for x in top_n_arg]
    score = np.sum(top_n) / n
    return score
    
def weight_score(dist):
    # pass through log fn then just add
    dist = np.log10(dist)
    score = np.sum(dist) / dist.shape[0]
    return score

def mixed_score(dist):
    # pass through log fn, take top 20%
    sorted_dists = np.argsort(dist)
    n = int(0.2*sorted_dists.shape[0])
    top_n_arg = sorted_dists[:n]
    top_n = [dist[x] for x in top_n_arg]
    top_n = np.log10(top_n)
    score = np.sum(top_n) / n
    return score



def make_outputs(n, ln, min_dists):
    dist_dict = {}
    comps = []
    dist_scores = []
    x = np.array([a for a in range(ln)])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist_args = i*n*ln + j + x*n
            dists = min_dists[dist_args]
            # dist_scores.append(weight_score(dists))
            dist_scores.append(calc_score(dists))
            # dist_scores.append(mixed_score(dists))
            comps.append((i, j))
            dist_dict[(i, j)] = dists
    return dist_dict, dist_scores, comps


def main(n, data, ln):
    data = np.array(data, dtype="float64")
    min_dists_vec = gpu(n, data, ln)
    dist_dict, dist_scores, comps = make_outputs(n, ln, min_dists_vec.flatten())
    return dist_dict, dist_scores, comps


if __name__ == '__main__':
    n = 100
    ln = 75
    data = np.random.random(size=(n, ln, 2))
    start = timer()
    min_dists_vec = gpu(n, data, ln)
    dist_dict, dists, comps = make_outputs(n, ln, min_dists_vec.flatten())
    duration = timer() - start
    print("vect:", duration)

    print('\n')
    start = timer()
    min_dists_njit = results = in_one(n, data, ln)
    duration = timer() - start
    print("Just njit:", duration)
