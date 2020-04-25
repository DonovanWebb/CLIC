# Adapted Agglomerative hierarchy!
'''

1 Assign each line to a sinogram
2 Take pair of sinograms. Calculate total distance between them

3? some magic with weighting to fix anomalies
4 Group pair with lowest distance
5? Reasses weighting with cluster as oppose to single sinogram
6 Repeat from 3 until all in one cluster


How to do these things
1)) This info is easy, single lines are split from sinogram originally
2)) find eucl dist between all points. Take lowest from each point
3)) Ideas: sum all dists, sum all 1/dists, find mean -> remove +-3std and sum
4)) make note of which got paired
5)) replace paired points with new averaged

'''
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

import eucl_vec
import plot_clusters
import sys
sys.path.insert(0, '../')
import mvs


def config():
    num = 16*10
    return num


@jit(nopython=True, parallel=False)
def numba_c1(x, y):
    z = x
    for i in range(len(x)):
        x_ = x[i]
        y_ = y[i]
        if y_ < x_:
            z[i] = y_
    return z


def comp_two_c1(all_dist, l1, l2, c1):
    x = all_dist[(c1, l1)]
    y = all_dist[(c1, l2)]
    return numba_c1(x, y)


def comp_two_c2(all_dist, l1, l2, c2):
    x = all_dist[(l1, c2)]
    y = all_dist[(l2, c2)]
    z = np.concatenate((x, y))
    all_dist[(l1, c2)] = z
    del all_dist[(l2, c2)]
    return z, all_dist


############################################
'''
 OLD. NOT VECTORISED BUT LEFT IN TO COMPARE
'''
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
def compare2(sino1, sino2):
    '''
    input is two (l_sizex2) arrays
    '''
    min_dists = []
    for i in range(sino1.shape[0]):
        x = sino1[i]
        dists = eucl_dist(x, sino2)
        min_dist = np.argmin(dists)
        min_dists.append(dists[min_dist])
    return min_dists


def first_pass(comp_dict):
    dists = []
    comps = []
    all_dist = {}
    # insert vect calcs here
    for i in comp_dict:
        for j in comp_dict:
            if i == j:
                continue
            dist = compare2(comp_dict[i], comp_dict[j])
            all_dist[(i, j)] = dist
            dists.append(eucl_vec.calc_score(dist))
            comps.append((i, j))
    return all_dist, dists, comps


'''
END OLD
'''
########################################


def first_pass_gpu(n, data, ln):
    all_dist, dist_scores, comps = eucl_vec.main(n, data, ln)
    return all_dist, dist_scores, comps


def next_passes(paired, comps, dists, all_dist):
    # remove all in dist and comp
    comps_ = []
    dists_ = []
    la, lb = paired[-1]  # find last paired sinograms

    if la < lb:  # make l1 the smaller of both la and lb for convention
        l1 = la
        l2 = lb
    else:
        l1 = lb
        l2 = la

    for r in range(len(comps)):
        c1, c2 = comps[r]

        if c1 != l1 and c1 != l2 and c2 != l1 and c2 != l2:
            comps_.append(comps[r])
            dists_.append(dists[r])

        elif c2 == l1 and c1 != l2:
            # compare (c1,l1) and (c1,l2)
            comps_.append(comps[r])
            new_dist = comp_two_c1(all_dist, l1, l2, c1)
            dists_.append(eucl_vec.calc_score(new_dist))

        elif c1 == l1 and c2 != l2:
            # compare (l1,c2) and (l2,c2)
            comps_.append(comps[r])
            new_dist, all_dist = comp_two_c2(all_dist, l1, l2, c2)
            dists_.append(eucl_vec.calc_score(new_dist))

    comps = comps_
    dists = dists_
    return comps, dists, all_dist


def find2_pairs(dists, paired, comps, comp_dict):
    closest_args = np.argsort(dists)
    to_del = []
    for x in range(2):
        paired_2 = comps[closest_args[x]]
        paired.append(paired_2)
        i, j = paired_2
        print(paired_2)
        a = comp_dict[i]
        b = comp_dict[j]
        c = np.concatenate((a, b))
        if i < j:
            comp_dict[i] = c
            to_del.append(j)
        elif j < i:
            comp_dict[j] = c
            to_del.append(i)
    for x in set(to_del):
        print(x)
        del comp_dict[x]
    return comp_dict, paired

def find_pairs(dists, paired, comps, comp_dict):
    closest_two_arg = np.argmin(dists)
    # print(dists[closest_two_arg])
    paired_2 = comps[closest_two_arg]
    paired.append(paired_2)
    i, j = paired_2
    a = comp_dict[i]
    b = comp_dict[j]
    c = np.concatenate((a, b))
    if i < j:
        comp_dict[i] = c
        del comp_dict[j]
    elif j < i:
        comp_dict[j] = c
        del comp_dict[i]
    return comp_dict, paired


def find_cluster(num, sinos, l_size):
    # Note: dists is dist_scores...
    comp_dict = {}
    for i in range(num):
        comp_dict[i] = sinos[l_size*i:l_size*(i+1)]

    # First pass fn
    data = np.reshape(sinos, (num, l_size, -1))
    all_dist, dists, comps = first_pass_gpu(num, data, l_size)

    comp_dict, paired = find_pairs(dists, [], comps, comp_dict)

    old_cl_count = 0
    for it in range(num-2):  # -2 as this is last pairing
        comps, dists, all_dist = next_passes(paired, comps, dists, all_dist)
        comp_dict, paired = find_pairs(dists, paired, comps, comp_dict)
    return paired


def print_clusters(num, paired):
    gt_cl = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    scores = np.array([])
    clusters = np.array(list(range(num)))
    count = {i:1 for i in range(num)}

    large_merges = []
    for c in paired:
        s1, s2 = c
        c1 = clusters[s1]
        c2 = clusters[s2]
        if c1 == c2:
            continue
        
        # Look at if it was a large merge

        count1 = count[c1]
        count2 = count[c2]
        merge_size = min(count1, count2)
        if merge_size > 3:
            print("large merge")
            large_merges.append([c1,c2,merge_size, count1, count2])

        # update clusters by merging two
        if c1 < c2:
            clusters[clusters == c2] = c1
            count[c1] += count[c2]
        elif c2 < c1:
            clusters[clusters == c1] = c2
            count[c2] += count[c1]

        # Display cluster
        cl = np.reshape(clusters, (-1, 16))
        print(cl)
        score = 0
        for it in range(num//16):
            score += np.sum((cl[it] - gt_cl)**2)
        scores = np.append(scores, score)
        
    print(large_merges)
    print(np.min(scores))
    return large_merges


def main():
    # num = config()
    num = 16*40
    ds = 4
    dataset = 'mixed'
    noise = 2.5
    pca_comp = 2

    if dataset == 'ctfcor':
        im_size = 196
    else:
        im_size = 150

    l_size = im_size//ds

    sinos, model = mvs.add_bg_sins(num, noise, ds, pca_comp, dataset)
    # 'Options are "NN" "all" "no e" and "mixed", mixedOff1,2,4,8

    # sinos = np.random.randint(50, size=(num*l_size, 2))  # simulated

    cl_start = time.time()
    paired = find_cluster(num, sinos, l_size)
    print_clusters(num, paired)
    cl_end = time.time()
    print(f'Clustering Time = {cl_end-cl_start:.3f}')

    plot_clusters.plot(sinos, num, l_size)
    # plt.show()


if __name__ == '__main__':
    main()
