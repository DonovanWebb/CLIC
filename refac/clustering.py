import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as eucl_dist
import matplotlib.pyplot as plt


def debug_p(msg):
    pass
    # print(msg)


def initial_dict(lines, num):
    per_sino = lines.shape[0] // num  # How many single lines per sino
    # Put each line in an initial cluster
    cluster_dict = {i: lines[per_sino*i:per_sino*(i+1)] for i in range(num)}
    return cluster_dict


def plot_hist(dists, name):
    # optional
    if name[0] < 5:
        if name[1] < 5:
            # Get histogram
            hist, bins = np.histogram(dists, bins=100)
            hist = hist[1:]
            bins = bins[1:]

            # Plot
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)


def find_score(clX, clY, name):
    paired_dists = eucl_dist(clX, clY).ravel()
    dists = 1/paired_dists
    dists[np.where(dists <= 0.05)] = 0
    # plot_hist(dists, name)
    score = np.mean(dists)
    return score


def find_scoretable(cluster_dict, cluster_labels):
    num_clusters = len(cluster_labels)
    scoretable = np.zeros((num_clusters, num_clusters))
    for X in range(num_clusters):
        for Y in range(num_clusters):
            if X == Y:
                scoretable[X, Y] = 0
                continue
            else:
                clX = cluster_dict[cluster_labels[X]]
                clY = cluster_dict[cluster_labels[Y]]
                scoretable[X, Y] = find_score(clX, clY, (X, Y))
    return scoretable


def update_scoretable(scoretable, a, b, num_clusters):
    ''' remove paired scored and add new empty row for new scores '''
    # delete paired
    if a > b:  x = b;  y = a
    else:  x = a;  y = b
    # clear rows in order to update
    scoretable = np.delete(scoretable, x, 0)
    scoretable = np.delete(scoretable, y-1, 0)
    scoretable = np.delete(scoretable, x, 1)
    scoretable = np.delete(scoretable, y-1, 1)
    scoretable_e = np.zeros((num_clusters, num_clusters))
    # fill top left with original scoretable missing new entry
    scoretable_e[:-1, :-1] = scoretable
    scoretable = scoretable_e
    return scoretable


def update_clusterlabels(cluster_labels, p0, p1):
    ''' remove paired and add lowest as new cl label '''
    cluster_labels.remove(p0)
    cluster_labels.remove(p1)
    cluster_labels.append(p0)
    debug_p(f'after merge {cluster_labels}')
    return cluster_labels


def update_scores(cluster_dict, p0, num_clusters, cluster_labels, scoretable):
    ''' fill in new empty scoretable slots  '''
    clY = cluster_dict[p0]  # find score with new merged group
    for X in range(num_clusters):
        if cluster_labels[X] == p0:
            continue
        else:
            clX = cluster_dict[cluster_labels[X]]
            score = find_score(clX, clY, (X, p0))
            scoretable[X, -1] = score
            scoretable[-1, X] = score
    return scoretable


def update_cl(cluster_dict, scoretable, cluster_labels):
    a, b = np.where(scoretable == np.max(scoretable))
    if len(a) > 1:  # only take one entry
        a = a[0]
        b = b[0]

    paired = (cluster_labels[int(a)], cluster_labels[int(b)])
    debug_p(f'paired arg{(a, b)}')
    debug_p(f'paired {paired}')
    p0 = np.min(paired)  # By convention new group name is lowest of two
    p1 = np.max(paired)
    new_group = np.concatenate((cluster_dict[p0],
                                cluster_dict[p1]))
    debug_p(f'before merge {cluster_dict[p0].shape}')
    cluster_dict[p0] = new_group
    debug_p(f'after merge {cluster_dict[p0].shape}')
    del cluster_dict[p1]  # is this needed? save memory, but slow down running

    cluster_labels = update_clusterlabels(cluster_labels, p0, p1)
    num_clusters = len(cluster_labels)
    scoretable = update_scoretable(scoretable, a, b, num_clusters)
    scoretable = update_scores(cluster_dict, p0, num_clusters,
                               cluster_labels, scoretable)

    debug_p(f'{(scoretable*100).astype(int)}')
    return cluster_dict, scoretable, cluster_labels, paired


def print_clusters(num, all_paired):
    clusters = np.array(list(range(num)))

    for paired in all_paired:
        s1, s2 = paired
        c1 = clusters[s1]
        c2 = clusters[s2]

        # update clusters by merging two
        if c1 < c2:
            clusters[clusters == c2] = c1
        elif c2 < c1:
            clusters[clusters == c1] = c2

        # Display cluster
        cl = np.reshape(clusters, (-1, 16))
    return cl


def clustering_main(lines, Config):
    cl_labels = list(range(Config.num))
    print(cl_labels)
    cl_dict = initial_dict(lines, Config.num)
    scoretable = find_scoretable(cl_dict, cl_labels)
    all_paired = []
    for i in range(Config.num - 1):
        cl_dict, scoretable, cl_labels, paired = update_cl(cl_dict, scoretable,
                                                           cl_labels)
        print(paired)
        all_paired.append(paired)
        print(cl_labels)
        cl = print_clusters(Config.num, all_paired)
        print(cl)
