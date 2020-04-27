import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as eucl_dist
import matplotlib.pyplot as plt


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


            # plt.figure(name[0]+ name[1]*10)
            # plt.hist(paired_dists.ravel(), bins=100)
            # plt.title(name)
            # plt.figure(0)
            # plt.hist(paired_dists.ravel(), bins=100)


def find_score(clX, clY, name):
    paired_dists = eucl_dist(clX, clY).ravel()
    dists = 1/paired_dists
    # dists[np.where(dists <= 0.05)] = 0
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


def update_cl(cluster_dict, scoretable, cluster_labels):
    a, b = np.where(scoretable == np.max(scoretable))
    if len(a) > 1:
        a = a[0]
        b = b[0]
    paired = (cluster_labels[int(a)], cluster_labels[int(b)])
    p0 = np.min(paired)
    p1 = np.max(paired)
    new_group = np.concatenate((cluster_dict[p0],
                                cluster_dict[p1]))
    cluster_labels.remove(p1)
    cluster_dict[p0] = new_group
    del cluster_dict[p1]
    return cluster_dict, cluster_labels, paired


def clustering_main(lines, Config):
    cl_labels = list(range(Config.num))
    print(cl_labels)
    cl_dict = initial_dict(lines, Config.num)
    for i in range(Config.num - 1):
        scoretable = find_scoretable(cl_dict, cl_labels)
        cl_dict, cl_labels, paired = update_cl(cl_dict, scoretable,
                                               cl_labels)
        print(paired)
        print(cl_labels)
