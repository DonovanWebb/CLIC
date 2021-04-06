"""
input: dimensionally reduced lines
output: stepwise clustering of sinograms

Initially each sinogram is one cluster, so we have clusters C1, C2, C3 ... CN,
where N is num of projections.

A score is made for each pair of clusters by finding euclidian distance between
lines contained within each group.

Lowest scoring cluster pair is merged and the new larger cluster is named after
the smaller cluster number of the merging clusters. i.e. if C4 and C7 merge
they will now all be contained in C4.

This iterates until all points are contained within one cluster - C1.

At each merging we get an output of which cluster each sinogram belongs to.
Large merges are points of interest and so this is printed at the end of the
program.
The clusters prior to a large merge should be analysed further.
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as eucl_dist
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import star_writer

import time


def debug_p(msg):
    pass
    # print(msg)


def initial_dict(lines, num):
    ''' make dictionary of initial classes i.e. which sinogram each line
    belongs to '''
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


def cap_cluster(cl):
    ''' For speed up and to save memory. Caps cluster size to
    N particles (with 120 lines each) '''
    N = 400
    len_cl = np.shape(cl)[0]

    if len_cl >= N*120:  # For speed up and save mem.
        rand_selection = np.random.randint(0, len_cl, N*120)
        cl = cl[rand_selection]
    return cl

def find_score(clX, clY, name):
    ''' find score between two clusters '''
    clX = cap_cluster(clX)
    clY = cap_cluster(clY)
            
    paired_dists = eucl_dist(clX, clY)
    paired_dists = paired_dists.ravel()
    dists = 1/paired_dists
    #dists[np.where(dists <= 0.05)] = 0
    # plot_hist(dists, name)
    score = np.mean(dists)
    debug_p(f'Score: {score}')
    return score


def dist_hist(clX):
    ''' find histogram of dists in one cluster '''
    plt.figure(176)
    plt.title(f'{np.shape(clX)}')
    paired_dists = eucl_dist(clX, clX)
    plt.hist(paired_dists.ravel(), bins='auto')
    plt.show()


def find_scoretable(cluster_dict, cluster_labels):
    ''' All scores between all pairs of clusters, looks like:

       | C1 | C2 | C3 | C4 |
    C1 | XX | .. | .. | .. |
    C2 | .. | XX | .. | .. |
    C3 | .. | .. | XX | .. |
    C4 | .. | .. | .. | XX |

    ''' 
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
    if p0 == p1:
        print("ERROR, p0=p1")
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


def update_scores_quick(cluster_dict, pointsab, pointsp, num_clusters, cluster_labels, scoretable):
    ''' fill in new empty scoretable slots  trying new method'''
    # add empty row
    len_sc = np.shape(scoretable)[0] + 1
    scoretable_e = np.zeros((len_sc, len_sc))
    # fill top left with original scoretable missing new entry
    scoretable_e[:-1, :-1] = scoretable
    scoretable = scoretable_e

    a, b = pointsab  # ind of score table max
    p0, p1 = pointsp # cl labels for merging groups
    clX = cluster_dict[p0]
    lenX = np.shape(clX)[0]
    clY = cluster_dict[p1]
    lenY = np.shape(clY)[0]
    for Z in range(np.shape(scoretable)[0]):
        if Z == a or Z == b:  # dont update own?
            continue
        else:
            scoresZ = scoretable[Z]
            scoreX = scoresZ[a]
            scoreY = scoresZ[b]
            score = (lenX * scoreX + lenY * scoreY) / (lenX + lenY)
            scoretable[Z, -1] = score
            scoretable[-1, Z] = score
    # delete paired
    if a > b:  x = b;  y = a
    else:  x = a;  y = b
    # clear rows in order to update
    scoretable = np.delete(scoretable, x, 0)
    scoretable = np.delete(scoretable, y-1, 0)
    scoretable = np.delete(scoretable, x, 1)
    scoretable = np.delete(scoretable, y-1, 1)
    return scoretable


def update_cl(cluster_dict, scoretable, cluster_labels, Z, Z_corr):
    score = np.max(scoretable)
    a, b = np.where(scoretable == score)
    if len(a) > 1:  # only take one entry
        a = a[0]
        b = b[0]

    paired = (cluster_labels[int(a)], cluster_labels[int(b)])
    debug_p(f'paired arg{(a, b)}')
    debug_p(f'paired {paired}')
    p0 = np.min(paired)  # By convention new group name is lowest of two
    p1 = np.max(paired)

    cluster_labels = update_clusterlabels(cluster_labels, p0, p1)
    num_clusters = len(cluster_labels)
    scoretable = update_scores_quick(cluster_dict, (a, b), paired, num_clusters,
                                         cluster_labels, scoretable)

    # update cluster dict
    new_group = np.concatenate((cluster_dict[p0],
                                cluster_dict[p1]))
    cluster_dict[p0] = new_group
    del cluster_dict[p1]  # is this needed? save memory, but slow down running

    # OLD Method ##
    # scoretable = update_scoretable(scoretable, a, b, num_clusters)
    # scoretable = update_scores(cluster_dict, p0, num_clusters,
    #                            cluster_labels, scoretable)
    # ## 

    debug_p(f'{(scoretable*100).astype(int)}')

    # Dendrogram update
    Z.append([Z_corr[p0], Z_corr[p1], 1/score, 0])
    Z_corr[p0] = np.max(Z_corr) + 1

    return cluster_dict, scoretable, cluster_labels, paired, Z, Z_corr, 1/score


def print_clusters_all(num, all_paired):
    # takes all paired values to make final clustering
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

    return clusters


def print_clusters(clusters, count, large_merges, paired):
    # Takes one paired at a time
    s1, s2 = paired
    c1 = clusters[s1]
    c2 = clusters[s2]

    count1 = count[c1]
    count2 = count[c2]
    merge_size = min(count1, count2)
    if merge_size > 3:
        large_merges.append([c1, c2, count1, count2])

    # update clusters by merging two
    if c1 < c2:
        clusters[clusters == c2] = c1
        count[c1] += count[c2]
    elif c2 < c1:
        clusters[clusters == c1] = c2
        count[c2] += count[c1]

    # Display cluster
    cl = clusters
    return cl, clusters, count, large_merges


def clustering_main(lines, config, star_file, clic_dir):
    cl_labels = list(range(config.num))
    print(cl_labels)
    cl_dict = initial_dict(lines, config.num)
    scoretable = find_scoretable(cl_dict, cl_labels)
    all_paired = []
    Z = []  # Linkage matrix for drawing dendrogram
    Z_corr = list(range(config.num))
    for i in range(config.num - 1):
        timer_iter = time.time()
        cl_dict, scoretable, cl_labels, paired, Z, Z_corr, z_score = update_cl(cl_dict, scoretable,
                                                                      cl_labels, Z, Z_corr)
        # print(paired)  # To see which are paired
        all_paired.append(paired)
        # print(cl_labels)  # To see current clusters
        if i == 0:  # First pass
            clusters = np.array(list(range(config.num)))
            count = {x: 1 for x in range(config.num)}
            cl, clusters, count, large_merges = print_clusters(
                clusters, count, [], paired)
        else:
            cl, clusters, count, large_merges = print_clusters(
                clusters, count, large_merges, paired)
        # print(cl)  # To see which assignment to clusters
        star_file = star_writer.update(star_file, cl, i, z_score, clic_dir)
        print(f"{i}: {time.time() - timer_iter}")
        # ### dist hist
        # N = 10
        # for cl_i in cl_dict:
        #     cl_ar = cl_dict[cl_i]
        #     if np.shape(cl_ar)[0] >= N * 120:
        #         dist_hist(cl_ar)
        # ###
    np.save(f"{clic_dir}/large_merges", large_merges)

    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)
    np.save(f"{clic_dir}/dendrogram", Z)
    # Add color to dendro
    colors = ['r', 'b', 'g', 'yellow', 'purple', 'brown']
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        for x in range(2):
            if int(lbl.get_text()) % 2 == x:
                lbl.set_color(colors[int(x)])
    plt.savefig(f"{clic_dir}/dendrogram.pdf")
    plt.show()
