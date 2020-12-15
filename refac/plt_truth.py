import matplotlib.pyplot as plt
import numpy as np


def get_stats(group_lines):
    num_lines = group_lines.shape[0]
    eucl_dists = np.array([])
    for x in range(num_lines): 
        for y in range(x+1, num_lines): 
            dist = np.linalg.norm(group_lines[x]-group_lines[y])
            eucl_dists = np.append(eucl_dists, dist)
    mean = np.mean(eucl_dists)
    std = np.std(eucl_dists)
    print('mean: ', mean)
    print('std: ', std)


def plot(lines, num, comps):
    group = ((0, 141.0), (53, 93.0), (66, 270.0), (80, 90.0), (230, 90.0), (252, 270.0), (435, 270.0), (626, 270.0), (640, 90.0), (652, 267.0), (672, 270.0)) #, (687, 267.0), (699, 39.0))
    group_lines = np.zeros((3*len(group), comps))
    per_sino = lines.shape[0] // num
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    ax1.scatter(lines[:, 0],
                lines[:, 1], color='b')
    count = 0
    for x in range(len(group)): 
        cl = group[x]
        s, a = cl
        for i in range(-1,2):
            l = a//3 + i
            ind = int((s)*per_sino + l)
            group_lines[count] = lines[ind]
            count += 1
            ax1.scatter(lines[ind, 0],
                        lines[ind, 1], color='r')
            ax2.plot(lines[ind+1])
    get_stats(group_lines)
    ax1.axis('off')
    plt.show()


if __name__ == '__main__':
    a = np.array([(0,1),(3,4),(10,8),(7,4)])
    get_stats(a)
