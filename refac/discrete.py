import matplotlib.pyplot as plt
import numpy as np


def get_stats(group_lines):
    ''' get useful stats from a the group of lines provided.
        Input is the group of lines as a np array,
        output is mean and std of lines. '''

    num_lines = group_lines.shape[0]
    eucl_dists = np.array([])
    for x in range(num_lines): 
        for y in range(x+1, num_lines):  # no pair repeated
            dist = np.linalg.norm(group_lines[x]-group_lines[y])
            eucl_dists = np.append(eucl_dists, dist)
    mean = np.mean(eucl_dists)
    std = np.std(eucl_dists)

    print('mean: ', mean)
    print('std: ', std)
    return mean, std


def rand_lines(all_lines, n_rand=100):
    ''' Randomly choose lines from all possible lines.
        Input is all lines from dataset,
        output is a random group of lines as an np array '''

    n = all_lines.shape[0]
    rand = np.random.randint(n, size=n_rand)
    r_lines = all_lines[rand]

    return r_lines


def get_discrete_lines(lines, th_lines, r, theta):
    ''' Get all discrete lines from sinograms in range r.
        Input: all lines, theoretical common line positions, radius
        to consider, angular step on sinograms.
        Output is group of these lines as np array '''
    group_lines = np.array([])
    for cl in th_lines:
        s, x = cl  # format of each entry of th_lines is (sinogram, line)
        lower_x = x-r
        if lower_x < 0: lower_x = 0
        upper_x = x+r
        if upper_x > 360: upper_x = 360
        lower_discrete = np.ceil(lower_x / theta) * theta
        if lower_discrete < upper_x and lower_discrete <= 360:  # lines in range
            upper_discrete = np.floor(upper_x / theta) * theta
            per_sino = 360 // theta
            if lower_discrete == upper_discrete:
                ind = int(s*per_sino + lower_discrete/theta)
                group_lines = np.append(group_lines, lines[ind])
            else:
                for l in range(int(lower_discrete/theta), int(upper_discrete/theta)):
                    ind = int(s*per_sino + l)
                    group_lines = np.append(group_lines, lines[ind])
        else:
            print("No lines in range: ", lower_x, upper_x)
    n_comps = lines[0].shape[-1]
    group_lines = np.reshape(group_lines, (-1, n_comps))
    print(f"Found {group_lines.shape[0]} lines")
    return group_lines


def plot(group_lines, all_lines):
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    ax1.scatter(all_lines[:, 0],
                all_lines[:, 1], color='b')
    for x in group_lines: 
        ax1.scatter(x[0],
                    x[1], color='r')
        ax2.plot(x)
    ax1.axis('off')
    plt.show()


if __name__ == '__main__':
    a = np.array([(0,1),(3,4),(10,8),(7,4)])
    get_stats(a)
