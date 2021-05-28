import collections


def from_z(z_row, z_cut):
    for x in range(len(z_row)):
        if z_cut < float(z_row[x]):
            return x-1
    print("ERROR: Cut too high, all in one class")
    return len(z_row)

def cut(table, z_row, z_cut):
    it = from_z(z_row, z_cut)
    vals = table[:, it]
    classes = [int(x) for x in vals]

    c_freq = collections.Counter(classes)
    c_freq_ordered = c_freq.most_common()
    # two most common class ids
    c0 = c_freq_ordered[0][0]
    c1 = c_freq_ordered[1][0]

    bin_classes = []
    for x in classes:
        if x == c0:
            bin_classes.append(0)
        elif x == c1:
            bin_classes.append(1)
        else:
            bin_classes.append(-1)

    return bin_classes
