import gemmi
import numpy as np
import mrcfile
import matplotlib.pyplot  as plt
import argparse


def from_z(z_row, z_cut):
    for x in range(len(z_row)):
        if z_cut < float(z_row[x]):
            return x-1
    print("ERROR: Cut too high, all in one class")
    return len(z_row)

def cut(star_file, z_cut):
    dendro_star = gemmi.cif.read_file(star_file)
    block = dendro_star.find_block('particles')
    locations = [x.split('@') for x in block.find_values(f'_path')]
    for x in block:
        table = x.loop
    tags = table.tags
    table = block.find(tags)
    z_row = list(table[0])[2:]
    it = from_z(z_row, z_cut)
    vals = list(block.find_values(f'_it{it}'))[1:]
    print(vals)
    classes = [int(x) for x in vals]
    n = len(classes)

    '''
    Now need to open each dataset with labelled classes
    '''

    clusters = np.array(classes)

    fig, axs = plt.subplots(10, n//10 + 1)
    for x in range(10):
        for y in range(n//10 + 1):
            a = axs[x, y]
            a.set_xticks([])
            a.set_yticks([])
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["left"].set_visible(False)
            a.spines["bottom"].set_visible(False)

    counter = 0
    group_counter = -1
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange',
            'pink', 'brown', 'black', 'magenta', 'cyan']

    for i in range(n):
        cl = np.where(clusters == i)[0]
        cl = np.array([x+1 for x in cl])  # correct for indexing as z first row
        if cl.size != 0:
            group_counter += 1
            color = colors[group_counter%len(colors)]
            for x in cl:
                if locations[x][-1].endswith('mrcs'):
                    with mrcfile.open(locations[x][-1]) as f:
                        all_ims = f.data
                    im = all_ims[int(locations[x][0]) - 1]
                elif locations[x][-1].endswith('mrc'):
                    with mrcfile.open(locations[x][0]) as f:
                        im = f.data


                a = axs[counter//(n//10+1), counter%(n//10+1)]
                a.spines["top"].set_visible(True)
                a.spines["right"].set_visible(True)
                a.spines["left"].set_visible(True)
                a.spines["bottom"].set_visible(True)

                a.spines["top"].set_lw(2)
                a.spines["right"].set_lw(2)
                a.spines["left"].set_lw(2)
                a.spines["bottom"].set_lw(2)

                a.imshow(im, cmap='gray')
                a.spines['right'].set_color(color)
                a.spines['left'].set_color(color)
                a.spines['top'].set_color(color)
                a.spines['bottom'].set_color(color)
                a.set_ylabel(x)
                counter += 1

    plt.subplots_adjust(left=0.01, 
                        bottom=0.01,
                        right=0.99,  
                        top=0.99,  
                        wspace=0.01,  
                        hspace=0.15) 
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="star file from CLIC", required=True, type=str)
    parser.add_argument("-c", "--cut", help="level to cut at", required=True, type=float)

    args = parser.parse_args()

    cut(args.input, args.cut)
