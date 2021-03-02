import gemmi
import numpy as np
import mrcfile
import matplotlib.pyplot  as plt
import argparse


def cut(star_file, it):
    dendro_star = gemmi.cif.read_file(star_file)
    block = dendro_star.find_block('particles')
    classes = [int(x) for x in block.find_values(f'_it{it}')]
    print(classes)

    '''
    Now need to open each dataset with labelled classes
    '''
    with mrcfile.open('../../CLIC_exp/SLICEM_exp/mixture_2D.mrcs') as f:
        all_ims = f.data

    clusters = np.array(classes)

    fig, axs = plt.subplots(10, 10)
    for x in range(10):
        for y in range(10):
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
    for i in range(100):
        cl = np.where(clusters == i)[0]
        if cl.size != 0:
            group_counter += 1
            color = colors[group_counter%len(colors)]
            for x in cl:
                im = all_ims[x]
                #im = cv2.resize(im, (100,100))
                a = axs[counter//10, counter%10]
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
    parser.add_argument("-c", "--cut", help="level to cut at", required=True, type=int)

    args = parser.parse_args()

    cut(args.input, args.cut)
