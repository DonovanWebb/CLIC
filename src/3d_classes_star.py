import gemmi
import argparse

def from_z(z_row, z_cut):
    for x in range(len(z_row)):
        if z_cut < float(z_row[x]):
            return x-1
    print("ERROR: Cut too high, all in one class")
    return len(z_row)

def new_star(class_n, members, relion_star, table):
    block = relion_star.find_block('particles')
    loop = block.find_loop_item('_rlnimagename').loop
    tags = loop.tags
    loop = block.init_loop('', tags)
    for m in members:
        loop.add_row(table[m])

    relion_star.write_file(f'particles_CLIC_{class_n}.star')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    t = ''' CLIC star file '''
    parser.add_argument("-i", "--clic_input", help=t, required=True, type=str)
    t = ''' Relion star file '''
    parser.add_argument("-r", "--relion_input", help=t, required=True, type=str)
    t = ''' z score to cut at '''
    parser.add_argument("-c", "--cut", help="level to cut at", required=True, type=float)

    args = parser.parse_args()

    clic_file = args.clic_input
    relion_file = args.relion_input
    z_cut = args.cut


    clic_star = gemmi.cif.read_file(clic_file)
    clic_block = clic_star.find_block('particles')
    for x in clic_block:
        table = x.loop
    tags = table.tags
    table = clic_block.find(tags)
    z_row = list(table[0])[2:]
    it = from_z(z_row, z_cut)
    im_names = [x for x in clic_block.find_values(f'_path')][1:]
    vals = list(clic_block.find_values(f'_it{it}'))[1:]
    classes = [int(x) for x in vals]
    # dict of classes and members (rln index)
    cl_dict = {i : [] for i in set(classes)}


    relion_star = gemmi.cif.read_file(relion_file)
    relion_block = relion_star.find_block('particles')
    all_parts = [x for x in relion_block.find_values(f'_rlnimagename')][1:]


    for x in range(len(im_names)):
        im = im_names[x]
        im_class = classes[x]
        rln_ind = all_parts.index(im) + 1
        mems = cl_dict[im_class]
        mems.append(rln_ind)
        cl_dict[im_class] = mems

    block = relion_star.find_block('particles')
    loop = block.find_loop_item('_rlnimagename').loop
    tags = loop.tags
    table = [list(x) for x in block.find(tags)]
    for x in set(classes):
        mems = cl_dict[x]
        new_star(x, mems, relion_star, table)

