import gemmi

def create(ids, clic_dir):
    out_doc = gemmi.cif.Document()
    block = out_doc.add_new_block('particles')
    tags = ['_id', '_path']
    loop = block.init_loop('', tags)
    loop.add_row([f'-1\t', f'z_score\t'])  # z scores
    for x in range(1, len(ids)):
        loop.add_row([f'{x}\t', f'{ids[x]}\t'])

    out_doc.write_file(f'{clic_dir}/particles.star')
    return out_doc


def update(star_file, labels, it, z_score, clic_dir):
    block = star_file.find_block('particles')
    ''' a bit hacky... this gets correct particle table '''
    for x in block:
        table = x.loop
    tags = table.tags

    table = block.find(tags)
    tags.append(f'_it{it}')

    new_doc= gemmi.cif.Document()
    block_temp = new_doc.add_new_block('particles')
    #block_z = new_doc.add_new_block('z_scores')
    loop = block_temp.init_loop('', tags)  # make temp new table
#    loop_z = block_z.init_loop('', ['_it', '_z'])  # make temp new table
#    loop_z.add_row([f'{it}\t', f'{-999}'])

    # z score 1st row
    row = table[0]
    new_row = list(row)
    new_row.append(f'{z_score}')
    loop.add_row(new_row)  # update temp new table with all data

    for i in range(1, len(table)):
        row = table[i]
        new_row = list(row)
        new_row.append(f'{labels[i]}\t')
        loop.add_row(new_row)  # update temp new table with all data

    new_doc.write_file(f'{clic_dir}/particles_CLIC.star')
    return new_doc
