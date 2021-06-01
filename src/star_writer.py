import gemmi
import time

def create(ids, clic_dir):
    out_doc = gemmi.cif.Document()
    block = out_doc.add_new_block('particles')
    tags = ['_id', '_rlnimagename']
    loop = block.init_loop('', tags)
    loop.add_row([f'-1\t', f'z_score\t'])  # z scores
    for x in range(1, len(ids)):
        loop.add_row([f'{x}\t', f'{ids[x]}\t'])

    out_doc.write_file(f'{clic_dir}/particles.star')
    return out_doc


def update_data(tags, labels, table, it):
    tags.append(f'_it{it}')

    for i in range(len(table)):
        row = table[i]
        row[it] = f'{labels[i]}\t'

    return tags, table


def end_write(tags, table, z_score_list, clic_dir, ids):
    new_doc= gemmi.cif.Document()
    block_temp = new_doc.add_new_block('particles')
    loop = block_temp.init_loop('', tags)  # make temp new table

    row = ['-', 'z_score']
    row.extend(z_score_list)
    loop.add_row(row)

    for x in range(len(ids)):
        row = [f'{x}\t', f'{ids[x]}\t']
        row.extend(table[x])
        # print(tags.shape)
        # print(row.shape)
        # exit()
        loop.add_row(row)  # update temp new table with all data

    new_doc.write_file(f'{clic_dir}/particles_CLIC.star')


def update(star_file, labels, it, z_score, clic_dir):
    """ old slower method """
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

    tot_time = 0
    for i in range(1, len(table)):
        row = table[i]
        st_time = time.time()
        new_row = list(row)
        tot_time += time.time() - st_time
        new_row.append(f'{labels[i]}\t')
        loop.add_row(new_row)  # update temp new table with all data
    print(tot_time)

    new_doc.write_file(f'{clic_dir}/particles_CLIC_old.star')
    return new_doc
