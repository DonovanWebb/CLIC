import gemmi

def create(ids):
    out_doc = gemmi.cif.Document()
    block = out_doc.add_new_block('particles')
    tags = ['_id', '_path']
    loop = block.init_loop('', tags)
    for x in range(len(ids)):
        loop.add_row([f'{x}\t', f'{ids[x]}\t'])

    out_doc.write_file('particles.star')
    return out_doc


def update(star_file, labels, it):
    block = star_file.find_block('particles')
    ''' a bit hacky... this gets correct particle table '''
    for x in block:
        table = x.loop
    tags = table.tags

    table = block.find(tags)
    tags.append(f'_it{it}')

    new_doc= gemmi.cif.Document()
    block_temp = new_doc.add_new_block('particles')
    loop = block_temp.init_loop('', tags)  # make temp new table


    for i in range(len(table)):
        row = table[i]
        new_row = list(row)
        new_row.append(f'{labels[i]}\t')
        loop.add_row(new_row)  # update temp new table with all data

    new_doc.write_file('particles.star')
    return new_doc
