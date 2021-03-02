import gemmi

def cut(star_file='particles.star', it=0):
    dendro_star = gemmi.cif.read_file(star_file)
    block = dendro_star.find_block('particles')
    classes = [int(x) for x in block.find_values(f'_it{it}')]
    print(classes)
    '''
    Now need to open each dataset with labelled classes
    '''

cut()
