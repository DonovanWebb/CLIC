import math
import numpy as np


def mat2euZYZ(R):
    '''
    ZYZ convention -relion
    '''
    theta = math.acos(round_n(R[2,2]))
    if theta == np.pi:
        # Not unique
        phi = -math.atan2(R[1,0],R[1,1])
        psi = 0
    elif theta == 0:
        # Not unique
        phi = math.atan2(R[1,0],R[1,1])
        psi = 0
    else:
        phi = math.atan2(R[1,2],R[0,2])
        psi = math.atan2(R[2,1],-R[2,0])
    return np.array([phi, theta, psi])


def rad2deg(n):
    deg = n/(2*np.pi) * 360
    return deg


def deg2rad(n):
    rad = n/360 * (2*np.pi)
    return rad


def round_n(n):
    return np.round(n,5)


def cs(phi,theta,psi):
    c1 = math.cos(phi)
    c2 = math.cos(theta)
    c3 = math.cos(psi)
    s1 = math.sin(phi)
    s2 = math.sin(theta)
    s3 = math.sin(psi)
    #ZYZ rot matrix
    ZYZ = np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                    [c1*s3+c2*c3*s1,  c1*c3-c2*s1*s3, s1*s2],
                    [        -c3*s2,           s2*s3,    c2]])
    return ZYZ


def predicter(eu):
    if eu[1] == 0:
        print('Warning: Sin1 = Sin2')
    lines = np.array([90-eu[0], 90+eu[2]])
    if lines[0] < 0 or lines[1] < 0:
        lines = lines+360
    lines = lines%360
    return lines


def shift_coord(phi1,theta1,psi1,phi2,theta2,psi2):
    phi1 = deg2rad(phi1)
    theta1 = deg2rad(theta1)
    psi1 = deg2rad(psi1)
    phi2 = deg2rad(phi2)
    theta2 = deg2rad(theta2)
    psi2 = deg2rad(psi2)

    ZYZ1 = cs(phi1,theta1,psi1)
    ZYZ2 = cs(phi2,theta2,psi2)

    invZYZ1 = np.linalg.inv(ZYZ1)

    zeros = round_n(rad2deg(mat2euZYZ(np.dot(invZYZ1,ZYZ1))))
    if not np.array_equal(zeros,[0,0,0]):
        print(f'Warning: ZYZ*InvZYZ = {zeros}')
        exit()
    new_eu = round_n(rad2deg(mat2euZYZ(np.dot(invZYZ1,ZYZ2))))
    return new_eu


def get_dataset(starfile):
    import gemmi
    import json
    in_doc = gemmi.cif.read_file(starfile)
    data_as_dict = json.loads(in_doc.as_json())['#']
    rots = np.array(data_as_dict['_rlnanglerot'])
    tilts = np.array(data_as_dict['_rlnangletilt'])
    psis = np.array(data_as_dict['_rlnanglepsi'])
    data = np.stack((rots,tilts,psis), axis=-1)
    return data
    

def app2dict(ldict,s1,l1,s2,l2):
    if (s1,l1) in ldict:
        ar = []
        for t in ldict[(s1,l1)]:
            ar.append(t)
        ar.append((s2, l2))
        ldict[(s1,l1)] = ar
    else:
        ldict[(s1,l1)] = [(s2, l2)]
    return ldict
    

def myround(x, base=3):
    return base * np.round(x/base)


def clines(c_line_dict, l1_l2, s1, s2):
    l1, l2 = tuple(myround(l1_l2)%360)
    c_line_dict = app2dict(c_line_dict,s1,l1,s2,l2)
    c_line_dict = app2dict(c_line_dict,s2,l2,s1,l1)
    return c_line_dict


def count_dict(ldict):
    groups = set()
    for x in ldict:
        g = ldict[x]
        g.append(x)
        g.sort(key=lambda tup: tup[0])
        g = tuple(g)
        groups.add(g)
    print(groups)
    
    sizes = []
    for x in groups:
        # print(len(x))
        sizes.append(len(x))

    print('sum ', np.sum(sizes))
    print('max ', np.max(sizes))
    print('min ', np.min(sizes))
    print('mean ', np.mean(sizes))
    print('std ', np.std(sizes))

'''
# wrong as too much overlaps!
def update_lines(all_lines, s1, s2, l1_l2, nlines):
    l1, l2 = tuple(l1_l2)
    i1 = all_lines[int(s1*nlines+l1)]
    i2 = all_lines[int(s2*nlines+l2)]
    if i1 < i2:
        all_lines[all_lines == i2] = i1
    elif i2 < i1: 
        all_lines[all_lines == i1] = i2
    else: 
        print("They are the same!")
    return all_lines
'''


def main(projs=700):
    rounding = 3
    nlines = 360//rounding
    all_lines = np.arange(projs * nlines)
    com_lines = {}
    all_sins = get_dataset('../angles/7_5_angles.star')
    for s1 in range(projs):
        for s2 in range(s1+1, projs):
    # for s1 in range(10):
        # for s2 in range(80, 90):
            eu1 = all_sins[s1]
            eu2 = all_sins[s2]

            phi1 = eu1[0]
            theta1 = eu1[1]
            psi1 = eu1[2]

            phi2 = eu2[0]
            theta2 = eu2[1]
            psi2 = eu2[2]

            new_eu = shift_coord(phi1,theta1,psi1,phi2,theta2,psi2)

            l1_l2 = predicter(new_eu)
            com_lines = clines(com_lines, l1_l2, s1, s2)
            l1_l2_2 = (l1_l2 + 180)%360  # to get mirror
            com_lines = clines(com_lines, l1_l2_2, s1, s2)
    count_dict(com_lines)
    return(all_lines)

if __name__ == '__main__':
    main()
