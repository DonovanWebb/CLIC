import collections
from collections import Counter
import csv
import pandas as pd
from pandas.plotting import scatter_matrix

import mrcfile
import numpy as np
pd.set_option("display.max_colwidth", 10000)
import matplotlib.pyplot as plt

def roundnum(num):
    rounded_num = int(num*1000)/1000
    return rounded_num

def cylinder(height, rot, tilt):
    rho = 10
    height += rho

    x = rho * np.sin(tilt) * np.cos(rot)
    y = rho * np.sin(tilt) * np.sin(rot)
    z = rho * np.cos(tilt)
    x_start = roundnum(x)
    y_start = roundnum(y)
    z_start = roundnum(z)

    x = height * np.sin(tilt) * np.cos(rot)
    y = height * np.sin(tilt) * np.sin(rot)
    z = height * np.cos(tilt)
    x_end = roundnum(x)
    y_end = roundnum(y)
    z_end = roundnum(z)
    return  f'.cylinder {x_start} {y_start} {z_start} {x_end} {y_end} {z_end} 0.1 \n'

def deg2rad(deg):
    rad = deg * 2 * np.pi / 360
    return rad
    





file = '7_5_projs.star'

last=99999999
header = ''

with open(file, 'r') as f:
    for num,line in enumerate(f,1):
        if '_rln' in line:
            last=num
        elif(num>last):
            break
        header += line
new_list=[]
res_list = (header.strip().split('\n'))

res_list = [x.strip(' ') for x in res_list]

new_list = res_list[5:]

cols = new_list

d1 = pd.read_csv(file,sep='\s+',skiprows=last, names = cols, header=None)
Angle_Rot = d1['_rlnAngleRot #1']
Angle_Tilt = d1['_rlnAngleTilt #2']

def eachI(all_orients,x):
    Rot = Angle_Rot[x]
    Tilt = Angle_Tilt[x]
    try:
        all_orients[(Rot,Tilt)] += 1
    except:
        all_orients[(Rot,Tilt)] = 1
    return all_orients

all_orients0 = {}
for x in range(len(Angle_Rot)):
    all_orients0 = eachI(all_orients0,x)

max0 = all_orients0[max(all_orients0, key=all_orients0.get)]
all_orients0 = {k: 10*v / max0 for k, v in all_orients0.items()}

for x in all_orients0:
    with open('bild_test/test0.bild', 'a+') as f:
        f.write(cylinder(0.1*all_orients0[x],deg2rad(x[0]),deg2rad(x[1])))
