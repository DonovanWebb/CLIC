"""
Replots dendrogram from .npy file
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--z_file",  required=True, type=str)
args = parser.parse_args()

z_file = args.z_file

Z = np.load(z_file)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)

plt.show()
