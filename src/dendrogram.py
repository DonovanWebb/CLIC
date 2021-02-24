# Init
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()

# Load data
from sklearn.datasets import load_diabetes

# Clustering
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list
from scipy.spatial import distance
#from fastcluster import linkage # You can use SciPy one too
from scipy.cluster.hierarchy import linkage

# Color mapping
dflt_col = "#808080"   # Unclustered gray
# Dataset
A_data = load_diabetes().data
DF_diabetes = pd.DataFrame(A_data, columns = ["attr_%d" % j for j in range(A_data.shape[1])])

# Absolute value of correlation matrix, then subtract from 1 for disimilarity
DF_dism = 1 - np.abs(DF_diabetes.corr())

# Compute average linkage
A_dist = distance.squareform(DF_dism.values)
Z = linkage(A_dist,method="average")

D_leaf_colors = {"attr_1": dflt_col,

                 "attr_4": "#B061FF", # Cluster 1 indigo
                 "attr_5": "#B061FF",
                 "attr_2": "#B061FF",
                 "attr_8": "#B061FF",
                 "attr_6": "#B061FF",
                 "attr_7": "#B061FF",

                 "attr_0": "#61ffff", # Cluster 2 cyan
                 "attr_3": "#61ffff",
                 "attr_9": "#61ffff",
                 }

# notes:
# * rows in Z correspond to "inverted U" links that connect clusters
# * rows are ordered by increasing distance
# * if the colors of the connected clusters match, use that color for link
link_cols = {}
for i, i12 in enumerate(Z[:,:2].astype(int)):
  c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors["attr_%d"%x]
    for x in i12)
  link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col

# Dendrogram
D = dendrogram(Z=Z, labels=DF_dism.index, color_threshold=None,
  leaf_font_size=12, leaf_rotation=45, link_color_func=lambda x: link_cols[x])
# Apply the right color to each label
label = [0, 1, 0, 0, 0, 1, 1, 1, 2, 0]
colors = ['r', 'b', 'g']
ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
num = 0
for lbl in xlbls:
    lbl.set_color(colors[label[num]])
    num += 1
plt.show()
