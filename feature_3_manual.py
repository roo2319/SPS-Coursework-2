from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
CLASS_COLOURS = [CLASS_1_C,CLASS_2_C,CLASS_3_C]
train_set, train_labels, test_set, test_labels = load_data()
int_labels = list(map(lambda x: int(x),train_labels))
colour_list = list(map(lambda x : CLASS_COLOURS[x-1], int_labels))

plt.rc('figure', figsize=(13, 7), dpi=110)
plt.rc('font', size=6)
fig= plt.figure()
p=0

for i in range(train_set.shape[1]):
    if i == 6:continue
    if i < 6: p=i+1
    if i > 6: p=i
    ax = fig.add_subplot(3,4,p, projection='3d')
    ax.scatter(train_set[:,6],train_set[:,9],train_set[:,i], color=colour_list,s=10, depthshade=True)
    ax.set_title("{} vs {} vs {}".format(6,9,i))

plt.tight_layout()
plt.show()
