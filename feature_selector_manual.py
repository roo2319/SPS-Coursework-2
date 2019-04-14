from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
CLASS_COLOURS = [CLASS_1_C,CLASS_2_C,CLASS_3_C]

plt.rc('figure', figsize=(12, 8), dpi=110)
plt.rc('font', size=12)

train_set, train_labels, test_set, test_labels = load_data()

n_features = train_set.shape[1]
fig, ax = plt.subplots(n_features, n_features)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
int_labels = list(map(lambda x: int(x),train_labels))
colour_list = list(map(lambda x : CLASS_COLOURS[x-1], int_labels))

for i in range(n_features):
    for j in range(n_features):
        ax[i][j].scatter(train_set[:,i],train_set[:,j], color=colour_list,s=1)
        ax[i][j].set_title("{} vs {}".format(i,j))

plt.show()