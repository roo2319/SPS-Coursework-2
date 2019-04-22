from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
CLASS_COLOURS = [CLASS_1_C,CLASS_2_C,CLASS_3_C]

plt.rc('figure', figsize=(12, 12), dpi=110)
plt.rc('font', size=10)

train_set, train_labels, test_set, test_labels = load_data()
cormat = np.zeros((2,train_set.shape[1]))
for j in range(train_set.shape[1]):
    _,_, r,_,_ = stats.linregress(train_labels,train_set.transpose()[j,:])
    cormat[0,j] = abs(round(r,3))
    cormat[1,j] = cormat[0,j]
#n_features = train_set.shape[1]
def plot_matrix(matrix, ax=None):
    """
    Displays a given matrix as an image.
    
    Args:
        - matrix: the matrix to be displayed        
        - ax: the matplotlib axis where to overlay the plot. 
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`. 
          If you do not explicitily create a figure, then pass no extra argument.  
          In this case the  current axis (i.e. `plt.gca())` will be used        
    """    
    if ax is None:
        ax = plt.gca()
        
    # write your code here
    handle = ax.imshow(matrix,cmap=plt.get_cmap('hot'))
    plt.colorbar(handle)
    return ax

plot_matrix(cormat)
plt.show()