import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from wine_classifier import knn, calculate_accuracy

train_set, train_labels, test_set, test_labels = load_data()
plt.rc('figure', figsize=(12, 12), dpi=110)



def calculate_confusion_matrix(gt_labels, pred_labels):
    confusion = np.zeros((3,3))
    for i in range(1,4):
        for j in range(1,4):
            confusion[i-1,j-1] = sum((pred_labels == np.ones(pred_labels.shape)*j) &\
                                 (gt_labels == np.ones(gt_labels.shape)*i))/    \
                                 float(sum(gt_labels == np.ones(gt_labels.shape)*i))
    return confusion

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
    handle = ax.imshow(matrix,cmap=plt.get_cmap('summer'))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,i,round(matrix[i,j],3))
    return ax

fig, ax = plt.subplots(3,3)
for i in range(0,6):
    plot_matrix(calculate_confusion_matrix(test_labels,np.array(knn(train_set,train_labels,test_set,i+1))),ax[i//3][i%3])
    ax[i//3][i%3].set_title("K={},accuracy={}".format(i+1,calculate_accuracy(test_labels,knn(train_set,train_labels,test_set,i+1))))
    ax[i//3][i%3].set_xticks([0,1,2])
    ax[i//3][i%3].set_yticks([0,1,2])

plot_matrix(calculate_confusion_matrix(test_labels,np.array(knn(train_set,train_labels,test_set,7))),ax[2][1])
ax[2][1].set_title("K=7,accuracy={}".format(calculate_accuracy(test_labels,knn(train_set,train_labels,test_set,7))))
ax[2][1].set_xticks([0,1,2])
ax[2][1].set_yticks([0,1,2])
fig.delaxes(ax[2,0])
fig.delaxes(ax[2,2])
plt.tight_layout()
plt.show()
    