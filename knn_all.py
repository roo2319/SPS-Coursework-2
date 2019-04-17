import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
CLASS_COLOURS = [CLASS_1_C,CLASS_2_C,CLASS_3_C]

train_set, train_labels, test_set, test_labels = load_data()

def calculate_accuracy(gt_labels, pred_labels):
    return np.sum(gt_labels==pred_labels)/float(len(gt_labels))


def feature_extract(feature1,feature2):
    reduced_train = train_set[:,[feature1,feature2]]
    reduced_test  = test_set [:,[feature1,feature2]]
    return reduced_train,reduced_test

def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    #I think this is correct. I need to refactor it anyway.
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
    k_nearest_points = lambda x: sorted([(dist(x,point[0]),point[1]) for point in zip(train_set,train_labels)], key = lambda x: x[0])
    k_nearest_neighbours = lambda x,k: [m[1] for m in k_nearest_points(x)[:k]]
    classification = lambda x,k: int(max(set(k_nearest_neighbours(x,k)),key=k_nearest_neighbours(x,k).count))
    return [classification(p,k) for p in test_set]

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
    plt.colorbar(handle)
    plt.show()


accuracy_mat = np.zeros((13,13))
for i in range(13):
    for j in range(13):
        r_tr, r_te = feature_extract(i,j)
        accuracy_mat[i,j] = calculate_accuracy(knn(r_tr,train_labels,r_te,7),test_labels)
        
plot_matrix(accuracy_mat)
