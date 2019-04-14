#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
CLASS_COLOURS = [CLASS_1_C,CLASS_2_C,CLASS_3_C]

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    

def calculate_accuracy(gt_labels, pred_labels):
    return np.sum(gt_labels==pred_labels)/len(gt_labels)

def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of 
    # the function

            
    return [0,6]

#Helper function to get features out
def feature_extract(train_set, test_set, feature1,feature2):
    reduced_train = train_set[:,[feature1,feature2]]
    reduced_test  = test_set [:,[feature1,feature2]]
    return reduced_train,reduced_test

def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    #I think this is correct. I need to refactor it anyway.
    features = feature_selection(train_set,train_labels)
    r_tr, r_te = feature_extract(train_set,test_set,features[0],features[1])
    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
    k_nearest_points = lambda x: sorted([(dist(x,point[0]),point[1]) for point in zip(r_tr,train_labels)], key = lambda x: x[0])
    k_nearest_neighbours = lambda x,k: [m[1] for m in k_nearest_points(x)[:k]]
    classification = lambda x,k: int(max(set(k_nearest_neighbours(x,k)),key=k_nearest_neighbours(x,k).count))
    return [classification(p,k) for p in r_te]



def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    #PCA FOR n_components = 2
    covariance = np.cov(train_set,rowvar=False)
    eigenval,eigenvec = np.linalg.eig(covariance)
    eigenvec = eigenvec.transpose()

    #Sort eigenvectors on decreasing eigenvalues
    s_eigenvec = [x for y,x in sorted(zip(eigenval,eigenvec),reverse=True)]
    #Create a transformation matrix (potential refactor)
    w = np.transpose(np.vstack((s_eigenvec[0],s_eigenvec[1])))

    #Transform the dataset
    w_train = np.dot(train_set , w)
    w_test = np.dot(test_set, w)
    return knn(w_train,train_labels,w_test,k)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
        print(test_labels)
        print(calculate_accuracy(test_labels,predictions))
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))