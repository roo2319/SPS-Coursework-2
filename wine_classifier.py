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
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
CLASS_COLOURS = [CLASS_1_C,CLASS_2_C,CLASS_3_C]

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    

##Takes in class counts

def random_forest(train_set,train_labels,t):
    trees = []
    for i in range(t):
        train_bag = np.zeros(train_set.shape)
        train_bag_labels = []
        for j in range(train_set.shape[0]):
            index = random.randrange(0,len(train_set))
            train_bag[j] = train_set[index]
            train_bag_labels.append(train_labels[index])
        
        trees.append(decision_tree().build(train_bag,train_bag_labels))
    return trees


def impurity(classes,total):
    entropy = 0
    for i in range(len(classes)):
        if sum(classes[i]) != 0:
            probability =  list(map(lambda x: x/sum(classes[i]),classes[i]))
        else:
            probability = []
        entropy += (sum(classes[i])/total) * stats.entropy(probability)
    return entropy
     


class decision_tree:
    def __init__(self,left=None,right=None,feature=None, split=None,leaf=False):
        #Left tree
        self.left = left
        #Right tree
        self.right = right
        #The actual literal value in the condition, scope issues
        self.split = split
        #feature to split on
        self.feature = feature
        #Is leaf?
        self.leaf = leaf

    def build(self,observations,labels):
        if len(set(labels)) == 1:
            #If all labels are the same, we will make the node terminal
            self.split = labels[0]
            self.leaf = True
            return self
        else:
            #We need to find the best feature to split on
            score = np.zeros(observations.shape)
            for feature in range(observations.shape[1]):
                for sample in range(observations.shape[0]):
                    classCount = [[0,0,0],[0,0,0]]
                    #Compare each observation to the sample 
                    for i in range(observations.shape[0]):
                        if observations[i,feature] <= observations[sample,feature]:
                            classCount[0][int(labels[i])-1] += 1
                        else:
                            classCount[1][int(labels[i])-1] += 1

                    score[sample,feature] = impurity(classCount,len(observations))
            best = np.argmin(score)
            self.split = observations[best//2,best%2]
            self.feature = best%2
            lobservations = []
            llabels = []
            robservations = []
            rlabels = []
            for i in range(len(observations)):
                if observations[i,best%2] <= observations[best//2,best%2]:
                    lobservations.append(observations[i])
                    llabels.append(labels[i])
                else: 
                    robservations.append(observations[i])
                    rlabels.append(labels[i])

            #Convert observation types
            lobservations = np.array(lobservations)
            robservations = np.array(robservations)
            self.left = decision_tree().build(lobservations,llabels)
            self.right = decision_tree().build(robservations,rlabels)
            return self
            

        

    #Classify a single observation on a prebuilt model
    def classify(self,observation):
        if self.split == None:
            print("Have you built the model?")
            return None
        elif self.leaf:
            return self.split
        elif observation[self.feature] <= self.split:
            return self.left.classify(observation)
        else:
            return self.right.classify(observation)
    


def calculate_accuracy(gt_labels, pred_labels):
    return np.sum(gt_labels==pred_labels)/len(gt_labels)

def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of 
    # the function

            
    return [6,9]

#Helper function to get features out
def feature_extract(train_set, test_set, features):
    reduced_train = train_set[:,features]
    reduced_test  = test_set [:,features]
    return reduced_train,reduced_test

def knn_alg(train_set, train_labels, test_set, k, n):

    dist = lambda x, y: np.sqrt(np.sum((x-y)**2))


    #Return a list of tuples (dist,class) sorted on dist
    k_nearest_points = lambda x: sorted([(dist(x,point[0]),point[1]) for point in zip(train_set,train_labels)], key = lambda x: x[0])

    #Access the class given by the slice of size k of the sorted list
    k_nearest_neighbours = lambda x,k: [m[1] for m in k_nearest_points(x)[:k]]

    #The classification will be the mode
    classification = lambda x,k: stats.mode(k_nearest_neighbours(x,k))[0][0]
    return [classification(p,k) for p in test_set]

def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    features = feature_selection(train_set,train_labels)
    r_tr, r_te = feature_extract(train_set,test_set,features)
    scaler = StandardScaler()
    r_tr = scaler.fit_transform(r_tr)
    r_te = scaler.transform(r_te)
    return knn_alg(r_tr,train_labels,r_te,k,2)




def alternative_classifier(train_set, train_labels, test_set,trees, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    ## Let's go for decision trees
    #First we need to decide on the distribution. We will model as normal
    r_tr, r_te = feature_extract(train_set,test_set,feature_selection(train_set,train_labels))
    model = random_forest(r_tr,train_labels,trees)
    c = lambda p: stats.mode(list(map(lambda x: x.classify(p),model)))[0][0]
    return [c(p) for p in r_te]


    '''
    model = decision_tree()
    model.build(r_tr,train_labels)
    return[model.classify(observation) for observation in r_te]
    '''
        





def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    r_tr, r_te = feature_extract(train_set,test_set,[6,9,12])
    scaler = StandardScaler()
    r_tr = scaler.fit_transform(r_tr)
    r_te = scaler.transform(r_te)
    return knn_alg(r_tr,train_labels,r_te,k,3)


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    pca = PCA(n_components)
    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)
    pca.fit(train_set)
    w_train = pca.transform(train_set)
    w_test  = pca.transform(test_set)
    return knn_alg(w_train,train_labels,w_test,k,2)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--trees', nargs='?',type=int, default=16,help='Number of trees for random forest')
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

    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set,args.trees)
        print_predictions(predictions)


    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)


    elif mode == 'knn_pca':
        predictions = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)

    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))