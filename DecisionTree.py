# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:20:03 2018
Decision Tree classifier, used to classify datasets with any number of continuous attributes.


@author:Samuel Oswald
"""
##Import numpy for management of arrays.
import numpy as np

"""Build decision tree. data refers to the training dataset.
max_depth refers to how deep the tree can get. min_size is the minimum
amount of samples before a leaf node must be classified."""
def dt_train( data, max_depth, min_size = 1):
    max_depth = int(max_depth)
    min_size = int(min_size)
    attr, split_val, left, right = split(data)
    tree = {"attribute": attr, "split": split_val, "left": left, "right": right, "current_mode": leaf(data)}
    decision(tree,max_depth,min_size)
    return tree

def gini(node):
    """Calculate the gini impurity for a node. Aim is to minimize gini impurity(gain function)."""
    ##Find the number of classifications in current node.
    classifications = node[:,-1]
    samples = classifications.size
    unique, counts = np.unique(classifications, return_counts = True)
    ##calculate gini based on number of classes
    gini = 1
    for i in range (0, unique.size):
        proportion =  counts[i] / samples
        ##
        gini = gini - proportion * proportion
    return gini
    
def gain(values, cur_gini, attribute, split):
    """Calculate information gain for an attribute split at each level.
    Inputs are the current subset of data, initial gini at parent node,
    attribute to be split and split number."""
    i = attribute
    samples = values[:,-1].size
    left = values[values[:,i] < split, :]
    right = values[values[:,i] >= split, :]
    left_samples = left[:,-1].size
    right_samples = right[1:,-1].size
    
    ##Calculate left and right side gini
    left_gini = gini(left)
    right_gini = gini(right)
    
    ##Calculate information gain at this split value.
    gain = cur_gini - (left_samples/samples)*left_gini - (right_samples/samples)*right_gini
    return gain, left, right
    
def split(node):
    """Find the ideal split point by searching for the best information gain
    of all attributes and their potential split values.
    If no gain improves, node is split for leaf node creation as right side left at 0 samples."""
    cur_gini = gini(node)
    best_gain = 0
    best_attr = 0
    best_split = 0
    ##Implement greedy, exhaustive search for best information gain
    variables = len(node[0])
    best_left = node
    best_right = np.empty([0,variables])
    
    ##Seach through each unique value to find best division
    for v in range(0, variables-1):
        uniques = np.unique(node[:, v])
        for row in uniques:
            new_gain, left, right  = gain(node, cur_gini, v, row)
            
            ##Select the best gain, and associated attributes
            if new_gain > best_gain:
                best_gain = new_gain
                best_attr = v
                best_split = row
                best_left = left
                best_right = right
    #return {"attribute": best_attr, "split": best_split, "left": best_left, "right": best_right}
    return best_attr, best_split, best_left, best_right

def leaf(node):
    """Return classification value for leaf node, 
    when either maximum depth of tree reached or node is suitably weighted to one class."""
    classes = node[:, -1].tolist()
    return max(set(node[:,-1]), key = classes.count)

def decision(tree, max_depth=10, min_size=0, depth=0):
    """Uses split and leaf functions to build a tree, using a root data set.
    Will assign leaf nodes if either maximum depth or minimum samples are reached.
    root node contains both current node data, as well as decision rules to that point.
    """
    left = tree["left"]
    right = tree["right"]
      
    ##If tree is at max depth, assign most common member.
    if depth >= max_depth:
        tree['left'] = leaf(left)
        tree['right'] = leaf(right)
    ##If continuing sampling
    else:
        
        ##Left side child
        ##If minimum samples exist in current node, make it a leaf with max occuring value in samples.
        if left[:, -1].size <= min_size:
            tree['left'] = leaf(left)
        ##Else continue building tree.
        else:
            left_attr, left_split, left_left, left_right = split(left)
            ##Check if node is terminal. Make it a leaf node if so.
            if left_left.size == 0 or left_right.size == 0:
                tree['left'] = leaf(np.vstack([left_left,left_right]))   
            ##Continue elsewise.
            else:
                tree['left'] = {"attribute": left_attr, "split": left_split, "left": left_left, "right": left_right, "current_mode": leaf(left)}
                decision(tree['left'], max_depth, min_size, depth+1)
                
        ##right side child. Same process as above.
        if right[:, -1].size <= min_size:
            tree['right'] = leaf(right)
        else:
            right_attr, right_split, right_left, right_right = split(right)
            if right_left.size == 0 or right_right.size == 0:
                tree['right'] = leaf(np.vstack([right_left,right_right]))
            else:
                tree['right'] = {"attribute": right_attr, "split": right_split, "left": right_left, "right": right_right, "current_mode": leaf(right)}
                decision(tree['right'], max_depth, min_size, depth+1)

def classify(tree,row):
    """classify new data based on current row.
    Involves searching through tree based on the attributes of validation data.
    Will return classification value once leaf of tree is reached."""
    ##Look at each sample to classify. append to list of output values.
    ##Recursively search through branches until an append can be made.
    if row[tree['attribute']] < tree['split']:
        if isinstance(tree['left'],dict):
            return classify(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'],dict):
            return classify(tree['right'], row)
        else:
            return tree['right']

def dt_predict( tree, data):
    """For every row in the validation data,
    a call to the classify function is done,
    with results appended to prediction data."""
    predictions = []
    for row in data:
        pred = classify(tree, row)
        predictions.append(int(pred))
    return predictions

##functions for validation and pruning.
def dt_confusion_matrix( predicted, actual,classes):
    """Return a confusion matrix showing the difference between actual values,
    and model predicted values. Also returns total accuracy"""
    
    matrix = np.zeros((len(classes), len(classes)))
    for a, p in zip(actual, predicted):
        matrix[a][p] += 1
    accuracy = (actual == predicted).sum() / float(len(actual))*100
    return matrix, accuracy        
    
def print_dt(tree, depth = 0):
    """"Iterate through decision tree, printing out values."""
    print ((" " * depth) + "attribute " + str(tree['attribute']) + " > " + str(tree['split']))
    if isinstance(tree['left'], dict):
        print_dt(tree['left'], depth + 1)
    else:
        print ((" " *(depth + 1)) + str(tree['left']))
    if isinstance(tree['right'], dict):
        print_dt(tree['right'], depth + 1)
    else:
        print ((" " *(depth + 1)) + str(tree['right']))
        

"""Bagged decision trees contain a user-specified number of decision trees.
Classification of a sample is done by using the mode of each of these decision trees.
subsample is a fraction of the total dataset to be used.
trees refers to the number of trees to use in "forest" of trees.
By leaving default values for subsample and trees, a single decision tree classifier is created."""
def bt_train( data, max_depth, min_size = 1, subsample_ratio = 1,trees =1):
    
    ##Create a series of trees using sampling with replacement.
    size = data[:, -1].size
    division = int(size * subsample_ratio)        
    forest = []
    for i in range (0,trees):
        samples = data[np.random.choice(data.shape[0], division, replace = True)]
        forest.append([])
        forest[i] = dt_train(samples, max_depth, min_size)
    return forest

def bt_predict( forest, data):
    """"Classify validation data set based on built bagged trees.
    This is done by taking the mode of the classifications of each decision tree."""
    ##Use predict function from decision tree.
    ##Number of trees in forest, number of validation samples. Used to create empty array showing classifications.
    forest_size = len(forest)
    samples = len(data)
    tree_classification = np.zeros((samples, forest_size))
    ##With each tree, find the classification of each validation sample.
    for i in range (0, forest_size):
        tree_classification[:, i] = dt_predict(forest[i], data)
    ##Create list of modes for each sample, using tree_classification matrix.
    predictions = []
    for i in range(0, samples):
        tree_pred = tree_classification[i,:].tolist()
        predictions.append(int(max(set(tree_pred), key = tree_pred.count)))
    return predictions

def bt_confusion_matrix( predicted, actual,classes):
    """Create confusion matrix for bagged trees. Makes call to DT method."""
    matrix, accuracy = dt_confusion_matrix(predicted, actual, classes)
    return matrix, accuracy
    
    
    