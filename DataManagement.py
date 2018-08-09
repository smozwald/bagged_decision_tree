# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:25:05 2018

@author: SAM
"""

import pandas as pd
import numpy as np

def data_prep_single_file(data_folder,filename, delimiter = ",", data_split = 0.8):
    raw = pd.read_csv(data_folder + filename, delimiter = delimiter)
    if raw.iloc[:,-1].dtype != 'int64':
        raw_classes = raw.iloc[:,-1].unique()
        class_key = {}
        for i in range (0, len(raw_classes)):
            class_key[raw_classes[i]] = i
        ##Change classifications to ints
        raw_class = raw.iloc[:,-1]
        raw_name = raw_class.name##To use pandas replaces
        #raw_length = len(raw_class)
        raw_classified = raw.replace({raw_name:class_key})
        raw_train = raw_classified.sample(frac=0.8,random_state=40)
        raw_test = raw_classified.drop(raw_train.index)
        data = raw_classified.values
        train = raw_train.values
        test = raw_test.values
    else:
        raw_train = raw.sample(frac=data_split, random_state=40)
        raw_test = raw.drop(raw_train.index)
        data = raw.values
        train = raw_train.values
        test = raw_test.values
    return train, test, data
    
def data_prep_two_files(data_folder,train_name, test_name, delimiter = ","):
    raw_train = pd.read_csv(data_folder + train_name, delimiter = delimiter)
    raw_test = pd.read_csv(data_folder + test_name, delimiter = delimiter)
    if raw_train.iloc[:,-1].dtype != 'int64':
        raw_classes = raw_train.iloc[:,-1].unique()
        class_key = {}
        for i in range (0, len(raw_classes)):
            class_key[raw_classes[i]] = i
        ##Change classifications to ints
        raw_class = raw_train.iloc[:,-1]
        raw_name = raw_class.name##To use pandas replaces
        #raw_length = len(raw_class)
        raw_train_classified = raw_train.replace({raw_name:class_key})
        raw_test_classified = raw_train.replace({raw_name:class_key})
        train = raw_train_classified.values
        test = raw_test_classified.values
        data = np.append(train, test, axis = 0)
    else:
        train = raw_train.values
        test = raw_test.values
        data = np.append(train, test, axis = 0)
    return train, test, data