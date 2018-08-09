# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:16:26 2018

##Implement the decision tree and bagged tree classifiers.

@author: Samuel Oswald
"""
##Import necessary classes. Decision Tree is custom class, numpy used for array processing,
##pandas used to manage overall data.
import DecisionTree as dt
import DataManagement as dm
import numpy as np

##Prep data. Index -1 assumed to be the dependent variable.
##Continuous data assumed for independent variables.
##Alter as needed for dataset.

data_folder = "Data/"
#image_folder = "Img/"

train_name = "iris.csv"
test_name = "XOR_train.csv"
filename = "Social_Network_Ads_Cont.csv"
#image_file = "APEX.hdr"
##Prep data using functions at bottom
#train, test, data = dm.data_prep_two_files(data_folder, train_name, test_name, ";")
##prep data using single file
train, test, data = dm.data_prep_single_file(data_folder, filename, ";")

##Regularize APEX dataset as it starts 1 index 1, not 0.
train[:,-1] = train[:,-1] - 1
test[:,-1] = test[:,-1] - 1
data[:,-1] = data[:,-1] - 1

##Decision tree classes(used for confusion matrix, automated in other processes)
#classes = np.unique(data[:,-1]-1).astype(int) ##Begin with class 0 for ease of indexing, if given classes start with 1. For APEX set.
classes = np.unique(data[:,-1]).astype(int) ##Begin with class 0 for ease of indexing, if given classes start with 1.


##Decision Tree Building
tree = dt.dt_train(train, 20,5)
validation_dt = dt.dt_predict(tree, test)
confusion_dt,accuracy_dt = dt.dt_confusion_matrix(validation_dt, test[:, -1].astype(int), classes)
print (accuracy_dt)
print (dt.print_dt(tree))

forest = dt.bt_train(train, 20, 5, 1, 500)
validation_rf = dt.bt_predict(forest, test)
confusion_rf, accuracy_rf = dt.bt_confusion_matrix(validation_rf, test[:, -1].astype(int),classes)
print (accuracy_rf)

##Apply learned decision tree to classification of imagery.
##matplotlib to plot, spectral to manipulate hyperspectral imagery.
#import matplotlib.pyplot as plt
#from spectral import io
#
#image = io.envi.open(image_folder + image_file)[:,:,:]
## plt.imshow(image[:,:, 0:3])
#shp = image.shape
#image_array = np.reshape(image, (shp[0] * shp[1], shp[2]))
#
###Make predictions
#dt_predict = dt.dt_predict(tree, image_array)
#bt_predict = dt.bt_predict(forest, image_array)
#
###Reshape and show predicted values
#dt_classified = np.asarray(dt_predict).reshape(shp[0], shp[1])
#bt_classified = np.asarray(bt_predict).reshape(shp[0], shp[1])
#
#plt.imshow(dt_classified)
#plt.show()
#plt.imshow(bt_classified)
#plt.show()