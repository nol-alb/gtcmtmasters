import numpy as np
import scipy 
import matplotlib.pyplot as plt
from scipy.spatial import distance as edist


"""
Helper Functions
"""

"""
A. k-NN classifier: [30 points]
[25 points] Write a function [est_class] = knearestneighbor(test_data, train_data, train_label, k) that implements a k-NN classifier based on the Euclidean distance (L-2 norm). test_data and train_data are np arrays with the dimensions from the attached zip.
[5 points] In your report, clearly explain how you solve the equal distance problem in your implementation and why you think your approach makes sense. 
"""

def euclidiean(d_point1,d_point2):
    return edist.euclidean(d_point1,d_point2)

def knearestneighbor(test_data, train_data, train_label, k):
    distances = np.zeros(train_data.shape[0])
    pairs = np.empty((), dtype=object)
    pairs[()] = (0, 0)
    elementsdata = np.full((train_data.shape[0]), pairs, dtype=object)
    for i,rows in enumerate(train_data):
        distances[i] = euclidiean(rows,test_data)
        elementsdata[i] = (distances[i],train_label[i])
    elements = sorted(elementsdata, key=lambda t: t[0])
    nearest = np.full((k), pairs, dtype=object)
    for i in range(k):
        nearest[i] = elements[i]
    neighbours = [elements[i][1] for i in range(k)]
    counts = np.bincount(neighbours)
    est_class = np.argmax(counts)
    
    
    return (est_class)

def cross_validate(data, gt_labels, k, num_folds):
    num_classes = max(gt_labels)
    size_of_partition = int(data.size/num_folds)
    testing_set_data = np.zeros(size_of_partition)
    training_set_data = np.zeros((size_of_partition,num_folds-1))
    testing_set_labels = np.zeros(size_of_partition)
    training_set_labels = np.zeros((size_of_partition,num_folds-1))
    fold_accuracies = np.zeros(num_folds)
    for j in range(num_folds):    
        TP = 0
        data = np.roll(data,size_of_partition)
        for i in range(num_folds):
            start_index = size_of_partition*i
            end_index = start_index+size_of_partition
            if (end_index <=size_of_partition):
                testing_set_data = data[start_index:end_index]
                testing_set_labels= gt_labels[start_index:end_index]
            else:
                training_set_data[:,i-1] = data[start_index:end_index]
                training_set_labels[:,i-1] = gt_labels[start_index:end_index]
        for i in range(testing_set_data.size):
            res = knearestneighbor(testing_set_data[i], training_set_data.flatten(), training_set_labels.flatten(), k)
            if (res==testing_set_labels[i]):
                TP+=1
        fold_accuracies[j] = TP/testing_set_data.size
    
    avg_accuracy = np.mean(fold_accuracies)   
    return avg_accuracy,fold_accuracies

def find_best_features(data, labels, k, num_folds):
    avg_accuracies = np.zeros(data.shape[0])
    for i,j in enumerate(data):
        avg_accuracies[i],fold_accuracy = cross_validate(j, labels, k, num_folds)
    feature_index = np.argmax(avg_accuracies)
    return feature_index

