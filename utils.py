# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from numpy.core.numeric import cross


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))#euclidean distance
    

    # raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1-x2))#manhattan distance

    # raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    if derivative:
        return np.ones([len(x),len(x[0])])#return the value 1 as derivative of x is 1
    else:
        return x#the value itself is returned

    # raise NotImplementedError('This function must be implemented by the student.')


#The activation functions below were implemented after refering this article: https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/

def sigmoid(x, derivative = False):
    x = np.clip(x, -1e105, 1e105)#clip the value between these values to avoid underflow and overflow
    sigmoid_value = 1 / (1 + np.exp(-x))
    if derivative:#if derivative is asked
        return sigmoid_value * (1 - sigmoid_value) 
    else:
        return sigmoid_value

    # raise NotImplementedError('This function must be implemented by the student.')


def tanh(x, derivative = False):
    # print('we are in tanh')
    x = np.clip(x, -1e105, 1e105)
    tanh_value = (2 / (1 + np.exp(-2 * x))) - 1
    
    if derivative:#derivative
        return (1 - tanh_value**2) 
    else:
        return tanh_value

    # raise NotImplementedError('This function must be implemented by the student.')


def relu(x, derivative = False):
    x = np.clip(x, -1e105, 1e105)
    if derivative:
        return np.greater(x,0).astype(int)#Line taken from https://stackoverflow.com/a/47380053
    else:
        return np.maximum(x,0)

    # raise NotImplementedError('This function must be implemented by the student.')


def softmax(x, derivative = False):
    x = np.clip(x, -1e105, 1e105)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))

#reference ended

def cross_entropy(y, p):
    np.seterr(divide='ignore', invalid='ignore')
    p = np.clip(p, -1e105,1e105)

    cross_entropy_value = -y * np.log(p) - (1-y) * np.log(1-p)
    # print(cross_entropy_value)
    return cross_entropy_value 
    # raise NotImplementedError('This function must be implemented by the student.')


def one_hot_encoding(y):
    unique_tagerts = np.unique(y)#find unique values
    one_hot_list = []
    for i in range(len(y)):
        output = [1 if y[i]==j else 0 for j in unique_tagerts]#create a matrix to create a list 
        one_hot_list.append(output)#append the list
    return np.array(one_hot_list)
    # raise NotImplementedError('This function must be implemented by the student.')
