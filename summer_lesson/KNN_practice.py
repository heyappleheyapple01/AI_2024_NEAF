import numpy as np
np.set_printoptions(threshold=np.inf)  ## print all values of matrix without reduction
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance     ## calculate the distance between two points


## first part : load dataset(very famous called iris)
"""
your code
"""

## second part : choose the label that we want (can be based on your preference)
"""
your code
"""

## third part : plot the distribution of data
"""
your code
"""

## forth part : the principle of KNN
"""
your code 

1. determine the value of K, the category of data, and the test point
2. calculate the distance between the test point and all data
3. find the top k nearest data
4. select the categories with the most votes
"""

## fifth part : plot the test point

"""
your code
"""