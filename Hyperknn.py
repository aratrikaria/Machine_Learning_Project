import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.externals.six import StringIO  
#import pydot 
from subprocess import call

import warnings
warnings.filterwarnings('ignore')

import kaggle

def main() :

#Load the Digits Data
    path = '../../Data/Digits/'
    data = np.load(path + 'Data.npz')
    features_train = data['X_train']
    labels_train = data['y_train']
    features_test = data['X_test']
    labels_test = data['y_test']
    print "Digits:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 
    
    train1,train2 = np.vsplit(features_train,2)
    label1,label2 = np.split(labels_train,2)
    
    for k in range(2, 150):
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    