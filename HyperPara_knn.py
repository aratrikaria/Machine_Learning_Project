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
    
def classifierKnn(train1,label1, train2,label2):
    metric='euclidean'
    k=2
    
    clf = neighbors.KNeighborsClassifier(n_neighbors = k, weights='distance',metric=metric)
    clf.fit(features_train,labels_train)

    #Compute a prediction for every point in the grid
     
    for i in range (len(train2)):
        x = features_test[i,:].reshape(1,-1)
        predictions = clf.predict(x)
        labels_test[i] = predictions
    #Save prediction file in Kaggle format
    #predictions = np.zeros(labels_test.shape)
    for i in range (len(labels_test)):
        print features_test[i,:],labels_test[ssi]
    
    kaggle.kaggleize(labels_test, "../Predictions/Digits/test.csv")

if __name__ == "__main__":main();