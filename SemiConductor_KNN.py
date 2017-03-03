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
#warnings.filterwarnings('default')

import kaggle

import csv
import numpy as np

#Save prediction vector in Kaggle CSV format
#Input must be a Nx1, 1XN, or N long numpy vector
def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
        predictions.shape = [predictions.shape[0],1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids,predictions)).astype(int)
    writer = csv.writer(open(file, 'w'))
    writer.writerow(['ID','Category'])
    writer.writerows(kaggle_predictions)


def main() :

#Load the Semiconductor Data
    path = '../../Data/Semiconductor/'
    data = np.load(path + 'Data.npz') 
    features_train = data['X_train']
    labels_train = data['y_train']
    features_test = data['X_test']
    labels_test = data['y_test']
    print "Semiconductor:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape

def classifierKnn(features_train,labels_train, features_test,labels_test):
    metric='euclidean'
    k=4

    #Fit the specified classifier
    clf = neighbors.KNeighborsClassifier(n_neighbors = k, weights='distance',metric=metric)
    clf.fit(features_train,labels_train)


    #Compute a prediction for every point in the grid
     
    for i in range (len(features_test)):
        x = features_test[i,:].reshape(1,-1)
        predictions = clf.predict(x)
        labels_test[i] = predictions
    #Save prediction file in Kaggle format
    #predictions = np.zeros(labels_test.shape)
    for i in range (len(labels_test)):
        print features_test[i,:],labels_test[i]
    
    kaggle.kaggleize(labels_test, "../Predictions/Submission/test.csv")

if __name__ == "__main__":main();

