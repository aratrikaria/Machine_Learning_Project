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

def main() :

#Load the Digits Data
    path = '../../Data/Digits/'
    data = np.load(path + 'Data.npz')
    features_train = data['X_train']
    labels_train = data['y_train']
    features_test = data['X_test']
    labels_test = data['y_test']
    print "Digits:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 
    classifierKnn(features_train,labels_train, features_test, labels_test)


#Plot data set 
'''labels=['sr','og','^b']
for i in [1,2,3]:
 plt.plot(x[y==i,0],x[y==i,1],labels[i-1]);
plt.xlabel('area');
plt.ylabel('compactness');
print y.shape
'''
def classifierKnn(features_train,labels_train, features_test,labels_test):
    metric='euclidean'
    k=2

    #Fit the specified classifier
    clf = neighbors.KNeighborsClassifier(n_neighbors = k, weights='distance',metric=metric)
    clf.fit(features_train,labels_train)

    #print "gx=",gx

    #Compute a prediction for every point in the grid
     
    for i in range (len(features_test)):
        x = features_test[i,:].reshape(1,-1)
        predictions = clf.predict(x)
        labels_test[i] = predictions
    #Save prediction file in Kaggle format
    #predictions = np.zeros(labels_test.shape)
    for i in range (len(labels_test)):
        print features_test[i,:],labels_test[i]
    
    kaggle.kaggleize(labels_test, "../Predictions/Digits/test.csv")

if __name__ == "__main__":main();

