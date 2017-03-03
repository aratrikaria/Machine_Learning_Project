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
from sklearn import linear_model

def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
        predictions.shape = [predictions.shape[0],1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids,predictions)).astype(int)
    writer = csv.writer(open(file, 'w'))
    writer.writerow(['ID','Category'])
    writer.writerows(kaggle_predictions)


#Load the EmailSpam Data
path = '../../Data/EmailSpam/'
data = np.load(path + 'Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']
print "EmailSpam:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 

#USE higher C value
clf = linear_model.LogisticRegression(C=2000)
clf.fit(features_train,labels_train)


#Save prediction file in Kaggle format
predictions = np.zeros(labels_test.shape)
kaggle.kaggleize(predictions, "../Predictions/EmailSpam/Emailtest.csv")
