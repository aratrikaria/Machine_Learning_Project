#Load Packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from scipy import linalg
import kaggle
from sklearn.naive_bayes import GaussianNB

#Load the Semiconductor Data
path = '../../Data/Semiconductor/'
data = np.load(path + 'Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']
print "Semiconductor:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 

#Fit the specified classifier
clf = GaussianNB()
clf.fit(features_train,labels_train)
predictions = clf.predict(features_test)
print  predictions
    
predictions = np.zeros(labels_test.shape)
kaggle.kaggleize(predictions, "../Predictions/Semiconductor/test.csv")


