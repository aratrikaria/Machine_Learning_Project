import kaggle

# Assuming you are running run_me.py from the Submission/Code
# directory, otherwise the path variable will be different for you
import numpy as np


#Load the Digits Data
path = '../../Data/Digits/'
data = np.load(path + 'Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']
print "Digits:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 

#Save prediction file in Kaggle format
predictions = np.zeros(labels_test.shape)
kaggle.kaggleize(predictions, "../Predictions/Digits/test.csv")



#Load the Semiconductor Data
path = '../../Data/Semiconductor/'
data = np.load(path + 'Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']
print "Semiconductor:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 

#Save prediction file in Kaggle format
predictions = np.zeros(labels_test.shape)
kaggle.kaggleize(predictions, "../Predictions/Semiconductor/test.csv")


#Load the EmailSpam Data
path = '../../Data/EmailSpam/'
data = np.load(path + 'Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']
print "EmailSpam:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 

#Save prediction file in Kaggle format
predictions = np.zeros(labels_test.shape)
kaggle.kaggleize(predictions, "../Predictions/EmailSpam/test.csv")


