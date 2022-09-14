import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils


# for result replication
seed = 21

#importing data set
from keras.datasets import cifar10


# importing the test and train data, which are in this case the same thing
# in a real world application the test or applied data would need to be different from my understanding
#currently each image is jsut a tuple of values, or a ndarray, nd array is just a classification of tuples special to numpy

(X_train, y_train), (X_test, y_test) = cifar10.load_data()



#changing the data type for the values in the numpy array


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#making the numbers on a scale of 0-1

X_train = X_train / 255.0
X_test = X_test / 255.0

#this splits the data into categories for the nueral net to evaluate and process
#this also allows us to know how many end point nuerons we need

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



class_num = y_test.shape[1]
