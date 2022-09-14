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

print(X_train[0])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train[0])




