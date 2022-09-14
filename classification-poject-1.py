import numpy
from tensorflow import keras
from keras.constraints import maxnorm
from keras.utils import np_utils


# for result replication
seed = 21

#importing data set
from keras.datasets import cifar10

print(seed)