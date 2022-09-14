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

#this line basically confirms that there is only one dimension to the numpy array at this point

class_num = y_test.shape[1]


# creating layers of a nueral net
model = keras.Sequential()

#adding the first layer
#padding ensures that all values are of same input length
#there are 64 filters to this layer and there are 3 dimension to each of these filters or "averages"
#the relu activation is more of a all or none type of summation

model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
#max pooling maximizes the value that each feature can take on
model.add(keras.layers.MaxPooling2D(2))
#this destroys connections to prevent overfitting
model.add(keras.layers.Dropout(0.2))
#ensures that new nuerons arent created with too much to it 
model.add(keras.layers.BatchNormalization())
    

#same thing as above except it has more filters(sugested to go by factors of 2), doesnt need max pooling because its taking data thats already been normalized?
model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

#preparation for output
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

# find difference between flatten and dense?

#preparation for classification?
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())


#output layer
model.add(keras.layers.Dense(class_num, activation='softmax'))

#run network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'val_accuracy'])




