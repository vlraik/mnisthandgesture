'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import gzip
import numpy as np
from keras.models import load_model

batch_size = 2048
num_classes = 36
epochs = 400

# input image dimensions
img_rows, img_cols = 28, 28

with gzip.open('emnist-letters-train-images-idx3-ubyte.gz') as f:
    trainfile = np.frombuffer(f.read(),np.uint8, offset=16)
trainfile = trainfile.reshape(-1,784)

with gzip.open('emnist-digits-train-labels-idx1-ubyte.gz') as f:
    trainlabel = np.frombuffer(f.read(),np.uint8,offset=8)

with gzip.open('emnist-digits-train-images-idx3-ubyte.gz') as f:
    trainfile2=np.frombuffer(f.read(),np.uint8, offset=16)
trainfile2 = trainfile2.reshape(-1,784)

with gzip.open('emnist-letters-train-labels-idx1-ubyte.gz') as f:
    trainlabel2 = np.frombuffer(f.read(),np.uint8,offset=8)

trainlabel2 = [x+9 for x in trainlabel2]

trainfile = np.concatenate((trainfile,trainfile2))
trainlabel = np.concatenate((trainlabel,trainlabel2))


# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, y_train) = trainfile, trainlabel

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)


if os.path.isfile("emnist_train.h5"):
    print("Loading model from the SSD...")
    model = load_model("emnist_train.h5")
else:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,)
#score = model.evaluate(x_test, y_test, verbose=0)
model.save('emnist_train.h5')
#print('Test loss:', score[0])
