import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys

import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint

from sklearn.metrics import confusion_matrix

df = pd.read_csv(sys.argv[1])

datasize  = df.shape[0]
imagesize = len(str(df.feature[0]).split())
width     = int(imagesize**0.5)
height    = int(imagesize**0.5)
datain_X  = np.empty((datasize, height, width,1))
datain_Y  = np.empty((datasize,1))

for i in range(datasize):
    datain_X[i] = np.array(str(df.feature[i]).split()).reshape((height,width,1))
    datain_Y[i,0] = df.label[i]
datain_X /= 255
 
index = list(range(datasize))
validsize = int(datasize/15)
trainsize = int(datasize-validsize)
random.seed(10)
random.shuffle(index)
valid_X = datain_X[index[:validsize]]
train_X = datain_X[index[validsize:]]
valid_Y = datain_Y[index[:validsize]]
train_Y = datain_Y[index[validsize:]]


num_classes = 7
input_shape = (height,width,1)

valid_Y = to_categorical(valid_Y, num_classes)
train_Y = to_categorical(train_Y, num_classes)

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', input_shape=input_shape))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=input_shape))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# Model Compiling
model.compile(loss      = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics   = ['accuracy'])
              
# Image PreProcessing
train_gen = ImageDataGenerator(rotation_range    = 30,
                               width_shift_range  = 0.2,
                               height_shift_range = 0.2,
                               shear_range        = 0.2,
                               zoom_range         = [1-0.2, 1+0.2],
                               horizontal_flip    = True)
train_gen.fit(train_X)

# Callbacks
callbacks = []
modelcheckpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
callbacks.append(modelcheckpoint)
csv_logger = CSVLogger('recorder.csv', separator=',', append=False)
callbacks.append(csv_logger)

batch_size = 128
epochs = 250

model.fit_generator(train_gen.flow(train_X, train_Y, batch_size = batch_size),
                    steps_per_epoch = 10*datasize//batch_size,
                    epochs          = epochs,
                    callbacks       = callbacks,
                    validation_data = (valid_X, valid_Y))

scoret = model.evaluate(train_X, train_Y, verbose=0)
scorev = model.evaluate(valid_X, valid_Y, verbose=0)
print('Train loss:', scoret[0])
print('Train accuracy:', scoret[1])
print('Valid loss:', scorev[0])
print('Valid accuracy:', scorev[1])
