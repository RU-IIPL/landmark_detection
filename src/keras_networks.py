# -*- coding: utf-8 -*-
"""
@author: Terada
"""
from keras.models import Sequential, Model
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, MaxPool2D
from keras.layers import Input, Convolution2D, AveragePooling2D, merge, Reshape, Activation, concatenate
from keras.regularizers import l2
#from keras.engine.topology import Container

def net7(input_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(input_size[0], input_size[1], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(28))
    return model

def lenet(input_size):
    model = Sequential()
    model.add(Conv2D(20, kernel_size=5, strides=1, activation='relu', input_shape=(input_size[0], input_size[1], 1)))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Conv2D(50, kernel_size=5, strides=1, activation='relu'))
    model.add(MaxPooling2D(2, strides=2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(28)) #activation='softmax'
    return model

def alexnet(input_size):
    model = Sequential()
    model.add(Conv2D(48, 11, strides=3, activation='relu', padding='same', input_shape=(input_size[0], input_size[1], 1)))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 5, strides=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28)) #activation='softmax'
    return model

def malti_net(input_size):
    inputs = Input(shape=(input_size[0], input_size[1], 1))
    conv1 = Conv2D(18, (3, 3), activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, (3, 3), activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    flat1 = Flatten()(pool4)
    fc1 = Dense(1000, activation='relu')(flat1)
    fc2 = Dense(500, activation='relu')(fc1)
    x_main = Dense(28, name='main')(fc2)
    x_sub1 = Dense(2, name='sub1', activation='softmax')(fc2)
    x_sub2 = Dense(5, name='sub2', activation='softmax')(fc2)
    model = Model(inputs=inputs, outputs=[x_main, x_sub1, x_sub2])

    return model
