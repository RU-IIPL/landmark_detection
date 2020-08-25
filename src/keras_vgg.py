# -*- coding: utf-8 -*-
"""
@author: Terada
"""
from keras.models import Sequential, Model
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, MaxPool2D
from keras.layers import Input, Convolution2D, AveragePooling2D, merge, Reshape, Activation, concatenate
from keras.regularizers import l2

def vgg16(input_size):
    input = Input((input_size[0], input_size[1], 1))
    kernel_size = (3, 3)
    max_pool_size = (2, 2)
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block1_pool')(x)
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block2_pool')(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block3_pool')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block4_pool')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block5_pool')(x)

    flattened = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(flattened)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x_main = Dense(28, name='main')(x)
    model = Model(inputs=input, outputs=x_main)

    return model

def vgg16mtl(input_size):

    input = Input((input_size[0], input_size[1], 1))
    kernel_size = (3, 3)
    max_pool_size = (2, 2)
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv2D(64, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block1_pool')(x)
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, kernel_size, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block2_pool')(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block3_pool')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block4_pool')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, kernel_size, activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D(max_pool_size, strides=(2, 2), padding='same', name='block5_pool')(x)

    flattened = Flatten()(x)
    x1 = Dense(1024, activation='relu', name='fc1_1')(flattened)
    x1 = Dropout(0.5, name='dropout1_1')(x1)
    x2 = Dense(1024, activation='relu', name='fc1_2')(flattened)
    x2 = Dropout(0.5, name='dropout1_2')(x2)
    x3 = Dense(1024, activation='relu', name='fc1_3')(flattened)
    x3 = Dropout(0.5, name='dropout1_3')(x3)
    x4 = Dense(1024, activation='relu', name='fc1_4')(flattened)
    x4 = Dropout(0.5, name='dropout1_4')(x4)
    x1 = Dense(512, activation='relu', name='fc2_1')(x1)
    x2 = Dense(512, activation='relu', name='fc2_2')(x2)
    x3 = Dense(512, activation='relu', name='fc2_3')(x3)
    x4 = Dense(512, activation='relu', name='fc2_4')(x4)
    x_main = Dense(28, name='main')(x1)
    x_sub1 = Dense(2, name='sub1', activation='softmax')(x2)
    x_sub2 = Dense(5, name='sub2', activation='softmax')(x3)
    x_sub3 = Dense(42, name='sub3')(x4)
    model = Model(inputs=input, outputs=[x_main, x_sub1, x_sub2, x_sub3])

    return model

def vgg16_ex(input_size):
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(input_size[0], input_size[1], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28)) #1000, activation='softmax'

    return model

def vgg19_ex(input_size):
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(input_size[0], input_size[1], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28)) #1000, activation='softmax'

    return model
