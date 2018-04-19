# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
"""
Created on Wed Apr 18 16:11:25 2018

@author: xingshuli
"""
import keras
from keras.datasets import mnist

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization

from layers import ConvOffset2D

from keras import backend as K

from keras.models import Model
#pre parameters
batch_size = 64
num_classes = 10
epochs = 50

#input image dimensions
img_rows, img_cols = 28, 28

#data split into train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#data pre-processing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#convert class vectors into binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#construct deformable cnn model for mnist dataset
def deform_cnn(input_tensor = None, input_shape = None, classes = 10, trainable = True):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
        bn_axis = 1
    else:
        input_shape = (img_rows, img_cols, 1)
        bn_axis = 3
    
    if input_tensor is None:
        inputs = Input(shape = input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor
    
    #Conv_1 layer
    x = Conv2D(32, (3, 3), padding = 'same', name = 'conv_1', trainable = trainable)(inputs)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation('relu')(x)
    
    #Conv_2 layer
    x_offset = ConvOffset2D(32, name = 'conv_2_offset')(x)
    x = Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', name = 'conv_2', trainable = trainable)(x_offset)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation('relu')(x)
    
    #Conv_3 layer
    x_offset = ConvOffset2D(64, name = 'conv_3_offset')(x)
    x = Conv2D(128, (3, 3), strides = (2, 2), padding = 'same', name = 'conv_3', trainable = trainable)(x_offset)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation('relu')(x)
    
    #Conv_4 layer
    x_offset = ConvOffset2D(128, name = 'conv_4_offset')(x)
    x = Conv2D(128, (3, 3), padding = 'same', name = 'conv_4', trainable = trainable)(x_offset)
    x = BatchNormalization(axis = bn_axis)(x)
    x = Activation('relu')(x)
    
    #Pooling layer
    x = GlobalAveragePooling2D()(x)
    
    #fc layer
    outputs = Dense(classes, activation = 'softmax', name = 'fc', trainable = trainable)(x)
    
    return inputs, outputs

#determine the input for tensorflow: channels_last
input_tensor = Input(shape = (img_rows, img_cols, 1))
inputs, outputs = deform_cnn(input_tensor = input_tensor, trainable = True)


#train model 
model = Model(inputs = inputs, outputs = outputs)
model.summary()

loss = keras.losses.categorical_crossentropy
optimizer = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, 
          epochs = epochs, validation_data = (x_test, y_test), verbose = 1)


#evaluate model after training
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
    
    
        