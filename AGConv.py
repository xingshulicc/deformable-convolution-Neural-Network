# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Oct  9 15:15:48 2019

@author: Admin
"""

import tensorflow as tf

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import multiply
import numpy as np

from keras.layers import Lambda

from keras import backend as K

def squeeze_excite_block(input, ratio = 4):
    init = input
    channel_axis = -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(init)
#    the shape of se: (batch_size, channels)
    
    se = Reshape(se_shape)(se)
#    the shape of se: (batch_size, 1, 1, channels)
    
    se = Dense(filters // ratio, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False)(se)
    se = Dense(filters, activation = 'sigmoid', kernel_initializer = 'he_normal', use_bias = False)(se)
#    the shape of se: (batch_size, 1, 1, channels)
    
    x = multiply([init, se])
#    the shape of x: (batch_size, width, height, channels)
    
    return x, se
    

def adaptive_group(inputs, ratio):
    init = inputs
    x, se = squeeze_excite_block(input = init, ratio = ratio)
    se = Flatten()(se)
#    the shape of se: (batch_size, channels)
    se = tf.argsort(se, axis=1, direction='DESCENDING')
    


    
    
    
    

    
    

    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





































    
    
    
    
    