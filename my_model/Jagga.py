#!/usr/bin/env python
# coding: utf-8

# In[5]:


# I will be creating a CNN for my dataset
# Importing packages

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as bk


# In[6]:


class Jagga:
    
    @staticmethod
    def build(width, height, depth, classes):
        
        # initialize the model along with input shape to be channels last and channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        
        if bk.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # Adding layers to our CNN
        # first CONV => RELU => CONV => RELU => POOL layer set
        
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        
        # second CONV => RELU => CONV => RELU => POOL layer set
        
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        
        # first (and only) set of FC => RELU layers
        
        model.add(Flatten())
        model.add(Dense(units=64))
        model.add(Activation(activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        
        # softmax classifier
        
        model.add(Dense(classes))
        model.add(Activation(activation="softmax"))
        
        # returning the model now
        return model

