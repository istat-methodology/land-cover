'''
Created on 01/01/2020
Modified on 28/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class SatelliteNet:
    @staticmethod
    def build(width, height, depth, classes, summary, weightsPath=None):
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        else:
            inputShape = (height, width, depth)
        
        # initialize the model
        model = Sequential()
        
        # first set of CONV 
        model.add(Conv2D(48, (3, 3), padding='same',input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Conv2D(48, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        model.add(Dropout(0.5))
    
        # second set of CONV 
        model.add(Conv2D(96, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(96, (3, 3), padding='same'))

        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
        model.add(Dropout(0.5))
    
        # first (and only) set of FC 
        model.add(Flatten())
        model.add(Dense(512))
        
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dropout(0.5))
    
        # softmax classifier
        model.add(Dense(classes))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
                
        #if summary==True:
        model.summary()
        #plot_model(model, to_file='../Output/Metrics/model_plot.png', show_shapes=True, show_layer_names=True)


        #if a weights path is supplied (indicating that the model was pre-trained), then load the weights
        if weightsPath is not None: 
            model.load_weights(weightsPath)
    
        # return the constructed network architecture
        return model