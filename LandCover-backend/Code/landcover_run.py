'''
Created on 01/01/2020
Modified on 04/03/2020

@author: Fabrizio De Fausti, Francesco Pugliese, Angela Pappagallo, Erika Cerasti
'''

# Land Cover Imports
from Misc.utils import Utilities                                                # Utilities Class
from Initialization.init import Init                                            # Initialization Class
from Settings.settings import SetParameters                                     # Settings 
from Preprocessing.lc_preprocessing import LandCoverPreprocessing               # Preprocessing Class
from Classification.lc_classification import LandCoverClassification            # Classification Class
from Training.lc_training import LandCoverTraining                              # Training Class
from Testing.lc_testing import LandCoverTesting                                 # Testing Class
from Postprocessing.lc_postprocessing import LandCoverPostprocessing            # Postprocessing Class

# Other Imports
import platform
import time
import cv2
import os
import numpy as np
import pandas as pd
import sys
    
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import tensorflow as tf
import matplotlib.image as img

import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
#import collections

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow import keras
from tensorflow.keras import models
import cv2

import pdb

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

# Globals
times = []
default_config_file = "landcover.ini"                                                # Default Configuration File
gpu_desc = ""

# Operating System
OS = platform.system()                                                               # returns 'Windows', 'Linux', etc


## Configuration
# Command Line Parser

parser = Init.parser()                                                               # Define arguments parser

(arg1) = parser.parse_args()
config_file = arg1.conf


if arg1.mode is None: 
    lc_mode = None
else: 
    lc_mode = int(arg1.mode)

if config_file is None: 
    config_file = default_config_file                                                # Default Configuration File

## CONFIGURATION FILE PARSER
# Read the Configuration File
set_parameters = SetParameters("../Conf", config_file, OS) 
params = set_parameters.read_config_file()

if params.gpu==True:
    # GPUs
    gpu_desc = Utilities.get_available_gpus()                                            # Returns a list of available gpus

# Overwrite configuration file
if lc_mode is not None: 
    params.lc_mode = lc_mode

# Set the CPU or GPU
if gpu_desc == "" or gpu_desc == None:                                        # if gpu not available use CPU automatically
    params.gpu = False
Utilities.set_cpu_or_gpu(params)

# Changes number of classes in case of excluding some classes from the original dataset
if params.exclude_classes is not None: 
    exclude_classes_list = [el.strip().lower() for el in params.exclude_classes.split(',')]
    #params.model_output_n_classes -= len(exclude_classes_list)

# Select the action based on the chosen land cover mode 
if params.lc_mode == 0:                                     # Classification
    lc_classification = LandCoverClassification(params) 
    lc_classification.classify()
elif params.lc_mode == 1:                                   # Preprocessing
    lc_preprocessing = LandCoverPreprocessing(params) 
    lc_preprocessing.preprocess()
elif params.lc_mode == 2:                                   # Training
    lc_training = LandCoverTraining(params) 
    lc_training.train()
elif params.lc_mode == 3:                                   # Testing
    lc_testing = LandCoverTesting(params)
    lc_testing.test()
elif params.lc_mode == 4:                                   # Postprocessing
    lc_postprocessing = LandCoverPostprocessing(params)         
    lc_postprocessing.postp()

