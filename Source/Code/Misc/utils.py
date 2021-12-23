'''
Created on 01/01/2020
Modified on 17/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''

# Other imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import csv
import math
import cv2
import os
import pdb

import numpy as np

from tensorflow.python.client import device_lib

# Utilities Class
class Utilities:

    @staticmethod
    # Returns a list of available gpus
    def get_available_gpus():
        try:
            local_device_protos = device_lib.list_local_devices()
            return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU'][0].split(':')[2].split(',')[0].strip()
        except(IndexError):
            return None

    @staticmethod
    def set_cpu_or_gpu(parameters):
            # Set CPU or GPU type
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
            if parameters.gpu == False: 
                os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # Set CPU
            else: 
                os.environ["CUDA_VISIBLE_DEVICES"] = parameters.gpu_id
        
    @staticmethod
    def count_files(datapath):    
        n=0
        for root,dir,files in os.walk(datapath):
            for file in files:
                n+=1
        NumImages=n
        print ("IMAGES in:",datapath,NumImages)
        return  NumImages
        
    @staticmethod
    def dict_to_csv(data_dict, datapath):    
        try:
            with open(datapath, 'w') as f:
                for key in data_dict.keys():
                    f.write("%s, %s\n"%(data_dict[key],key))
        except IOError:
            print("I/O error")
            
