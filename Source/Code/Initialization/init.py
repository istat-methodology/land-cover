'''
Created on 01/01/2020
Modified on 17/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''

import timeit
import random
import argparse
import pdb

# Initialization Class
class Init:

    def __init__(self, parameters, set_seed):                                        # Class Initialization (constructor) 
        # General Initializations

        if set_seed == True: 
            random.seed(42)                                                          # Set seed
        
        self.valid_set_split = parameters.valid_set_perc / 100
        self.test_set_split = parameters.test_set_perc / 100
        self.global_start_time = timeit.default_timer()
        parameters.valid_set_perc = self.valid_set_split
        parameters.test_set_perc = self.test_set_split

        # Keras Initializations 
        self.deepnetworks = []
        self.default_callbacks = []
               
        self.parameters = parameters 

    def read_initialization(self):
        return [self.valid_set_split, self.test_set_split, self.global_start_time, self.deepnetworks, self.default_callbacks, self.parameters]
        
    def parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--conf", help="configuration file name", required = False)
        parser.add_argument("-m", "--mode", help="Land Cover mode", required = False)
        
        return parser