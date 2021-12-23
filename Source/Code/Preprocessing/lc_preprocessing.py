'''
Created on 01/01/2020
Modified on 17/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''
# Land Cover Imports
from Preprocessing.data_loading import DataLoading      # Reads dataset images in different format (EUROSAT,SENTINEL_REGION,)

# Other Imports
import os

class LandCoverPreprocessing:
# Reads dataset images in different format (EUROSAT,SENTINEL_REGION,)

    def __init__(self, params):                                                            # Class Initialization (constructor) 
        # General Initializations
        self.params = params 

    def preprocess(self):
        #EuroSatClasses=sorted(os.listdir(self.params.eurosat_dataset_path))

        DataLoading.load_eurosat(self.params.eurosat_dataset_path, self.params.eurosat_dataset_training_set_path, self.params.eurosat_dataset_validation_set_path, self.params.test_set_perc, False, self.params.limit_per_class,  self.params.rescale, self.params.exclude_classes)
        
        return None
