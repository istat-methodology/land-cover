'''
Created on 01/01/2020
Modified on 17/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''

# Other imports
import os
import sys
import pdb

class ModelPreparation(object):
    
    @staticmethod
    def topology_selector(params):
        if params.neural_model.lower() == "satellitenet":
            topology = SatelliteNet
        else:
            print("Model does not exist.")
            sys.exit()
        
        return topology

    @staticmethod
    def build_eurosat_deepnetwork(params, model_prefix):
        summary = True
        if test == True: 
            summary = False
        
        opt = DataPreparation.get_training_algorithm(params)                                            # Training Algorithm Selection                                                                                  

        topology = DataPreparation.topology_selector(params)                                            # Topology Selection

        deepnetwork = topology.build(width=params.input_size[0], height=params.input_size[1], depth=params.input_channels, classes=params.output_size, summary=summary)
        if test == True: 
            if params.skip_model_prefix == False: 
                model_path = os.path.join("./",params.models_path)+"/"+model_prefix+params.model_file
            else:
                model_path = os.path.join("./",params.models_path)+"/"+params.model_file
                
            if os.path.isfile(model_path):
                deepnetwork.load_weights(model_path)   
            else:
                print('\nPre-trained model not found in : %s.' % (params.models_path))
                sys.exit("")
        deepnetwork.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return deepnetwork
    # unused method! marie kondo
    #@staticmethod
    #def model_selector(params):
    #    if params.eurosat_data == True:                                                   # EuroSat Model Building
    #        deepnetwork = self.build_eurosat_deepnetwork(params, model_prefix)
    #    
    #    return deepnetwork
        
