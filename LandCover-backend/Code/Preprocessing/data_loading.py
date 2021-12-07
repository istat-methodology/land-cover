'''
Created on 01/01/2020
Modified on 17/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''

from sklearn.model_selection import train_test_split
from shutil import copy, copytree, rmtree
from Misc.utils import Utilities                        # Utilities Class

import os
import pdb
import numpy as np

class DataLoading:
# Reads dataset images in different format (EUROSAT,SENTINEL_REGION,)

    @staticmethod
    def load_eurosat(datapath, training_set_path, validation_set_path, test_split, shuffle, limit_per_class,  rescale, exclude_classes):

        print("##############################################################################")
        print("###############  PRE-PROCESSING DONE! ########################################")
        
        print("PreProc mode : creazione di una cartella di training e una cartella di test")

        print("Cartella del TRAINING Set:",training_set_path)
        print("Cartella del VALIDATION Set:",validation_set_path)
        
        test_split = test_split / 100
        if exclude_classes is not None: 
            exclude_classes_list = [el.strip().lower() for el in exclude_classes.split(',')]

        ######################################################################
        X_train_all=[]
        Y_train_all=[]
        Y_test_all=[]
        X_test_all=[]

        output_classes = {}
        n_class = 0   
        print("Cartella del datapath:",datapath)
        print( os.walk(datapath))
        for roots, dirs, files in os.walk(datapath,followlinks=True):
            print(roots, dirs, files)
            class_name = os.path.split(roots)[1]
            if exclude_classes is None or class_name.lower() not in exclude_classes_list : 
                DataFull_X=[]
                DataFull_Y=[]
                
                if (roots==datapath):
                    continue
                print ("Balanced split in:",roots)
                output_classes.update({class_name : n_class})

                for i in range(len(files)):
                    DataFull_X.append(roots+os.path.sep+files[i])
                    DataFull_Y.append(roots)

                    if limit_per_class is not None and i>=limit_per_class-1: 
                        break 

                X_train, X_test, Y_train, Y_test=train_test_split(DataFull_X,DataFull_Y,test_size=test_split,random_state=21081978)     
                        
                X_train_all=X_train_all+X_train
                Y_train_all=Y_train_all+Y_train
                X_test_all=X_test_all+X_test
                Y_test_all=Y_test_all+Y_test
                    
                n_class += 1

        Utilities.dict_to_csv(output_classes, os.path.join('../Conf', 'output_classes.csv'))
  
        print("N images split TRAIN:",len(X_train_all))
        print("N images split TEST:",len(X_test_all))
        ##################################################################  

        ##################################################################      
        def prepare_data(listFile,dest, output_classes):
            for file in listFile:

                #print(file)
                appo=file.split(os.path.sep)
                root="."
                dir= str(output_classes[appo[-2]])+'_'+appo[-2]
                filename=appo[-1]
                destFile=os.path.join(root,dest,dir,filename)
                if not os.path.exists(os.path.join(root,dest,dir)):
                    os.makedirs(os.path.join(root,dest,dir))
                copy(file,destFile)
                #print(file)
                #print(destFile)
            print("Copying Done!")
        
        print("Creating test data...")
        if os.path.exists(os.path.join('.',validation_set_path)):
            print("Removing already existing Validation dir")
            rmtree(os.path.join('.',validation_set_path))
        prepare_data(X_test_all, validation_set_path, output_classes)
        Utilities.count_files(validation_set_path)
        
        print("Creating train data...")
        if os.path.exists(os.path.join('.',training_set_path)):
            rmtree(os.path.join('.',training_set_path))
            print("Removing already existing Validation dir")
        prepare_data(X_train_all, training_set_path, output_classes)    
        Utilities.count_files(training_set_path)
        
        
        print("###############  PRE-PROCESSING DONE! ########################################")
        
    ##################################################################                  
    '''
    def saturate_and_normalize(band, saturation_level):
        band[np.where(band >= saturation_level)] = saturation_level
        return band / saturation_level

    #@staticmethod
    def load_eurosat_13_channels(datapath, training_set_path, validation_set_path, test_split, shuffle, limit_per_class,  rescale):
        # load data
        load_start_time = timeit.default_timer()
        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(datapath)))
        # loop over the input images
        data = []
        labels = []
        count = 0
        for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = gdal.Open(imagePath)            # Read geo-tiff at moment for the training set
            blue_band = saturate_and_normalize(image.GetRasterBand(2).ReadAsArray(), parameters.rgb_saturation_level)   # Read the band, saturate and normalize
            green_band = saturate_and_normalize(image.GetRasterBand(3).ReadAsArray(), parameters.rgb_saturation_level)  # Read the band, saturate and normalize
            red_band = saturate_and_normalize(image.GetRasterBand(4).ReadAsArray(), parameters.rgb_saturation_level)    # Read the band, saturate and normalize
            if parameters.nir == True: 
                nir_band = saturate_and_normalize(image.GetRasterBand(8).ReadAsArray(), parameters.other_bands_saturation_level)   # Read the band and saturate
            
            image = np.stack((nir_band, blue_band, green_band, red_band))

            data.append(image)
            # extract the class label from the image path and update the labels list
            label = imagePath.split(os.path.sep)[-2]
            if label == "AnnualCrop":
                label = 0
            elif label == "Forest":
                label = 1
            elif label == "HerbaceousVegetation":
                label = 2
            elif label == "Highway":
                label = 3
            elif label == "Industrial":
                label = 4
            elif label == "Pasture":
                label = 5
            elif label == "PermanentCrop":
                label = 6
            elif label == "Residential":
                label = 7
            elif label == "River":
                label = 8
            else :
                label = 9
            labels.append(label)
        
            if limit_per_class is not None:
                count += 1
                
                if count>limit_per_class:
                    break

        data_set = np.array(data, dtype="float")
        labels = np.array(labels)

        # split the data into a training set and a validation set
        indices = np.arange(data_set.shape[0])
        if shuffle == True: 
            np.random.shuffle(indices)
        data_set = data_set[indices]
        labels = labels[indices]
        
        num_validation_samples = int(round(validation_split * data_set.shape[0]))
        num_test_samples = int(round(test_split * data_set.shape[0]))
        train_set_x = data_set[:-(num_test_samples+num_validation_samples)]
        train_set_y = labels[:-(num_test_samples+num_validation_samples)]
        val_set_x = data_set[-(num_test_samples+num_validation_samples):-num_test_samples]
        val_set_y = labels[-(num_test_samples+num_validation_samples):-num_test_samples]
        test_set_x = data_set[-num_test_samples:]
        test_set_y = labels[-num_test_samples:]
        
        print ('\n\nLoading time: %.2f minutes\n' % ((timeit.default_timer() - load_start_time) / 60.))
       
        return [train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y]
    
        return None
        '''
    
    
    



    
