'''
Created on 01/01/2020
Modified on 21/04/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''
# Land Cover Imports
from Misc.utils import Utilities                                                # Utilities Class
from Models.advanced_cv_model import AdvancedCVModel


# Keras Imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import os,sys
import pdb

import time

import shutil

class LandCoverTraining:
    
    def __init__(self, params):                                                            # Class Initialization (constructor) 
        # General Initializations
        self.params = params 
        
    def train(self):
        
        print("Training mode is starting")
        ###############################################################

        nb_train_samples=Utilities.count_files(self.params.eurosat_dataset_training_set_path)
        nb_validation_samples=Utilities.count_files(self.params.eurosat_dataset_validation_set_path)

        ### DATA AUGMENTATION #############
        
        if self.params.multiband_eurosat_data==False:
            if self.params.eurosat_input_channels==4:
                cm='rgba'
            if self.params.eurosat_input_channels==3:
                cm='rgb'

            train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True) #,rotation_range=0.5)
            train_generator = train_datagen.flow_from_directory(self.params.eurosat_dataset_training_set_path, color_mode=cm,target_size = (self.params.model_input_width, self.params.model_input_height), batch_size=self.params.train_batch_size, class_mode='categorical')

            test_datagen = ImageDataGenerator(rescale=1. / 255)
            validation_generator = test_datagen.flow_from_directory(self.params.eurosat_dataset_validation_set_path,color_mode=cm, target_size = (self.params.model_input_width, self.params.model_input_height), batch_size=self.params.train_batch_size, class_mode='categorical')

        elif self.params.multiband_eurosat_data==True:
            train_datagen = ImageDataGenerator(rescale=1., horizontal_flip=True, vertical_flip=True) #,rotation_range=0.5)
            train_generator = train_datagen.flow_from_directory(self.params.eurosat_dataset_training_set_path, target_size = (self.params.model_input_width, self.params.model_input_height), batch_size=self.params.train_batch_size, class_mode='categorical')

            test_datagen = ImageDataGenerator(rescale=1. )
            validation_generator = test_datagen.flow_from_directory(self.params.eurosat_dataset_validation_set_path, target_size = (self.params.model_input_width, self.params.model_input_height), batch_size=self.params.train_batch_size, class_mode='categorical')
        else:
            print("Define multiband_eurosat_data parameter  :) !!!!!")
            sys.exit(1)
        ##############################################

        def write_classes_association(dict_classification):
            import pandas as pd
            df=pd.DataFrame.from_dict(dict_classification, orient='index').reset_index()
            df.columns=["ClassName","ClassValue"]
            df.to_csv(self.params.output_training_path+os.sep+self.params.output_classes,sep=",",index=False)


        def write_TrainingMetadati():
            fwrite=open(self.params.output_training_path+os.sep+"./landcoverClean.ini","w")
            for row in open("../Conf/landcover.ini"):
                row=row.strip()
                if len(row)>0:
                    if row[0]!="#":
                        fwrite.write(row+"\n")

        

        #model = AdvancedCVModel.build(self.params.neural_model, self.params.default_input_size, self.params.model_input_width, self.params.model_input_height, self.params.eurosat_input_channels, self.params.model_output_n_classes, self.params.imagenet_weights, self.params.summary,self.param)
        model = AdvancedCVModel.build(self.params)

        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        

        ModelTrainedName=self.params.output_training_path+os.sep+self.params.model_prefix+self.params.model_file

        csv_logger = CSVLogger(self.params.log_path+os.sep+self.params.model_prefix+'history.log')
        if self.params.save_best_model == True:
            print("SAVE BEST MODEL MODE",ModelTrainedName)
            check_point = ModelCheckpoint(ModelTrainedName, save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacksList=[csv_logger, check_point]
        else:
            callbacksList=[csv_logger]
        
        print("PRIMA DEL FIT")
        start_time = time.time()

        shutil.rmtree(self.params.output_training_path)
        os.mkdir(self.params.output_training_path)



        history = model.fit_generator(train_generator,
                            steps_per_epoch = nb_train_samples // self.params.train_batch_size,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples // self.params.train_batch_size,
                            epochs=self.params.epochs_number,
                            verbose=1,
                            callbacks=callbacksList
                                     )
        print("DOPO FIT")

        print("training time: --- %s seconds ---" % (time.time() - start_time))


        write_classes_association(train_generator.class_indices)
        write_TrainingMetadati()



        #model.save('model_trained_'+str(self.params.model_input_height)+"_"+str(self.params.epochs_number)+'.hdf5')
        ##############################################à
        DirModelTrainedName=ModelTrainedName.split(".")[0]

        try:
            shutil.copytree(self.params.output_training_path,self.params.models_path+os.sep+DirModelTrainedName)
        except:
            shutil.rmtree(self.params.models_path+os.sep+DirModelTrainedName)
            shutil.copytree(self.params.output_training_path,self.params.models_path+os.sep+DirModelTrainedName)
        return None
