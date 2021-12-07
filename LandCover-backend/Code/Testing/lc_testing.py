'''
Created on 01/01/2020
Modified on 17/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''
from Misc.utils import Utilities                                                # Utilities Class
import numpy as np

# Keras Imports
from tensorflow.keras.models import load_model
from Models.advanced_cv_model import AdvancedCVModel
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import os
import pdb
import sys

class LandCoverTesting:
    def __init__(self, params):                                                            # Class Initialization (constructor) 
        # General Initializations
        self.params = params 


    def test(self):

        def get_abs_path_model_name():
            filenamelist=[]
            print ("self.params.model_testing_dir",self.params.model_testing_dir)
            print ("self.param.output_training_path",self.params.output_training_path)
            print ("self.params.models_path",self.params.models_path)

            def find_file_name(dirName):
                for filename in os.listdir(dirName):
                    if filename.endswith("hdf5"):
                        filenamelist.append(filename)
    
                if len(filenamelist)!=1:
                    print("model not found")
                    sys.exit(1)
                return filenamelist[0]

            
            if self.params.model_testing_dir=="":
                # se non è settata la cartella dove leggere il modello 
                # allora si assegna quella di default Output_Training
                self.params.model_testing_dir=self.params.output_training_path
                abs_training_dir=self.params.model_testing_dir
                found_file_name=find_file_name(abs_training_dir)
                return self.params.model_testing_dir+os.sep+found_file_name,abs_training_dir
            else:
                abs_training_dir=os.sep.join([self.params.models_path,self.params.output_training_path,self.params.model_testing_dir])
                found_file_name=find_file_name(abs_training_dir)
                return os.sep.join([abs_training_dir,found_file_name]),abs_training_dir,found_file_name


            #return self.params.models_path+os.sep+filenamelist[0]


        training_abs_path_model_name,abs_training_dir,found_file_name=get_abs_path_model_name()
        print(training_abs_path_model_name)

        #IMAGINET_WEIGTHS='imagenet' # 'imagenet' None
        BATCH_SIZE=self.params.train_batch_size 
        #N_CLASSES=11
        model_file_name = training_abs_path_model_name.split('_')
        print (model_file_name)
        # gets the personalized parameters from the file name of the model
        #self.params.neural_model = model_file_name[8]
        #self.params.model_input_width = int(model_file_name[3])
        #self.params.model_input_height = int(model_file_name[3])

        print("Performance Mode Is Starting:")
        
        model = AdvancedCVModel.build(self.params)

        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        model.load_weights(training_abs_path_model_name)
        
        #model = load_model(self.params.models_path+os.sep+self.params.model_prefix+self.params.model_file, compile = True)
        
        print("On testing..")
        test_datagen = ImageDataGenerator(rescale=1. / 255)


        if self.params.eurosat_input_channels==4:
            cm='rgba'
        if self.params.eurosat_input_channels==3:
            cm='rgb'






        test_generator = test_datagen.flow_from_directory(self.params.eurosat_dataset_validation_set_path,target_size=(self.params.model_input_width, self.params.model_input_height),batch_size=BATCH_SIZE,class_mode='categorical', shuffle=False,color_mode=cm)
        #test_generator = test_datagen.flow_from_directory(TRAININGSET_DIR,target_size=(MODEL_IMG_WIDTH, MODEL_IMG_WIDTH),batch_size=BATCH_SIZE,class_mode='categorical', shuffle=False)
        print(Utilities.count_files(self.params.eurosat_dataset_validation_set_path))
        goldStandard=test_generator.labels

        
        print("\nLABELS FROM TEST DIR NAME\n")
        labelInferFromTestDir=pd.DataFrame.from_dict(test_generator.class_indices, orient='index').reset_index()
        labelInferFromTestDir.columns=["ClassName","ClassValue"]
        print (labelInferFromTestDir)
        print("\nLABELS FROM TRAINING MODEL METADATA\n")
        
        print(pd.read_csv(abs_training_dir+os.sep+self.params.output_classes))
        print(len(goldStandard))
        nb_validation_samples=len(goldStandard)
        score = model.predict_generator(test_generator, nb_validation_samples/BATCH_SIZE)#, pickle_safe=False)




        predStat=np.argmax(score,1)
        df = pd.DataFrame(goldStandard,columns=['GoldStand'])
        df['predStat'] = pd.DataFrame(predStat)

        dummy=pd.merge(labelInferFromTestDir,pd.merge(labelInferFromTestDir,df,left_on="ClassValue",right_on='GoldStand'),left_on="ClassValue",right_on='predStat')
        dummy.columns=['GoldStandardName', 'ClassValue_x', 'predStatName', 'ClassValue_y', 'GoldStand', 'predStat']
        df=dummy
        crossTab_tot=pd.crosstab(df.GoldStandardName,df.predStatName,margins=True)
        print(crossTab_tot)
       

        def accuracy(appo):
            appo=np.array(appo)[:-1,:-1]
            df=pd.DataFrame(appo).reset_index().melt("index")
            ALLTRUE=df[df["index"]==df["variable"]]["value"].sum()
            return ALLTRUE/np.sum(appo)

        #from sklearn.metrics import accuracy_score
        #for row in df.values:
        #    print (row)
        #print("1: ",accuracy_score(df['GoldStandardName'], df['predStatName']))
        #print("2: ", accuracy_score(df['GoldStand'], df['predStat']))

        # ACCURACY CALCUTATION
        print("Accuracy of the Full Confusion Matrix: " + str(accuracy(crossTab_tot)))


        def funDetectedOtherValue(labelInferFromTestDir):
                def detect(s):
                    if "other" in s.lower():
                        print("-- OTHER DETECT --: ",s)
                        return True
                    else:
                        print("------------------: ",s)
                        return False

                n=labelInferFromTestDir[labelInferFromTestDir["ClassName"].apply(detect)]["ClassName"].values[0]
                return n

        df=df[df.GoldStandardName!=funDetectedOtherValue(labelInferFromTestDir)]
        crossTab=pd.crosstab(df.GoldStandardName,df.predStatName,margins=True)
        print(crossTab)



        print("Accuracy of the Confusion Matrix without the class other: " + str(accuracy(crossTab)))
        #with open(self.params.output_path+os.sep+self.params.model_testing_file.split('.')[0].replace('linux', self.params.OS.lower()).replace('windows', self.params.OS.lower())+'_test_confusion_matrices.txt', "w") as text_file:
        with open('test_confusion_matrices'+found_file_name+'.txt', "w") as text_file:
            print("model name:",found_file_name+'\n', file=text_file)
            print("Testing Directory :",self.params.eurosat_dataset_validation_set_path+'\n', file=text_file)

            print(self.params.test_output_header+'\n', file=text_file)
            print(test_generator.class_indices, file=text_file)
            print('\n Full Confusion Matrix \n', file=text_file)
            print(crossTab_tot, file=text_file)            
            print('\n'+"Accuracy of the Full Confusion Matrix: " + str(accuracy(crossTab_tot))+'\n\n', file=text_file)
            print('\n Confusion Matrix without the class other \n', file=text_file)
            print(crossTab, file=text_file)            
            print('\n'+"Accuracy of the Confusion Matrix without the class other: " + str(accuracy(crossTab))+'\n\n', file=text_file)
        
        #pdb.set_trace()
            
            
        #exls=pd.ExcelWriter(self.params.output_path+os.sep+self.params.model_prefix+'test_confusion_matrices_.xlsx')
        #stats_LUCAS.to_excel(excel_writer=exls,sheet_name="stats_LUCAS")
        #stats.to_excel(excel_writer=exls,sheet_name="stats_CORINE")
        #exls.close()
