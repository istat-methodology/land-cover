'''
Created on 16/02/2021
Modified on 16/02/2021

@author: Fabrizio De Fausti, Erika Cerasti
'''

import cv2
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

from Misc.funforStatistiche import statistiche, print_Mappa

class LandCoverPostprocessing:
    def __init__(self, params):                                                            # Class Initialization (constructor)
        # General Initializations
        print("LAND COVER STATISTICS AND MAPS CREATOR")
        self.params = params

    def postp(self):
        print ("ciao")
        print(self.params.input_classification_matrix_file)
        nome_fileinput = self.params.input_classification_matrix_file
        nome_model = nome_fileinput.split(".hdf5")[0] 
        print(nome_model+'/')
        

'''
class LandCoverClassification:
    def __init__(self, params):                                                            # Class Initialization (constructor)
        # General Initializations
        print("LAND COVER CLASSIFIER AND MAP CREATOR")
        self.params = params


    def classify(self):


        def get_abs_path_model_name():
            filenamelist=[]
            print ("self.params.model_classification_dir",self.params.model_classification_dir)
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


            if self.params.model_classification_dir=="":
                # se non è settata la cartella dove leggere il modello
                # allora si assegna quella di default Output_Training
                self.params.model_classification_dir=self.params.output_training_path
                abs_training_dir=self.params.model_classification_dir
                found_file_name=find_file_name(abs_training_dir)
                return self.params.model_classification_dir+os.sep+found_file_name,abs_training_dir
            else:
                abs_training_dir=os.sep.join([self.params.models_path,self.params.output_training_path,self.params.model_classification_dir])
                found_file_name=find_file_name(abs_training_dir)
                return os.sep.join([abs_training_dir,found_file_name]),abs_training_dir,found_file_name


            #return self.params.models_path+os.sep+filenamelist[0]


        classification_abs_path_model_name,abs_classification_dir,found_file_name=get_abs_path_model_name()
        print(classification_abs_path_model_name)



        if True:
            #otherStat=other_stat(cv2.imread("Classification/other.png"))
            if self.params.other_image_type=="Texture":
                otherStat=other_stat(cv2.imread("Classification/other_Texture.jpg"))
            if self.params.other_image_type=="Full":
                otherStat=other_stat(cv2.imread("Classification/other_Full.tif"))


            print("MapsStatsCreate mode")

            ############################################################################

            img_input=cv2.imread(self.params.input_image)
            if self.params.input_image.endswith(".tif"):
                print("..tif file identified",self.params.input_image)
                import gdal
                #import numpy as np
                dataset = gdal.Open(self.params.input_image, gdal.GA_ReadOnly)
                array1 = dataset.GetRasterBand(1).ReadAsArray()
                print("band 1 loaded")
                array2 = dataset.GetRasterBand(2).ReadAsArray()
                print("band 2 loaded")
                array3 = dataset.GetRasterBand(3).ReadAsArray()
                print("band 3 loaded")
                img=np.stack((array1,array2,array3))
                del array1
                del array2
                del array3
                img_input=np.rollaxis(img, 0, 3)

            if img_input is None:
                print("input image not found:",self.params.input_image)
                sys.exit(-1)
            #plt.imshow(img_input)
            ni,nj,nc=img_input.shape
            K.clear_session()
            #model= load_model(self.params.model_classify_file,compile = True)




            print("Performance Mode Is Starting:")

            model = AdvancedCVModel.build(self.params)


            model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
            print("----------------")
            #print(self.params.models_path+os.sep+self.params.model_classify_file)
            model.load_weights(classification_abs_path_model_name)

            #model = load_model(self.params.models_path+'/'+self.params.model_prefix+self.params.model_file, compile = True)

            if self.params.summary==True:
                model.summary()


            map_nrow=1 + (self.params.eurosat_input_size//self.params.stride)*((ni//self.params.eurosat_input_size) - 1)
            map_ncol=1 + (self.params.eurosat_input_size//self.params.stride)*((nj//self.params.eurosat_input_size) - 1)
            V_fast=np.ones( (map_nrow ,map_ncol),dtype=np.int8 )*10
            N_tiles=map_nrow * map_ncol
            #pred_generator = pred_datagen.flow_from_directory(TilesDir,shuffle=False,target_size=(img_height, img_width),batch_size=batch_size,class_mode=None)
            ### ################## #############
            #pred=model.predict_generator(pred_generator)

            print("Map Original:N righe e colonne",ni,nj)
            print("Map low resolution: N righe e colonne",ni//self.params.eurosat_input_size,nj//self.params.eurosat_input_size)

            print("Rifinitura mappa (0 0 se rifilata):",ni-ni//self.params.eurosat_input_size*self.params.eurosat_input_size,nj-nj//self.params.eurosat_input_size*self.params.eurosat_input_size)
            print("Map effective resolution: N righe e colonne",map_nrow,map_ncol)
            print("Aggiustamento rifilatura...")

            rif_i=ni-ni//self.params.eurosat_input_size*self.params.eurosat_input_size
            rif_j=nj-nj//self.params.eurosat_input_size*self.params.eurosat_input_size
            img_input= img_input[:ni-rif_i,:nj-rif_j,:]
            ni,nj,nc=img_input.shape

            print("Rifinitura mappa (0 0 se rifilata):",ni-ni//self.params.eurosat_input_size*self.params.eurosat_input_size,nj-nj//self.params.eurosat_input_size*self.params.eurosat_input_size)
            print("Map effective resolution: N righe e colonne",map_nrow,map_ncol)
            print("N_tiles: ",N_tiles)





            ########


            ### BUFFER ASIMMETRICO
            buffAsimm=np.zeros((ni//self.params.eurosat_input_size,nj//self.params.eurosat_input_size),dtype=np.int8)
            for i in range(ni//self.params.eurosat_input_size):
                for j in range(nj//self.params.eurosat_input_size):
                    i_start=i*self.params.eurosat_input_size
                    i_end=(i+1)*self.params.eurosat_input_size
                    j_start=j*self.params.eurosat_input_size
                    j_end=(j+1)*self.params.eurosat_input_size
                    img_i_j_shii_shij= img_input[i_start:i_end,j_start:j_end,:]
                    if i_end<=ni and j_end<=nj:
                        if is_other(img_i_j_shii_shij,otherStat):
                            #print("OTHER",i,j)
                            pass
                        else:
                            #print("NO OTHER",i,j)
                            buffAsimm[i,j]=1
                            buffAsimm[i-1,j]=1
                            buffAsimm[i-1,j-1]=1
                            buffAsimm[i,j-1]=1
            ### BUFFER ASIMMETRICO

            #%%
            ### PLANNING
            PlanningTuple=[]
            for i in range(ni//self.params.eurosat_input_size):
                for j in range(nj//self.params.eurosat_input_size):
                    if buffAsimm[i,j]==1:
                        for shift_i in range(self.params.eurosat_input_size//self.params.stride):
                            for shift_j in range(self.params.eurosat_input_size//self.params.stride):
                                PlanningTuple.append((i,j,shift_i,shift_j))


            N_planning=len(PlanningTuple)
            print("PlanningTuple",N_planning)


            ########

            cont_img=0
            cont_landImg=0
            img_batch=[]
            coord_batch=[]
            #other_mask=[]


            ms0 = time.time()*1000.0

            for i,j,shift_i,shift_j in PlanningTuple:

                                cont_img+=1
                                if cont_img%(N_planning//1000)==0:
                                    print("immagini processate: "+str(cont_img)+" ("+str(int(cont_img/N_planning*100))+"%)",cont_landImg,time.time())
                                i_start=i*self.params.eurosat_input_size+shift_i*self.params.stride
                                i_end=(i+1)*self.params.eurosat_input_size+shift_i*self.params.stride
                                j_start=j*self.params.eurosat_input_size+shift_j*self.params.stride
                                j_end=(j+1)*self.params.eurosat_input_size+shift_j*self.params.stride

                                if i_end<=ni and j_end<=nj:

                                    # crop
                                    img_i_j_shii_shij= img_input[i_start:i_end,j_start:j_end,:]

                                    img_out=img_i_j_shii_shij
                                    #if is_other(img_out,otherStat):
                                    #   other_mask.append((i,j,shift_i,shift_j))
                                    #   #print ("OTHER",(i,j,shift_i,shift_j))
                                    #   V_fast[i*(self.params.eurosat_input_size//self.params.stride)+shift_i][j*(self.params.eurosat_input_size//self.params.stride)+shift_j]=10
                                    #   continue
                                    cont_landImg+=1
                                    #print("--------NO OTHER",(i,j,shift_i,shift_j))
                                    coord_batch.append((i,j,shift_i,shift_j))

                                    #scombio i canali BGR con RGB perchè cv2 read in BGR mentre mentre il load di keras si aspetta
                                    #RGB
                                    img_out = cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB)

                                    #ret,img_out_=cv2.imencode(".jpg",img_out)

                                    #img_out=cv2.imdecode(img_out_,flags=cv2.IMREAD_COLOR)



                                    img_out=cv2.resize(img_out,(self.params.dim_model_input_image ,self.params.dim_model_input_image),interpolation=cv2.INTER_NEAREST)

                                    img_out=img_out/255.

                                    img_batch.append(img_out)
                                    #if cont_img%self.params.test_batch_size==0:
                                    if cont_landImg%self.params.test_batch_size==0:
                                        #print("BATCH FULL")
                                        #print("immagini processate: "+str(cont_img)+" ("+str(int(cont_img/N_tiles*100))+"%)")
                                        array_img=np.asarray(img_batch)

                                        pred=model.predict_on_batch(array_img)
                                        predStat=list(np.argmax(pred,1))
										##### NO OTHER FILLING ######
                                        for (i,j,sh_i,sh_j),prev in zip(coord_batch,predStat):
                                            #print(prev)
                                            V_fast[i*(self.params.eurosat_input_size//self.params.stride)+sh_i][j*(self.params.eurosat_input_size//self.params.stride)+sh_j]=prev

                                        img_batch=[]
                                        coord_batch=[]

            array_img=np.asarray(img_batch)
            pred=model.predict_on_batch(array_img)
            predStat=list(np.argmax(pred,1))

            ms1 = time.time()*1000.0
            print("time",ms1-ms0)


            for (i,j,sh_i,sh_j),prev in zip(coord_batch,predStat):
                V_fast[i*(self.params.eurosat_input_size//self.params.stride)+sh_i][j*(self.params.eurosat_input_size//self.params.stride)+sh_j]=prev

            print("Pint V_fast in CSV")
            #np.savetxt(self.params.model_classify_file+"_"+self.params.tag+"_"+str(self.params.stride)+".csv",V_fast,delimiter=";",fmt='%i')
            np.savetxt(found_file_name+"_"+self.params.tag+"_"+str(self.params.stride)+".csv",V_fast,delimiter=";",fmt='%i')



            nomefile_classif = found_file_name+"_"+self.params.tag+"_"+str(self.params.stride)+".csv"
            #V_fast = np.genfromtxt(filepath, delimiter=';', dtype=np.uint8)
            V_fast = pd.read_csv(found_file_name+"_"+self.params.tag+"_"+str(self.params.stride)+".csv")

	        ############ CREAZIONE STATISTICHE DA MATRICE ###########################

            ########## PARAMETER ############
            regione = "PUGLIA"
            #regione = "PUGLIA"
            mapping = "LUCAS"
            ##################

            ### READ OUTPUT CLASSES FILE: class legend
            class_leg = pd.read_csv(abs_classification_dir+os.sep+"output_classes.csv", sep=',')#, header=None)
            class_leg.columns =['Legend','Class'] # MODIFICA FABRIZIO
            #class_leg.columns =['Class', 'Legend'] # MODIFICA FABRIZIO
            print(class_leg)

            def clear_Legend(x):
                return x.split("_")[-1].lower().strip()

            class_leg['Legend']= class_leg['Legend'].apply(clear_Legend)
            #class_leg['Legend']= class_leg['Legend'].str.strip()
            ### READ MAPPING FILE
            #mapFile = pd.read_csv(mappingfile, sep=';')
            mapFile = pd.read_csv("Classification/mapping_classes_LUCAS.csv", sep=';')

            x = V_fast

            #nome file di Output
            outputdir="./"
            nomefile = ''.join([outputdir, "Statistiche_", regione])
            nomefile_mappa = ''.join([outputdir, "Mappa_", regione])

            listofcolor=None
            mappa_eurosat=None

            classes_color=pd.read_csv("Classification/classes_color.csv",sep=";",header=None)
            classes_color.columns=["Color","Legend"]
            print(classes_color)


            listofcolor=list(classes_color.Color.values)

            statistiche(x, class_leg, mapFile, nomefile, mapping)
            print_Mappa(x, class_leg, listofcolor, nomefile_mappa, mappa_eurosat, mapping, mapFile)


        return None

'''
