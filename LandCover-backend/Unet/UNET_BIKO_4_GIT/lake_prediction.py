import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import random
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import minmax_scale
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from collections import defaultdict
import pdb
import random
import imutils
import sys,csv
from collections import Counter #mia
import numpy as np
from tifffile import imsave
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl



plt.rcParams.update({'font.size': 20})

from model_unet_ww import *
from utils_ww import *
from dataset_creator_ww import *
from model_training_ww import *
from prediction_ww import show_prediction


#model fit parameter:

bc_size  = 32
n_epochs = 60


#Unet's parameters:

n_filters_unet = 64
k_size_unet    = 3
n_channels     = 12
height_crop    = 64
width_crop     = 64


base_path_lake='/mnt/users/catalano/waterways/'
base_path_lake_RGB=save_path='/mnt/users/catalano/waterways/'

name_model='model_unet_60_32'
model_path='./weights/'+name_model

output_folder   ='./output_pred/'
name_pred_file  = 'prediction_lake_'+'model_unet_{}_{}_{}'.format(n_channels,n_epochs,bc_size)

output_path_val= os.path.join(output_folder,name_pred_file)


save_fig_path = './output_pred/plot_prediction/'+'lake_' + name_model
x_train_RGB = np.load(base_path_lake_RGB+'x_train_lake_rgb.npy')

# x_train = np.load(base_path_lake+'x_train_lake.npy')


# model= get_unet_64(n_ch = n_channels, patch_height = height_crop, patch_width = width_crop,n_classes=N_Cls)
# model.load_weights(path_model)

# prediction_data = model.predict(x_train)

#np.save(output_path_val,prediction_data)

pred_data=np.load(output_path_val+".npy")


def show_prediction(x,y_pred,idx_fig,trs=0.5):


    n_fig=4
    
    
    #print(idx_fig)

    f,axx=plt.subplots(n_fig,2,figsize=(10,10))

    axx[0,0].set_title("RGB Image")
    axx[0,1].set_title('Predicted  Mask')
    
    i=0
    for  idx in range(idx_fig[0],idx_fig[1]):
        
        
        
        image=np.transpose(x[idx,:],(1,2,0))
    
        image=percentile_cut(image,2,98)
    
        axx[i,0].imshow(image,figure=f)
    
    
        
        
       
        tmp_idx=y_pred[idx,0,:,:]
        tmp=np.where(tmp_idx>trs,1,0)
        
        
        axx[i,1].imshow(tmp,cmap='Greys_r')
       
    
        f.tight_layout()
        
        i+=1
        
    plt.close()

    


    
    return f




from matplotlib.backends.backend_pdf import PdfPages

index_list=np.arange(0,201,4)

chunk_index_list=[index_list[i:i+2] for i in range(len(index_list))][:-1]

pdf = PdfPages(save_fig_path + '_plots_pred'+ ".pdf")


for i in tqdm(chunk_index_list):
          
    
    fig=show_prediction(x_train_RGB[0:200],pred_data[0:200],i,trs=0.45) 

    pdf.savefig(fig)

    # destroy the current figure
    # saves memory as opposed to create a new figure
    plt.clf()
    

pdf.close()


