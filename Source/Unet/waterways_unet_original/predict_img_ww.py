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









def predict_img(model,img_all,size,n_steps_x,n_steps_y,patch,slide,padding):
 

    
    #img_all=normalize(img_all)
    img_all=np.pad(img_all, ((padding, padding),(padding,padding),(0,0)), 'symmetric')
    img_all=np.expand_dims(img_all.astype(float), axis=0)
    
    img = np.transpose(img_all, (0,3,1,2))
    img_all=standardize(img_all)
    print(img_all.shape)

    
    mask = np.zeros((n_steps_y*54,n_steps_x*54),'float')
    
    
    for i in tqdm(range(n_steps_y)):
 
        for j in range(n_steps_x):


                out_pred=model.predict(img[:,:,i*slide: patch+i*slide,j*slide:patch+j*slide])
                #print(out_pred.shape)
                mask[i*slide:slide+i*slide,j*slide:slide+j*slide]=out_pred[0,0,:,:]
                
    return mask        




if __name__ == '__main__':
        
    # for i in range(n_figures):

        # show_prediction(x,y,prediction,np.random.randint(len(x)),classes,save_fig_path,trs=0.1) 
        
        
    #if  os.path.exists(output_path_val + '.npy'):
    
        #prediction_val = np.load(output_path_val+ '.npy')
        #print(prediction_val.shape)
        
    
    #else:
    
    model= get_unet_64(n_ch = n_channels, patch_height = height_crop, patch_width = width_crop,n_classes=N_Cls)
    model.load_weights(path_model)
    
    save_path  ='/mnt/users/catalano/waterways/'
    folder_out='delta_po/'
    save_fig_path = './output_pred/plot_prediction/' +folder_out+ name_model

    file_land_name='delta_po_no_sea'
    
    img=np.load(save_path+ file_land_name + '.npy')
    
    print(img.shape)
    
    size_img=img.shape
    patches=64
    slides=54
    paddings=5
    
    n_steps_x=int(img.shape[1]/54)
    n_steps_y=int(img.shape[0]/54)
    
    sum_bands=np.sum(img[:n_steps_y*54,:n_steps_x*54],axis=2).astype(int)
    one_band=np.where(sum_bands==0,0,1)
  
    print("one band: ",one_band.shape)
    
    img_resized=img[:n_steps_y*54,:n_steps_x*54]
    
    print("img_res: ",img_resized.shape)
    
    if  os.path.exists(save_fig_path+"mask_"+file_land_name+ '.npy'):
    
        mask_img=np.load(save_fig_path+"mask_"+file_land_name +'.npy')
        print("mask_img:", mask_img.shape)
    
    else:
        mask_img=predict_img(model,img_resized,size_img,n_steps_x,n_steps_y,patch=patches,slide=slides,padding=paddings)
        np.save(save_fig_path+"mask_"+file_land_name+ '.npy',mask_img)
        print("mask_img:",mask_img.shape)

   
    
    trs=np.arange(0.01,0.7,0.05)
    cmap_my = mpl.colors.ListedColormap(['blue', 'black', 'yellow' ])
    
    for i in tqdm(trs):
    
        img_masked=np.where(mask_img<=i,1,2)
        
        img_masked=np.where(one_band*img_masked==0,0,img_masked)
       
        plt.imshow(img_masked,cmap=cmap_my,vmin=0,vmax=2)
        plt.savefig(save_fig_path+ "{}_mask_ww_{}.png".format(file_land_name,np.round(i,2)),dpi=1200)

