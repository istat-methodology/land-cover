#import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
#from shapely.wkt import loads as wkt_loads
import matplotlib.pyplot as plt
#import tifffile as tiff
#import os
#import random
#from keras.models import Model
#from keras.layers.normalization import BatchNormalization
#from keras.layers import *
#from keras.optimizers import Adam
#from keras.layers.merge import concatenate
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as K
#from sklearn.metrics import jaccard_similarity_score
#from sklearn.preprocessing import minmax_scale
#from shapely.geometry import MultiPolygon, Polygon
#import shapely.wkt
#import shapely.affinity
#from collections import defaultdict
#import pdb
#import random
#import imutils
import sys,csv
#from collections import Counter #mia
#import numpy as np
#from tifffile import imsave
#from tqdm import tqdm
#import matplotlib as mpl
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl



plt.rcParams.update({'font.size': 15})
#from model_unet_ww import *
#from utils_ww import *


#from dataset_creator_ww_sbagliato import *
from model_training_hw import *









def predict_img(model,img,valid_crop,patch,slide,padding):
 
    
    img_dim_x=int(img.shape[1]/valid_crop)
    img_dim_y=int(img.shape[0]/valid_crop)
    
    img_resized=img[:img_dim_y*valid_crop,:img_dim_x*valid_crop]
    
    shape_img_col=img_resized.shape[1]
    shape_img_row=img_resized.shape[0]
    
    n_steps_x=int(img_resized.shape[1]/slides)-1
    n_steps_y=int(img_resized.shape[0]/slides)-1
       
    
    img_resized=np.pad(img_resized, ((padding, padding),(padding,padding),(0,0)), 'symmetric')
    img_resized=np.expand_dims(img_resized.astype(float), axis=0)
    
    img_resized = np.transpose(img_resized, (0,3,1,2))
    img_resized=percentile_data(img_resized)
    img_resized=normalize_data(img_resized)
    #img_resized=standardize(img_resized)
    n_bands=img_resized.shape[1]
    print("img_resized.shape",img_resized.shape)

    
    tile_list = np.zeros((n_steps_x*n_steps_y,n_bands,patch,patch),'float')
    
    print("tile list creation")
    
    k=0
    for i in tqdm(range(n_steps_y)):
 
        for j in range(n_steps_x):

                tile_list[k]=img_resized[:,:,i*slide: patch+i*slide,j*slide:patch+j*slide]
               
                k+=1
    print("tile list created")
    
    print("tile_list.shape",tile_list.shape)
             
                
    
    print("tile prediction")
    tile_tot=model.predict(tile_list)
    print("tile prediction")
    tile_tot+=np.rot90(model.predict(np.rot90(tile_list,1,axes=(2, 3))),3,axes=(2, 3))
    print("tile prediction")
    tile_tot+=np.rot90(model.predict(np.rot90(tile_list,2,axes=(2, 3))),2,axes=(2, 3))
    print("tile prediction")
    tile_tot+=np.rot90(model.predict(np.rot90(tile_list,3,axes=(2, 3))),1,axes=(2, 3))
    print("tile prediction")
    tile_tot+=np.flipud(model.predict(np.flipud(tile_list)))
    print("tile prediction")
    tile_tot+=np.fliplr(model.predict(np.fliplr(tile_list)))
        
    tile_tot_pred=tile_tot/6
    
    print("tile_tot_pred.shape",tile_tot_pred.shape)
    
    mask = np.zeros((tile_tot_pred.shape[1],img_resized.shape[2],img_resized.shape[3]),'float')
    
    print("mask.shape",mask.shape)
    
    h=0
    
    for i in tqdm(range(n_steps_y)):
 
        for j in range(n_steps_x):


                tile_mask=tile_tot_pred[h]                     
                
               
                mask[:,i*slide:valid_crop+i*slide,j*slide:valid_crop+j*slide]+=tile_mask
                
                h+=1
                
                
                   
    mask[:,slide:shape_img_row-slide,0:slide]/=2                #leftside
    print(mask.shape)
    mask[:,slide:shape_img_row-slide,shape_img_col-slide::]/=2      #rightside
    print(mask.shape)
    mask[:,0:slide,slide:shape_img_col-slide]/=2                #up
    print(mask.shape)
    mask[:,shape_img_row-slide::,slide:shape_img_col-slide]/=2      #bottom
    print(mask.shape)

    mask[:,slide:shape_img_row-slide,slide:shape_img_col-slide]/=4  #center
    print(mask.shape)
    
    return mask



if __name__ == '__main__':
        
    # for i in range(n_figures):

        # show_prediction(x,y,prediction,np.random.randint(len(x)),classes,save_fig_path,trs=0.1) 
        
        
    #if  os.path.exists(output_path_val + '.npy'):
    
        #prediction_val = np.load(output_path_val+ '.npy')
        #print(prediction_val.shape)
        
    
    #else:
    N_Cls=1
    n_channels=12#8#13
    
    folder_model='./weights/'
    name_model="model_unet_20_32_12_1_percentile_min_max_HW"#'model_unet_150_32_13_1_min_max_River'
    path_model=folder_model+name_model
    model= get_unet_64(n_ch = n_channels, patch_height = height_crop, patch_width = width_crop,n_classes=N_Cls)
    model.load_weights(path_model)
    
   
    
    save_path  ='/mnt/users/defausti/ww/'
    #folder_out='italia_centrale/'
  
    #save_fig_path = './output_pred/plot_prediction/' +folder_out+ name_model
    save_fig_path='/mnt/users/defausti/ww/'+ name_model

    file_land_name='PISA'
    print("----------------------------------------------------")
    img=tiff.imread(save_path+"immagini_sentinel/"+file_land_name+ ".tif")
    # print(img.shape)
    #green=img[:,:,2]
    #nir=img[:,:,8]
    
    #img=np.load(save_path+"immagini_sentinel/"+ file_land_name + '.npy')
    print("-----------------------------------------------")
    print("original_img:",img.shape)
    print("-----------------------------------------------")
    
    img= np.delete(img,[9],2)
    
    print("after_drop_img:",img.shape)
    #sys.exit(0) 
    #img_RGB=RGB_img(img)
    
    #img=np.transpose(img,(1,2,0))
    #print(img_RGB.shape)
       
    print(img.shape)
    
    size_img=img.shape
    patches=64
    slides=27
    valid_crop=54
    paddings=5

    
    # if (file_land_name=='delta_po_no_sea' or file_land_name=='lazio_no_sea') :
        
        # sum_bands=np.sum(img[:size_img[1]*valid_crop,:size_img[0]*valid_crop],axis=2).astype(int)
        # one_band=np.where(sum_bands==0,0,1)
  
        #print("one band: ",one_band.shape)
    
    
    
 
    
    # if  os.path.exists(save_fig_path+"mask_"+file_land_name+ '.npy'):
    
        # mask_img=np.load(save_fig_path+"mask_"+file_land_name +'.npy')
        # print("mask_img:", mask_img.shape)
    
    # # else:
    mask_img=predict_img(model,img,valid_crop,patch=patches,slide=slides,padding=paddings)
    
    print(mask_img.shape)
    np.save(save_fig_path+"mask_"+file_land_name+ '.npy',mask_img)
    '''
    mask_img=np.load(save_fig_path+"mask_"+file_land_name+ '.npy')
    print("mask_img:",mask_img.shape)
    print(np.max(mask_img),np.min(mask_img))
    '''

    # trs_lake=0.2
    # trs_river=np.round(np.arange(0.01,0.9,0.05),3)
    # cmap_my_lake = mpl.colors.ListedColormap(['blue', 'black', 'purple' ])
    
    # # for i in tqdm(trs_river):

    # img_masked_lake=np.where(mask_img[1,:,:]<=trs_lake,1,2)


        # # # # if (file_land_name=='delta_po_no_sea' or  file_land_name=='lazio_no_sea'):
            # # # # img_masked=np.where(one_band*img_masked==0,0,img_masked)
       
    # plt.imshow(img_masked_lake,cmap=cmap_my_lake,vmin=0,vmax=2)
    # # # # #plt.imshow(mask_img[1,:,:])

    # plt.savefig(save_fig_path+ "_lake_" + "{}_mask_ww_{}.png".format(file_land_name,trs_lake),dpi=1200)
    # plt.close()
            
        
    cmap_my_river = mpl.colors.ListedColormap(['blue', 'black', 'yellow' ])
       
        
        # #for i in tqdm(trs):
    trs_river=0.480
    img_masked_river=np.where(mask_img[0,:,:]<=trs_river,1,2)


    # if (file_land_name=='delta_po_no_sea' or  file_land_name=='lazio_no_sea'):
        # img_masked=np.where(one_band*img_masked==0,0,img_masked)
   
    plt.imshow(img_masked_river,cmap=cmap_my_river,vmin=0,vmax=2)
    plt.savefig(save_fig_path+"_hw_"+ "{}_mask_hw_{}.png".format(file_land_name,trs_river),dpi=1200)
        # plt.close()
    
    #plt.imshow(mask_img[0,:,:])
    #plt.savefig(save_fig_path+"_river_"+ "{}_mask_ww_{}.png".format(file_land_name,'heat'),dpi=1200)
    # cmap_my_river = mpl.colors.ListedColormap(['blue', 'black', 'yellow' ])
    # cmap_my_lake = mpl.colors.ListedColormap(['blue', 'none', 'purple' ])

    # #fig,axx=plt.subplots(1,2,figsize=(12,12))

    # #axx[0].imshow(img_RGB)
  
    # plt.imshow(img_masked_river,cmap=cmap_my_river,vmin=0,vmax=2)
    # plt.imshow(img_masked_lake,cmap=cmap_my_lake,vmin=0,vmax=2,alpha=1)
    # plt.savefig(save_fig_path+"_river_lake"+ "{}_mask_ww.png".format(file_land_name),dpi=1200)
    # # plt.close()
     
     
