import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
from tifffile import imsave
import os
import random
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from collections import defaultdict
import pdb
import random
import imutils
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from joblib import Parallel, delayed
import multiprocessing
import time
from copy  import *
from sklearn.preprocessing import minmax_scale



smooth = 1e-12



dirNameOut="/mnt/UNET_DATASETS/OutputUNET/hw/"


def percentile_cut(bands, lower_percent,higher_percent):
    
    out=np.zeros((bands.shape[0], bands.shape[1], 3),"int")
    
    for i in range(3):
       
        t=bands[:,:,i]
        c = np.percentile(t, lower_percent)
        d = np.percentile(t, higher_percent)
        t[t < c] = c
        t[t > d] = d
        out[:,:, i] = t
    
    return out

def normalize(bands):

    out = np.zeros_like(bands,"float32")
    n = bands.shape[2]

    for i in range(n):
        a = 0  
        b = 1  
        bands_tmp=bands[:,:,i]
        c = np.amin(bands_tmp)
        d = np.amax(bands_tmp)
        t = a+(bands_tmp - c) * (b - a) / (d - c)
        out[:, :, i] = t

    return out

def normalize_data(dataset):
	
    print("Scale_Data")
    out = np.zeros_like(dataset,"float")
    n = dataset.shape[1]
  
    
    min_bands=[]
    max_bands=[]

    for i in range(n):
        
        tmp=np.copy(dataset)
        a = 0  
        b = 1  
        bands_tmp=tmp[:,i,:,:]
        c= np.amin(bands_tmp)
        d =np.amax(bands_tmp)
        t = a+(bands_tmp - c) * (b - a) / (d - c)
        t[t>b]=b
        t[t<a]=a
        out[:,i, :, :] = t
        print(c,d)
        max_bands.append(d)
        min_bands.append(c)
    np.save(dirNameOut+"max_bands_hw_norm",np.array(max_bands))
    np.save(dirNameOut+"min_bands_hw_norm",np.array(min_bands))
    
    # np.save('/mnt/users/catalano/waterways/'+"max_bands_river_lake_norm",np.array(max_bands))
    # np.save('/mnt/users/catalano/waterways/'+"min_bands_river_lake_norm",np.array(min_bands))
    
    
    #print(np.array(max_bands).shape,np.array(max_bands).shape)

    return out
    
    
def normalize_data_new(dataset):
	
    print("Scale_Data")
    out = np.zeros_like(dataset,"float")
    n = dataset.shape[1]
    
    # min_bands=np.load('/mnt/users/catalano/waterways/'+"max_bands_river_norm.npy")
    # max_bands=np.load('/mnt/users/catalano/waterways/'+"min_bands_river_norm.npy")
    
  
    
    # min_bands=np.load('/mnt/users/catalano/waterways/'+"min_bands_river_lake_norm.npy")
    # max_bands=np.load('/mnt/users/catalano/waterways/'+"max_bands_river_lake_norm.npy")
    
    #print(min_bands.shape,max_bands.shape)
    
    for i in range(n):
        
        tmp=np.copy(dataset)
        a = 0  
        b = 1  
        bands_tmp=tmp[:,i,:,:]
        c=min_bands[i]
        d=max_bands[i]
        
        t = a+(bands_tmp - c) * (b - a) / (d - c)
        t[t>b]=b
        t[t<a]=a
        out[:,i, :, :] = t
        print(c,d)
     

    return out

def percentile_data(dataset):
	
    print("Percentile_Cut")
    out = np.zeros_like(dataset,"float")
    n = dataset.shape[1]
    
    for i in range(n):
        
        tmp=np.copy(dataset)
        bands_tmp=tmp[:,i,:,:]
        c = np.percentile(bands_tmp,2)
        d = np.percentile(bands_tmp,98)
        print(c,d)
        bands_tmp[bands_tmp>d]=d
        bands_tmp[bands_tmp<c]=c
        out[:,i, :, :] = bands_tmp
        

    return out


def standardize(data):

    means = np.mean(data,axis=(0,2,3))
    std= np.std(data,axis =(0,2,3))

      
    # np.save('/mnt/users/catalano/waterways/'+"means_river",means)
    # np.save('/mnt/users/catalano/waterways/'+"std_river",std)
    
    means=np.load(dirNameOut+"means_river.npy")
    std=np.load(dirNameOut+"std_river.npy")

    for k in range(data.shape[1]):
        data[:,k,:,:] -= means[k]
        data[:,k,:,:]  = data[:,k,:,:]/std[k]
        
    return data    

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(y_pred)

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)        

def jaccard_predict(y_true,y_pred,trs):
    
    tmp=np.where(y_pred>trs,1,0)


    intersection = np.sum(y_true.flatten() * tmp.flatten())
    sum= np.sum(y_true.flatten() + tmp.flatten())
    jac = (intersection + smooth) / (sum - intersection + smooth)
    return jac
    
def jaccard_predict_dataset(y_true, y_pred,trs):
    # __author__ = Vladimir Iglovikov
    y_pred_pos =np.where(y_pred>trs,1,0)


    intersection = np.sum(y_true * y_pred_pos, axis=(0, -1, -2))
    sum_ = np.sum(y_true + y_pred_pos, axis=(0, -1, -2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return np.mean(jac) 



def trs_img(img,trs):
    
    tmp=np.copy(img)
    tmp[tmp<trs]=0
    tmp[tmp>=trs]=1

    return tmp
    
    
def RGB_img(image_org):

    

    image = np.zeros((image_org.shape[0], image_org.shape[1], 3),"int")
    
    out=np.zeros((image_org.shape[0], image_org.shape[1], 3),"int")
    perc_img=np.copy(image_org)
    
    k=0
    for i in [4,3,2]:
       
        t=perc_img[:,:,k]
        c = np.percentile(t, 0.2)
        d = np.percentile(t, 0.98)
        t[t < c] = c
        t[t > d] = d
        out[:,:, k] = t
        k+=1
    
    

    image[:, :, 0] = minmax_scale(out[:, :, 0] , feature_range=(0, 255), axis=0, copy=True) # red
    image[:, :, 1] = minmax_scale(out[:, :, 1] , feature_range=(0, 255), axis=0, copy=True)  # green
    image[:, :, 2] = minmax_scale(out[:, :, 2] , feature_range=(0, 255), axis=0, copy=True)  # blue
    
    return image
