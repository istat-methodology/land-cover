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



smooth = 1e-12

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


def standardize(data):

	means = np.mean(data,axis=(0,2,3))
	std= np.std(data,axis =(0,2,3))

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
    
    tmp=y_pred.copy()
    tmp[tmp<trs]=0
    tmp[tmp>=trs]=1


    intersection = np.sum(y_true.flatten() * tmp.flatten())
    sum= np.sum(y_true.flatten() + tmp.flatten())
    jac = (intersection + smooth) / (sum - intersection + smooth)
    return jac



def trs_img(img,trs):
    
    tmp=np.copy(img)
    tmp[tmp<trs]=0
    tmp[tmp>=trs]=1

    return tmp