import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import tifffile as tiff
from tifffile import imsave


base_path='/mnt/Datasets/UNET/'
save_path='/mnt/users/catalano/waterways/'

name_tiff='delta_po'

img= tiff.imread(base_path+'\\{}.tif'.format(name_tiff)


range_loop=int(min(img.shape[:2])/64)


np.save(save_path+"delta_po",img_basilicata_subset)