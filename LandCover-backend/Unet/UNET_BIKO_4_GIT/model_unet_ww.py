from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger
from keras import backend as K

from utils_ww import *


#Unet's Parameters

n_filters_unet = 64
k_size_unet    = 3
n_channels     = 13
height_crop    = 64
width_crop     = 64
N_Cls		   = 1



def BN_CONV_ACT(input_tensor,batchnorm, n_filters=n_filters_unet, k_size=k_size_unet):
    
    if batchnorm:
        
        x = BatchNormalization()(input_tensor)
        x = Conv2D(n_filters, kernel_size=(k_size, k_size),strides=(1,1), padding="same",data_format="channels_first")(x)
    
    else:
        
        x = Conv2D(n_filters, kernel_size=(k_size, k_size),strides=(1,1), padding="same",data_format="channels_first")(input_tensor)
    
    x = Activation("relu")(x)
   
    
    return x


def conv2d_block(input_tensor,batchnorm=[True,True,True]):

        
    x =       BN_CONV_ACT(input_tensor,      batchnorm[0])
    x_crop =  BN_CONV_ACT(x,                 batchnorm[1])
    x =       BN_CONV_ACT(x_crop,            batchnorm[2])
    x =       MaxPooling2D(pool_size=(2,2), data_format="channels_first")(x)
    
    
    return x,x_crop
	
	
def BN_UPCONV_ACT(input_tensor, n_filters=n_filters_unet, k_size=k_size_unet,batchnorm=True):
    
    x = BatchNormalization()(input_tensor)
    
    x = Conv2DTranspose(n_filters, kernel_size=(k_size, k_size),strides=(2,2), padding="same", data_format="channels_first")(x)
    
    x = Activation("relu")(x)
   
    
    return x
	
	
def deconv2d_block(input_tensor,conc,cropped_layer=None):
    
    concat_axis = 1
    
    if conc:
        
        x =  concatenate([input_tensor, cropped_layer], axis=concat_axis)
        x =  BN_CONV_ACT(x,batchnorm=True, n_filters=n_filters_unet, k_size=k_size_unet )
    
    else:
    
        x =  BN_CONV_ACT(input_tensor, batchnorm=True, n_filters=n_filters_unet, k_size=k_size_unet )
    
    
    x = BN_CONV_ACT(x, batchnorm=True,n_filters=n_filters_unet, k_size=k_size_unet)

    x = BN_UPCONV_ACT(x, n_filters=n_filters_unet, k_size=k_size_unet,batchnorm=True)
    
    
    return x


def get_unet_64(n_ch = n_channels, patch_height = height_crop, patch_width = width_crop,n_classes=N_Cls):
    
   

	input_tensor = Input((n_ch, patch_width, patch_height))


	#contracting part

	block_1,crop_1 = conv2d_block(input_tensor,batchnorm=[False,True,True])

	block_2,crop_2 = conv2d_block( block_1,    batchnorm=[True,True,True])
	block_3,crop_3 = conv2d_block( block_2,    batchnorm=[True,True,True])
	block_4,crop_4 = conv2d_block( block_3,    batchnorm=[True,True,True])
	block_5,crop_5 = conv2d_block( block_4,    batchnorm=[True,True,True])


	#expansive part

	block_6  = deconv2d_block( block_5,  conc=False)

	block_7  = deconv2d_block( block_6,  conc=True, cropped_layer= crop_5)
	block_8  = deconv2d_block( block_7,  conc=True, cropped_layer= crop_4)
	block_9  = deconv2d_block( block_8,  conc=True, cropped_layer= crop_3)
	block_10 = deconv2d_block( block_9,  conc=True, cropped_layer= crop_2)



	final_block_11  = concatenate([ block_10,crop_1], axis=1)
	final_block_11 = BN_CONV_ACT( final_block_11,batchnorm=True, n_filters=n_filters_unet, k_size=k_size_unet )
	final_block_11 = BN_CONV_ACT( final_block_11,batchnorm=True, n_filters=n_filters_unet, k_size=k_size_unet )


	conv12 = Conv2D(n_classes, (1, 1), padding="same", data_format="channels_first", activation="sigmoid")( final_block_11)
	crop_conv_final = Cropping2D(cropping=((5,5), (5,5)), data_format="channels_first")(conv12)

	model = Model(input= input_tensor, output=crop_conv_final )
	model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    
	return model

if __name__ == '__main__':
		
	model=get_unet_64()
	print(model.summary())