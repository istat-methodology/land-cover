from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import *
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger
from keras import backend as K

from utils_ww import *





#Unet's Parameters

bc_size  = 32
n_epochs = 80
N_Cls    =2

n_filters_unet = 64
k_size_unet    = 3
n_channels     = 12
height_crop    = 64
width_crop     = 64


path_model_root   = './weights/'
name_model   = 'model_unet_prova_{}_{}'.format(n_epochs,bc_size)
path_model   = os.path.join(path_model_root,name_model)





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



    # Step 3 - Flattening
    flatten=Flatten()(block_5)

    # Step 4 - Full connection
    Dense_1= Dense(units = 64, activation = 'relu')(flatten)
    Final= Dense(units = 1, activation = 'sigmoid')(Dense_1)

    
    model = Model(input= input_tensor, output=Final )
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


    return model

if __name__ == '__main__':


    bc_size  = 32
    n_epochs = 80
    N_Cls    =2
		
    save_path  ='/mnt/users/catalano/waterways/'
    csv_log_name = './history_training/his_trn_{}_{}_{}_{}.csv'.format(N_Cls,n_channels,n_epochs,bc_size)
    png_log_name = './history_training/plot_his_trn_{}_{}_{}_{}.png'.format(N_Cls,n_channels,n_epochs,bc_size)
    
    
    
    x_val_path = save_path + 'x_val_tot.npy'
    x_trn_path = save_path + 'x_train_tot.npy'

    y_val_path = save_path + 'y_val_tot.npy'
    y_trn_path = save_path + 'y_train_tot.npy' 



 
    x_val, y_val = np.load(x_val_path),np.load(y_val_path)
    print('x_val and y_val loaded')
    x_trn, y_trn = np.load(x_trn_path),np.load(y_trn_path)
    print('x_trn and y_trn loaded')

    model=get_unet_64()
    
    check_point = ModelCheckpoint(  path_model,
                                    save_weights_only=True,
                                    monitor='val_jaccard_coef_int',
                                    verbose=1,
                                    save_best_only=True, 
                                    mode='max')
        
    csv_logger = CSVLogger(csv_log_name) 

    model.fit(x_trn, y_trn, batch_size = bc_size, epochs = n_epochs, verbose=1, shuffle=True,
                  validation_data=(x_val, y_val),callbacks=[check_point,csv_logger])
  
    history=pd.read_csv(csv_log_name)
    history=history[['epoch','jaccard_coef_int','val_jaccard_coef_int']]
    history.columns=['epoch','jaccard_coef_trn','jaccard_coef_val']
    index_best_jacc=history.jaccard_coef_val.idxmax("val_jaccard_coef_int")
    max_val_jacc=np.round(history.jaccard_coef_val.max(),3)
    history.plot(x='epoch',secondary_y=['jaccard_coef_int', 'val_jaccard_coef_int'],figsize=(15,8))
    plt.plot(index_best_jacc,max_val_jacc,"ro")
    plt.legend(loc='lower right',prop={'size': 15})
    plt.title('Unet Training: '+ "Road_Track" + '\n\n' + 'jaccard_val: {}\n'.format(max_val_jacc),fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(png_log_name)