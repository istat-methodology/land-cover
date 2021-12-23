from model_unetUNET import *
from utilsUNET import *



#path dataset:
save_path  ='/mnt/UNET_DATASETS/OutputUNET/ww/'
x_val_path = save_path + 'x_val.npy'
x_trn_path = save_path + 'x_train.npy'

y_val_path = save_path + 'y_val.npy'
y_trn_path = save_path + 'y_train.npy' 

#x_val_path = save_path + 'x_val_river.npy'
#x_trn_path = save_path + 'x_train_river.npy'

#y_val_path = save_path + 'y_val_river.npy'
#y_trn_path = save_path + 'y_train_river.npy' 



#model fit parameter:

bc_size  = 32
n_epochs = 20 
N_Cls    =1

#Unet's parameters:

n_filters_unet = 64
k_size_unet    = 3
n_channels     = 4#12
height_crop    = 64
width_crop     = 64
type_preprocessing="percentile_min_max_HW"#River"






path_model_root   = './weights/'
name_model   = 'model_unet_{}_{}_{}_{}_{}'.format(n_epochs,bc_size,n_channels,N_Cls,type_preprocessing)
path_model   = os.path.join(path_model_root,name_model)
csv_log_name = './history_training/his_trn_{}_{}_{}_{}_{}.csv'.format(N_Cls,n_channels,n_epochs,bc_size,type_preprocessing)
png_log_name = './history_training/plot_his_trn_{}_{}_{}_{}_{}.png'.format(N_Cls,n_channels,n_epochs,bc_size,type_preprocessing)

#path_model_custom= os.path.join(path_model_root,'model_unet_90_24_1_80_8_Track')


def train_net(model_weights=None):
	
    print( "start train net")
    x_val, y_val = np.load(x_val_path),np.load(y_val_path)
    print('x_val and y_val loaded')
    x_trn, y_trn = np.load(x_trn_path),np.load(y_trn_path)
    print('x_trn and y_trn loaded')

    

    model = get_unet_64(n_ch = n_channels, patch_height = height_crop, patch_width = width_crop,n_classes=N_Cls)
    print('got unet')

    if model_weights:
        
        print("weights loaded")
        model.load_weights(path_model)

    check_point = ModelCheckpoint(  path_model,
                                    save_weights_only=True,
                                    monitor='val_jaccard_coef_int',
                                    verbose=1,
                                    save_best_only=True, 
                                    mode='max')

    csv_logger = CSVLogger(csv_log_name)										


    model.fit(x_trn, y_trn, batch_size = bc_size, epochs = n_epochs, verbose=1, shuffle=True,
                  validation_data=(x_val, y_val),callbacks=[check_point,csv_logger])
        


    return model
	
if __name__ == '__main__':
		
        
   
    
		
    csv_log_name = './history_training/his_trn_{}_{}_{}_{}_{}.csv'.format(N_Cls,type_preprocessing,n_channels,n_epochs,bc_size)
    png_log_name = './history_training/plot_his_trn_{}_{}_{}_{}_{}.png'.format(N_Cls,type_preprocessing,n_channels,n_epochs,bc_size)
    
    print(type_preprocessing)
    
    
    # x_val_path = save_path + 'x_val_tot.npy'
    # x_trn_path = save_path + 'x_train_tot.npy'

    # y_val_path = save_path + 'y_val_tot.npy'
    # y_trn_path = save_path + 'y_train_tot.npy' 

    # x_val_path = save_path + 'x_val_river.npy'
    # x_trn_path = save_path + 'x_train_river.npy'

    # y_val_path = save_path + 'y_val_river.npy'
    # y_trn_path = save_path + 'y_train_river.npy' 

 
    x_val, y_val = np.load(x_val_path),np.load(y_val_path)
    print('x_val and y_val loaded')
    x_trn, y_trn = np.load(x_trn_path),np.load(y_trn_path)
    print('x_trn and y_trn loaded')

    
    model = train_net()

    #plot train val 

    history=pd.read_csv(csv_log_name)
    history=history[['epoch','jaccard_coef_int','val_jaccard_coef_int']]
    history.columns=['epoch','jaccard_coef_trn','jaccard_coef_val']
    index_best_jacc=history.jaccard_coef_val.idxmax()
    max_val_jacc=np.round(history.jaccard_coef_val.max(),3)
    history.plot(x='epoch',secondary_y=['jaccard_coef_int', 'val_jaccard_coef_int'],figsize=(15,8))
    plt.plot(index_best_jacc,max_val_jacc,"ro")
    plt.legend(loc='lower right',prop={'size': 15})
    plt.title('Unet Training: {} \n\n jaccard_val: {}\n'.format(type_preprocessing,max_val_jacc),fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(png_log_name)
	
