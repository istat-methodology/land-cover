from model_unet_ww import *
from utils_ww import *



#path dataset:

save_path  ='/mnt/users/catalano/waterways/'
x_val_path = save_path + 'x_val_norm.npy'
x_trn_path = save_path + 'x_train_norm.npy'

y_val_path = save_path + 'y_val.npy'
y_trn_path = save_path + 'y_train.npy'



#model fit parameter:

bc_size  = 32
n_epochs = 60


#Unet's parameters:

n_filters_unet = 64
k_size_unet    = 3
n_channels     = 12
height_crop    = 64
width_crop     = 64






path_model_root   = './weights/'
name_model   = 'model_unet_{}_{}'.format(n_epochs,bc_size)
path_model   = os.path.join(path_model_root,name_model)
csv_log_name = './history_training/his_trn_{}_{}_{}.csv'.format(n_channels,n_epochs,bc_size)
png_log_name = './history_training/plot_his_trn_{}_{}_{}.png'.format(n_channels,n_epochs,bc_size)

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

        model.load_weights(path_model_custom)

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
		
    model = train_net()

    #plot train val 

    history=pd.read_csv(csv_log_name)
    history=history[['epoch','jaccard_coef_int','val_jaccard_coef_int']]
    history.columns=['epoch','jaccard_coef_trn','jaccard_coef_val']
    max_val_jacc=np.round(history.jaccard_coef_val.max(),3)
    history.plot(x='epoch',secondary_y=['jaccard_coef_int', 'val_jaccard_coef_int'],figsize=(15,8))
    plt.legend(loc='lower right',prop={'size': 15})
    plt.title('Unet Training: Waterways'+ '\n' + 'jaccard_val: {}'.format(max_val_jacc),fontsize=20)
    plt.grid()
    plt.savefig(png_log_name)
	