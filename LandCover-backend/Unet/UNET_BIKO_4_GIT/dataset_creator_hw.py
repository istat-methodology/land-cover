from utils_ww import *
import pdb
base_path='/mnt/Datasets/UNET/'
save_path='/mnt/users/defausti/highways/'

def dataset_creator(base_path):

    #open filename and sort 

    #train
    x_train_name = natsorted([f for f in glob(base_path+'TRAIN/Highways/'+'*.tif')])
    y_train_name = natsorted([f for f in glob(base_path+'TRAIN/Highways//'+'*.jpg') if 'Mask' in f])

    #test
    x_val_name = natsorted([f for f in glob(base_path+'TEST/Highways/'+'*.tif')])
    y_val_name = natsorted([f for f in glob(base_path+'TEST/Highways/'+'*.jpg' ) if 'Mask' in f])

    #rgb val
    x_val_RGB_file  = natsorted([f for f in tqdm(glob(base_path+'TEST/Highways/'+'*.jpg')) if not 'Mask' in f])
    x_train_RGB_file  = natsorted([f for f in tqdm(glob(base_path+'TRAIN/Highways/'+'*.jpg')) if not 'Mask' in f])

    #read image and convert from bgr to rgb
    x_val_RGB=np.array([ np.transpose(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB),(2,0,1)) for f in tqdm(x_val_RGB_file) ])
    x_train_RGB=np.array([ np.transpose(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB),(2,0,1)) for f in tqdm(x_train_RGB_file) ])

    #create dict 
    dict_train_name=dict(zip(x_train_name,y_train_name))
    dict_val_name=dict(zip(x_val_name,y_val_name))

    #read image and mask

    x_train=np.array([np.transpose(tiff.imread(k),(2,0,1)).astype(float) for k in tqdm(dict_train_name.keys()) ])
    y_train=np.array([np.round(cv2.imread(k,0)/255).astype(float) for k in tqdm(dict_train_name.values())])
    x_val=np.array([np.transpose(tiff.imread(k),(2,0,1)).astype(float)  for k in tqdm(dict_val_name.keys())])
    y_val=np.array([np.round(cv2.imread(k,0)/255).astype(float)   for k in tqdm(dict_val_name.values())])
    





    #standardize


    #x_train=standardize(x_train)
    #x_val=standardize(x_val) 

    #augmentation

    #rotation random vector: from 1 to 4 times  90 degree rotation
    np.random.seed(1988)
    random_rotate_state=np.random.randint(1,4, size=len(x_train)) 

    # random flipping left/right  up/down
    np.random.seed(1988)
    random_flip_state=np.random.randint(1,3, size=len(x_train))    

    #apply transformation

    #train
    random_rotate_x_train=np.array([np.rot90(x_train[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_train)) ])
    random_rotate_y_train=np.array([np.rot90(y_train[k],random_rotate_state[k]) for k in range(len(y_train)) ])

    random_flip_x_train=np.array([np.flip(x_train[k],axis=random_flip_state[k]) for k in range(len(x_train))])
    random_flip_y_train=np.array([np.flip(y_train[k],axis=random_flip_state[k]-1) for k in range(len(y_train)) ])

    random_rotate_x_train_RGB=np.array([np.rot90(x_train_RGB[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_train_RGB)) ])
    random_flip_x_train_RGB=np.array([np.flip(x_train_RGB[k],random_flip_state[k]) for k in range(len(x_train_RGB)) ])


    #val
    random_rotate_x_val=np.array([np.rot90(x_val[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_val)) ])
    random_rotate_y_val=np.array([np.rot90(y_val[k],random_rotate_state[k]) for k in range(len(y_val)) ])

    #val rgb
    random_rotate_x_val_RGB=np.array([np.rot90(x_val_RGB[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_val_RGB)) ])
    print(random_rotate_x_val_RGB.shape)

    #val flip
    random_flip_x_val=np.array([np.flip(x_val[k],axis=random_flip_state[k]) for k in range(len(x_val))])
    random_flip_y_val=np.array([np.flip(y_val[k],axis=random_flip_state[k]-1) for k in range(len(y_val)) ])

    #val rgb
    random_flip_x_val_RGB=np.array([np.flip(x_val_RGB[k],random_flip_state[k]) for k in range(len(x_val_RGB)) ])
    print(random_flip_x_val_RGB.shape)

    #final dataset augmented

    #train

    x_train_aug=np.concatenate((x_train,random_rotate_x_train,random_flip_x_train))
    y_train_aug=np.expand_dims(np.concatenate((y_train,random_rotate_y_train,random_flip_y_train)),axis=1)

    x_train_RGB_aug=np.concatenate((x_train_RGB,random_rotate_x_train_RGB,random_flip_x_train_RGB))


    #val  

    x_val_aug=np.concatenate((x_val,random_rotate_x_val,random_flip_x_val))
    y_val_aug=np.expand_dims(np.concatenate((y_val,random_rotate_y_val,random_flip_y_val)),axis=1)

    x_val_RGB_aug=np.concatenate((x_val_RGB,random_rotate_x_val_RGB,random_flip_x_val_RGB))

    #crop_mask 60*60
    #pdb.set_trace()
    y_val_aug=y_val_aug[:,:,5:59,5:59]
    y_train_aug=y_train_aug[:,:,5:59,5:59]

    #pop 10 bands

    x_train_aug=np.delete(x_train_aug,[9],1)
    x_val_aug=np.delete(x_val_aug,[9],1)

    print(x_train_aug.shape,y_train_aug.shape)
    print(x_val_aug.shape,y_val_aug.shape)


    dataset_tot_train=np.concatenate((x_train_aug,x_val_aug))
    dataset_tot_val=np.concatenate((y_train_aug,y_val_aug))
    dataset_tot_rgb=np.concatenate((x_train_RGB_aug,x_val_RGB_aug))
    # print("dataset_tot:",dataset_tot.shape)

    # #CCCI and NDWI band 

    # nir      = dataset_tot_train[:,7,:,:]
    # red_edge = dataset_tot_train[:,4,:,:]
    # red      = dataset_tot_train[:,3,:,:]
    # green    = dataset_tot_train[:,2,:,:]




    # CCCI=((nir - red_edge) / (nir + red_edge)) / ((nir - red) / (nir + red))
    #NDWI=((green- nir)/(green+ nir))

    # CCCI= np.expand_dims(CCCI,axis=1)
    #NDWI= np.expand_dims(NDWI,axis=1)

    # print("CCCI:", CCCI.shape)

    #dataset_tot_train=np.concatenate((dataset_tot_train,NDWI),axis=1)

    split_idx=int(dataset_tot_train.shape[0]/5)*4
    
    print(split_idx,dataset_tot_train.shape)

    dataset_tot_train=percentile_data(dataset_tot_train)
    dataset_tot_train=normalize_data(dataset_tot_train)
    #dataset_tot_train=standardize(dataset_tot_train)
   
    
    x_train_final=dataset_tot_train[:split_idx,:,:,:]
    x_val_final=dataset_tot_train[split_idx:,:,:,:]

    y_train_final=dataset_tot_val[:split_idx,:,:,:]
    y_val_final=dataset_tot_val[split_idx:,:,:,:]

    x_train_final_RGB=dataset_tot_rgb[:split_idx,:,:,:]
    x_val_final_RGB=dataset_tot_rgb[split_idx:,:,:,:]



    print(x_train_final.shape,y_train_final.shape)
    print(x_val_final.shape,y_val_final.shape)






    np.save(save_path+'x_train_Highways.npy',x_train_final)
    np.save(save_path+'y_train_Highways.npy',y_train_final)
    np.save(save_path+'x_val_Highways.npy',x_val_final)
    np.save(save_path+'y_val_Highways.npy',y_val_final)

    np.save(save_path+'x_train_RGB_Highways.npy',x_train_final_RGB)

    np.save(save_path+'x_val_RGB_Highways.npy',x_val_final_RGB)



        
            

    



if __name__ == '__main__':
    
    
    dataset_creator(base_path)
    
    


    
