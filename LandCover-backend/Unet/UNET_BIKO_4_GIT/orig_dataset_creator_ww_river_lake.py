from utils_ww import *
from sklearn.model_selection import train_test_split

base_path='/mnt/Datasets/UNET/'
save_path='/mnt/users/defausti/waterways/'

seed=1988

def dataset_creator(base_path,class_image):

    print("{} dataset".format(class_image))

    np.random.seed(seed)
    
    #open filename and sort 

    #train
    x_train_name = natsorted([f for f in glob(base_path+'TRAIN/{}/*.tif'.format(class_image))])
    y_train_name = natsorted([f for f in glob(base_path+'TRAIN/{}/*.jpg'.format(class_image)) if 'Mask' in f])

    #test
    x_val_name = natsorted([f for f in glob(base_path+'TEST/{}/*.tif'.format(class_image))])
    y_val_name = natsorted([f for f in glob(base_path+'TEST/{}/*.jpg'.format(class_image)) if 'Mask' in f])

    #rgb val
    x_val_RGB_file  = natsorted([f for f in tqdm(glob(base_path+'TEST/{}/*.jpg'.format(class_image))) if not 'Mask' in f])
    x_train_RGB_file  = natsorted([f for f in tqdm(glob(base_path+'TRAIN/{}/*.jpg'.format(class_image))) if not 'Mask' in f])

    #read image and convert from bgr to rgb
    x_val_RGB   = np.array([ np.transpose(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB),(2,0,1)) for f in tqdm(x_val_RGB_file) ])
    x_train_RGB = np.array([ np.transpose(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB),(2,0,1)) for f in tqdm(x_train_RGB_file) ])

    #create dict 
    dict_train_name=dict(zip(x_train_name,y_train_name))
    dict_val_name=dict(zip(x_val_name,y_val_name))

    #read image and mask
    
    x_train=np.array([np.transpose(tiff.imread(k),(2,0,1)).astype(float) for k in tqdm(dict_train_name.keys()) ])
    y_train=np.array([np.round(cv2.imread(k,0)/255).astype(float) for k in tqdm(dict_train_name.values())])
    
    x_val=np.array([np.transpose(tiff.imread(k),(2,0,1)).astype(float)  for k in tqdm(dict_val_name.keys())])
    y_val=np.array([np.round(cv2.imread(k,0)/255).astype(float)   for k in tqdm(dict_val_name.values())])

    

    #augmentation
    
    #rotation random vector: from 1 to 4 times  90 degree rotation
   
    random_rotate_state=np.random.randint(1,4, size=len(x_train)) 
    
    # random flipping left/right  up/down
  
    random_flip_state=np.random.randint(1,3, size=len(x_train))    
    
    #apply transformation
    
    #augmentation
    
    #rotation random vector: from 1 to 4 times  90 degree rotation
    
    random_rotate_state=np.random.randint(1,4, size=len(x_train)) 
    
    #random flipping left/right  up/down
   
    random_flip_state=np.random.randint(1,3, size=len(x_train))    
    
    #apply transformation
    
    #train rotation/flip
    random_rotate_x_train=np.array([np.rot90(x_train[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_train)) ])
    random_rotate_y_train=np.array([np.rot90(y_train[k],random_rotate_state[k]) for k in range(len(y_train)) ])

    random_flip_x_train=np.array([np.flip(x_train[k],axis=random_flip_state[k]) for k in range(len(x_train))])
    random_flip_y_train=np.array([np.flip(y_train[k],axis=random_flip_state[k]-1) for k in range(len(y_train)) ])
    
    
    #val rotation/flip
    random_rotate_x_val=np.array([np.rot90(x_val[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_val)) ])
    random_rotate_y_val=np.array([np.rot90(y_val[k],random_rotate_state[k]) for k in range(len(y_val))])
    
    random_flip_x_val=np.array([np.flip(x_val[k],axis=random_flip_state[k]) for k in range(len(x_val))])
    random_flip_y_val=np.array([np.flip(y_val[k],axis=random_flip_state[k]-1) for k in range(len(y_val)) ])
    
    #train rgb rotation/flip
    random_rotate_x_train_RGB=np.array([np.rot90(x_train_RGB[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_train_RGB)) ])
    print(random_rotate_x_train_RGB.shape)

    random_flip_x_train_RGB=np.array([np.flip(x_train_RGB[k],random_flip_state[k]) for k in range(len(x_train_RGB)) ])
    print(random_flip_x_train_RGB.shape)
    
    #val rgb rotation/flip
    random_rotate_x_val_RGB=np.array([np.rot90(x_val_RGB[k],random_rotate_state[k],axes=(1, 2)) for k in range(len(x_val_RGB)) ])
    print(random_rotate_x_val_RGB.shape)
    
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
    y_val_aug=y_val_aug[:,:,5:59,5:59]
    y_train_aug=y_train_aug[:,:,5:59,5:59]
    
    #pop 10 bands
    
    # x_train_aug=np.delete(x_train_aug,9,1)
    # x_val_aug=np.delete(x_val_aug,9,1)
    
    
  

    x_tot=np.concatenate((x_train_aug,x_val_aug))
    y_tot=np.concatenate((y_train_aug,y_val_aug))
    
    RGB_tot=np.concatenate((x_train_RGB_aug,x_val_RGB_aug))
    
    print("Shape  dataset:",x_tot.shape,y_tot.shape, RGB_tot.shape)
    
    return x_tot,y_tot,RGB_tot
    
   
def join_dataset(data_1,data_2,mask_1,mask_2,RGB_1,RGB_2):
    
    print("join dataset")
    
    min_len=min(len(data_1),len(data_2))
    
    mask_zero_1=np.zeros_like(mask_1)
    mask_zero_2=np.zeros_like(mask_2)
    
    mask_1=np.concatenate((mask_1,mask_zero_1),axis=1)
    mask_2=np.concatenate((mask_zero_2,mask_2),axis=1)
    
    river_label=np.array(["river"]*min_len)
    lake_label=np.array(["lake"]*min_len)
   
    
    x_tot=np.concatenate((data_1[:min_len],data_2[:min_len]))
    
    np.random.seed(seed)
    np.random.shuffle(x_tot)
  
    
    y_tot=np.concatenate((mask_1[:min_len],mask_2[:min_len]))
    
    np.random.seed(seed)
    np.random.shuffle(y_tot)
    
   
    RGB_tot=np.concatenate((RGB_1[:min_len],RGB_2[:min_len]))
    
    np.random.seed(seed)
    np.random.shuffle(RGB_tot)
    
    
    label_tot=np.concatenate((river_label,lake_label))
    
    np.random.seed(seed)
    np.random.shuffle(label_tot)
     
    x_tot=np.delete(x_tot,[8,9,10,11,12],1) 
    x_tot=percentile_data(x_tot) 
    #x_tot=normalize_data(x_tot)
    x_tot=standardize(x_tot)
    
    size_split=int(x_tot.shape[0]/5)*4
    
    print(size_split)
    
    x_train = x_tot[:size_split]
    x_val   = x_tot[size_split:]
    
    y_train = y_tot[:size_split]
    y_val   = y_tot[size_split:]
    
    RGB_train = RGB_tot[:size_split]
    RGB_val   = RGB_tot[size_split:]
    
    label_train=label_tot[:size_split]
    label_val= label_tot[size_split:]
    
    print("Shape train  dataset:",x_train.shape,y_train.shape, RGB_train.shape,label_train.shape)
    
    print("Shape val  dataset:",x_val.shape,y_val.shape, RGB_val.shape,label_val.shape)
   
    np.save(save_path+'x_train_tot.npy',x_train)
    np.save(save_path+'y_train_tot.npy',y_train)
    
    np.save(save_path+'x_val_tot.npy',x_val)
    np.save(save_path+'y_val_tot.npy',y_val)
    
    np.save(save_path+'x_train_RGB.npy',RGB_train)
    np.save(save_path+'x_val_RGB.npy',  RGB_val)
    
    np.save(save_path+'x_train_RGB.npy',RGB_train)
    np.save(save_path+'x_val_RGB.npy',  RGB_val)
    
    np.save(save_path+'x_train_label.npy',label_train)
    np.save(save_path+'x_val_label.npy',  label_val)


    



if __name__ == '__main__':
    
    
    x_tot_River,y_tot_River,RGB_tot_River=dataset_creator(base_path,"River")
    x_tot_Lake,y_tot_Lake,RGB_tot_Lake=dataset_creator(base_path,"Lake")
    
    join_dataset(x_tot_River, x_tot_Lake,y_tot_River,y_tot_Lake,RGB_tot_River,RGB_tot_Lake)

    
    
    
    


    
