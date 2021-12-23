import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import matplotlib as mpl

from model_unetUNET import *
from model_trainingUNET import *
from utilsUNET import *

plt.rcParams.update({'font.size': 20})
#path dataaset



#params




#class_name='ww'

#save output path


output_folder   ='/mnt/UNET_RUN/output_pred/'
save_path='/mnt/UNET_DATASETS/OutputUNET/ww/'


name_pred_file_val = 'prediction_'+'model_unet_{}_{}_{}'.format(n_channels,n_epochs,bc_size)
name_pred_file_tr  = 'training_'+'model_unet_{}_{}_{}'.format(n_channels,n_epochs,bc_size)

output_path_val= os.path.join(output_folder,name_pred_file_val)
output_path_tr= os.path.join(output_folder,name_pred_file_tr)

save_fig_path = '/mnt/UNET_RUN/output_pred/plot_prediction/' + name_model


#Code


def show_prediction(x,y_true,y_pred,idx_fig,label_list,trs=0.5):


    n_fig=4
    cmap_river = mpl.colors.ListedColormap(['black', 'yellow'])
    cmap_lake = mpl.colors.ListedColormap(['black', 'cyan'])
    
    cmap_river_pred = mpl.colors.ListedColormap(['black', 'green'])
    cmap_lake_pred = mpl.colors.ListedColormap(['black', 'purple'])
    
    #print(idx_fig)

    f,axx=plt.subplots(n_fig,3,figsize=(24,24))

    axx[0,0].set_title("RGB Image\n\n")
    axx[0,1].set_title('Val Mask \n\n')
    #axx[0,2].set_title('Val Mask Lake\n\n')
    axx[0,2].set_title('Val Pred \n\n')
    #axx[0,4].set_title('Val Pred Lake\n\n')
  
    
    i=0
    for  idx in range(idx_fig[0],idx_fig[1]):
        
        
        
        image=np.transpose(x[idx,:],(1,2,0))
    
        image=percentile_cut(image,2,98)
    
        axx[i,0].imshow(image,figure=f)
        axx[i,0].set_xlabel("label_list[idx]")
        
    
    
        axx[i,1].imshow(y_true[idx, 0 , :, :].astype(int),cmap=cmap_river,vmin=0,vmax=1)
        #axx[i,2].imshow(y_true[idx, 1 , :, :].astype(int),cmap=cmap_lake,vmin=0,vmax=1)
        
        
        tmp_1=np.where(y_pred[idx,0,:,:]>trs,1,0)
        #tmp_2=np.where(y_pred[idx,1,:,:]>trs,1,0)
     
        jac_1=jaccard_predict(y_true[idx, 0 , :, :],y_pred[idx,0,:,:],trs)
        #jac_2=jaccard_predict(y_true[idx, 1 , :, :],y_pred[idx,1,:,:],trs)
        
        axx[i,2].imshow(tmp_1,cmap=cmap_river_pred,vmin=0,vmax=1)
        axx[i,2].set_xlabel('jaccard_val: {}'.format(np.round(jac_1,3)))
        
         
        #axx[i,4].imshow(tmp_2,cmap=cmap_lake_pred,vmin=0,vmax=1)
        #axx[i,4].set_xlabel('jaccard_val: {}'.format(np.round(jac_2,3)))
    
        f.tight_layout()
        
        i+=1
        
    plt.close()

    


    
    return f







if __name__ == '__main__':
        
    print('loading dataset...')


    x_val,y_val = np.load(x_val_path),np.load(y_val_path)
    #x_val_label= np.load(save_path+'x_val_label.npy')




    x_val_RGB   = np.load(save_path + 'x_val_RGB.npy')

    print('dataset loaded...')

    print(x_val.shape)

    print(y_val.shape)

    print(x_val_RGB.shape)
    
    
    
    model= get_unet_64(n_ch = n_channels, patch_height = height_crop, patch_width = width_crop,n_classes=N_Cls)
    model.load_weights(path_model)

    prediction_val = model.predict(x_val)

    np.save(output_path_val,prediction_val)
        
    
#    if  os.path.exists(output_path_tr + '.npy'):
#        
#        training = np.load(output_path_tr + '.npy')
#    
#    else:
#    
#        model= get_unet_64(n_ch = n_channels, patch_height = height_crop, patch_width = width_crop,n_classes=N_Cls)
#        model.load_weights(path_model)
#        
#        prediction_tr = model.predict(x_tr)
#        
#        np.save(output_path_tr,prediction_tr)
    
    prediction_val=np.load(output_path_val+".npy")
    
    print(prediction_val.shape)
    
    trs_list=np.round(np.arange(0.1,1,0.025),3)
    jac=np.zeros_like(trs_list)



    for  i in range(len(trs_list)):

        jac[i]=jaccard_predict_dataset(y_val,prediction_val,trs_list[i])

    best_trs=np.round(trs_list[np.argmax(jac)],3)
    
    plt.figure(figsize=(12,10))
    plt.plot(trs_list,jac)
    plt.plot(best_trs,jac[np.argmax(jac)],"ro")
    plt.title("validation jaccard treshold\n") 
    plt.ylabel('jaccard')
    plt.xlabel('treshold')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_fig_path+"_jacc_plot")
    
    
    
    from matplotlib.backends.backend_pdf import PdfPages

    index_list=np.arange(0,401,4)

    chunk_index_list=[index_list[i:i+2] for i in range(len(index_list))][:-1]

    pdf = PdfPages(save_fig_path + '_plots_pred'+ ".pdf")


    for i in tqdm(chunk_index_list[:]):
              
        
        fig=show_prediction(x_val_RGB,y_val,prediction_val,i,"x_val_label",trs=best_trs) 
    
        pdf.savefig(fig)
    
        # destroy the current figure
        # saves memory as opposed to create a new figure
        plt.close()
        

    pdf.close()
    
    
    
#    
#    pdf = PdfPages(save_fig_path + '_plots_trn'+ ".pdf")
#
#
#    for i in range(n_figures):
#          
#        
#        fig=show_prediction(x_tr,y_tr,prediction_tr,np.random.randint(len(x_val)), class_name,save_fig_path,trs=0.5) 
#
#        pdf.savefig(fig)
#
#        # destroy the current figure
#        # saves memory as opposed to create a new figure
#        plt.clf()
#        
#
#    pdf.close()
#    
