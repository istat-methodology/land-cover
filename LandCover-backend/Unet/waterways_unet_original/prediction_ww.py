import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from model_unet_ww import *
from model_training_ww import *
from utils_ww import *


#path dataaset

print('loading dataset...')

x_val,y_val = np.load(x_val_path),np.load(y_val_path)
#x_tr,y_tr   = np.load(x_trn_path),np.load(y_trn_path)

save_path='/mnt/users/catalano/waterways/'


x_val_RGB   = np.load(save_path + 'x_val_RGB.npy')
#x_val_RGB=np.array([ cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in tqdm(x_val_RGB) ])
print('dataset loaded...')

print(x_val.shape)

print(y_val.shape)

print(x_val_RGB.shape)

#params




class_name='waterways'

#save output path


output_folder   ='./output_pred/'

name_pred_file_val = 'prediction_'+'model_unet_{}_{}_{}'.format(n_channels,n_epochs,bc_size)
name_pred_file_tr  = 'training_'+'model_unet_{}_{}_{}'.format(n_channels,n_epochs,bc_size)

output_path_val= os.path.join(output_folder,name_pred_file_val)
output_path_tr= os.path.join(output_folder,name_pred_file_tr)

save_fig_path = './output_pred/plot_prediction/' + name_model


#Code


def show_prediction(x,y_true,y_pred,idx_fig,trs=0.5):


    n_fig=4
    
    
    #print(idx_fig)

    f,axx=plt.subplots(n_fig,3,figsize=(10,10))

    axx[0,0].set_title("RGB Image")
    axx[0,1].set_title('Validation Mask')
    axx[0,2].set_title('Predicted  Mask')
    
    i=0
    for  idx in range(idx_fig[0],idx_fig[1]):
        
        
        
        image=np.transpose(x[idx,:],(1,2,0))
    
        image=percentile_cut(image,2,98)
    
        axx[i,0].imshow(image,figure=f)
    
    
        axx[i,1].imshow(y_true[idx, 0 , :, :])
        
        tmp=y_pred.copy()
        tmp=tmp[idx,0,:,:]
        tmp[tmp<trs]=0
        tmp[tmp>=trs]=1
        
        jac=jaccard_predict(y_true[idx, 0 , :, :],y_pred[idx,0,:,:],trs)
        
        axx[i,2].imshow(tmp,cmap='Greys_r')
        axx[i,2].set_xlabel('jaccard_val: {}'.format(np.round(jac,3)))
    
        f.tight_layout()
        
        i+=1
        
    plt.close()

    


    
    return f







if __name__ == '__main__':
        
    # for i in range(n_figures):

        # show_prediction(x,y,prediction,np.random.randint(len(x)),classes,save_fig_path,trs=0.1) 
        
        
    #if  os.path.exists(output_path_val + '.npy'):
    
        #prediction_val = np.load(output_path_val+ '.npy')
        #print(prediction_val.shape)
        
    
    #else:
    
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
    
    
    trs_list=np.arange(0.1,1,0.025)
    jac=np.zeros_like(trs_list)



    for  i in range(len(trs_list)):

        jac[i]=np.mean([jaccard_predict(y_val[idx, 0 , :, :],prediction_val[idx,0,:,:],trs_list[i]) for idx in range(600)])

    best_trs=trs_list[np.argmax(jac)]
    
    plt.plot(trs_list,jac)
    plt.ylabel('jaccard')
    plt.xlabel('Treshold')
    plt.grid()
    plt.savefig(save_fig_path+"_jacc_plot")
    
    
    
    from matplotlib.backends.backend_pdf import PdfPages

    index_list=np.arange(0,201,4)

    chunk_index_list=[index_list[i:i+2] for i in range(len(index_list))][:-1]

    pdf = PdfPages(save_fig_path + '_plots_pred'+ ".pdf")


    for i in tqdm(chunk_index_list):
              
        
        fig=show_prediction(x_val_RGB[0:200],y_val[0:200],prediction_val[0:200],i,trs=best_trs) 
    
        pdf.savefig(fig)
    
        # destroy the current figure
        # saves memory as opposed to create a new figure
        plt.clf()
        

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