import shutil
import os
import gdal
import numpy as np


L=[ "1_HerbaceousVegetation",  "2_Industrial", "3_Pasture", "4_AnnualCrop", "5_Forest", "6_Residential", "7_PermanentCrop"]
#L=[ "0_HerbaceousVegetation"]



def readTif(tifFile,outFile):
    print ("leggo: " ,tifFile,outFile)    


    #leggo il file_di_input 
    ds = gdal.Open(tifFile)

    R = ds.GetRasterBand(1).ReadAsArray()
#    R = R.astype(np.float)

    G = ds.GetRasterBand(2).ReadAsArray()
    B = ds.GetRasterBand(3).ReadAsArray()
   

    NIR = ds.GetRasterBand(4).ReadAsArray()
#    NIR = NIR.astype(np.float)


    R_orig=(2750-1)/(255-1)*(R-1)+1
    NIR_orig=(4000-1)/(255-1)*(NIR-1)+1

    NDVI=(NIR_orig-R_orig)/(NIR_orig+R_orig)

    
    NDVI=((1+NDVI)*255/2).astype(np.uint8)


    driver = gdal.GetDriverByName("GTiff")


    #outdata = driver.Create(outFile, 64, 64, 4, gdal.GDT_Byte)
    outdata = driver.Create(outFile, 64, 64, 3, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(R)
    outdata.GetRasterBand(2).WriteArray(G)
    outdata.GetRasterBand(3).WriteArray(NIR)
    #outdata.GetRasterBand(4).WriteArray(NDVI)

    #outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!





def f(classe,ds):
        print("CLASSE",classe)
        print("DATASET",ds)
        tif_4_BANDS_dir="/mnt/Datasets/"+ds+"_set4bands_from_RGB_modificato/"+classe+"/"
        tif_4_BANDS_list=os.listdir(tif_4_BANDS_dir)

        dirNDVI="/mnt/Datasets/"+ds+"_setNIR3_modificato/"+classe+"/"
        for filetif in tif_4_BANDS_list:
            try:
                pass
                os.mkdir(dirNDVI)
            except:
                print("exists",dirNDVI)

            readTif(tif_4_BANDS_dir+"/"+filetif,dirNDVI+"/"+filetif)

        '''
        src_RBGdir="/mnt/Datasets/EuroSAT4Bands/"+classe.split("_")[1]+"/"
        dst_RBGdir="/mnt/Datasets/"+ds+"_set4bands_from_RGB_modificato/"+classe+"/"
        try:
            os.mkdir(dst_RBGdir)
        except:
            print("exists",dst_RBGdir)
	#dst_RBGdir="/mnt/Datasets/"+ds+"_setRGB_4Bands/"+classe+"/"
        for fname in name4:
            if fname.endswith(".jpg")  :
                src=src_RBGdir+fname.replace(".jpg",".tif")
                dst=dst_RBGdir+fname.replace(".jpg",".tif")
                #print(src,dst)
                try:
                    shutil.copy(src,dst)
                except:
                    print("file non trovato:",src)
                    src=(src_RBGdir+fname.replace(".jpg",".tif")).replace("Forest","HerbaceousVegetation")
                    #dst=dst_RBGdir+fname.replace(".jpg",".tif")
                    print("file replace name trovato:",src)
                    print("destinazione:",dst)
                    shutil.copy(src,dst)

                #print(fname.split("_")[1].split(".")[0])
#
#        src_RBGdir="/mnt/Datasets/EuroSAToriginal/"+classe.split("_")[1]+"/"
#        name4=os.listdir("/mnt/Datasets/Training_set4Bands/"+classe+"/")
#        dst_RBGdir="/mnt/Datasets/Training_setRGB_4Bands/"+classe+"/"
#        for fname in name4:
#            src=src_RBGdir+fname.replace(".tif",".jpg")
#            dst=dst_RBGdir+fname.replace(".tif",".jpg")
#            print(src,dst)
##            shutil.copy(src,dst)
#            print(fname.split("_")[1].split(".")[0])
#
#
#
#
         '''
for ds in ["Validation","Training"]:
    for l in L:
        f(l,ds)

