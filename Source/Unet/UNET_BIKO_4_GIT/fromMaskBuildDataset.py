classeEUSAT="Highway"
dirRGB_EUSAT="/IstatDL/Users/public/Datasets/EuroSAT_RGB_tci/Highway/"
dirMask="/IstatDL/Users/public/Datasets/UNET_DATASETS/ResultsEuropeTiles_MTP/"
dirOutput="/IstatDL/Users/public/Datasets/UNET_DATASETS/Unet_hw/"
dirTIF_EUSAT="/IstatDL/Users/public/Datasets/EuroSATAllBands/Highway/"


classeEUSAT="River"
dirRGB_EUSAT="/IstatDL/Users/public/Datasets/EuroSAT_RGB_tci/River/"
dirMask="/IstatDL/Users/public/Datasets/UNET_DATASETS/output_river_ok/"
dirOutput="/IstatDL/Users/public/Datasets/UNET_DATASETS/Unet_ww/"
dirTIF_EUSAT="/IstatDL/Users/public/Datasets/EuroSATAllBands/River/"

classeEUSAT="SeaLake"
dirRGB_EUSAT="/IstatDL/Users/public/Datasets/EuroSAT_RGB_tci/SeaLake/"
dirMask="/IstatDL/Users/public/Datasets/UNET_DATASETS/OutputLakeOk/"
dirOutput="/IstatDL/Users/public/Datasets/UNET_DATASETS/Unet_sl/"
dirTIF_EUSAT="/IstatDL/Users/public/Datasets/EuroSATAllBands/SeaLake/"




import shutil
import os

for fileMask in os.listdir(dirMask):
	if fileMask.endswith("jpg") and "Mask" in fileMask  :

		codMask=fileMask.split("_")[1]
		print (codMask)
		#Highway_980.jpg
		fileRGB=classeEUSAT+"_"+codMask+".jpg"
		fileTIF=classeEUSAT+"_"+codMask+".tif"
		PathFileRGB=os.sep.join([dirRGB_EUSAT,fileRGB])
		PathFileTIF=os.sep.join([dirTIF_EUSAT,fileTIF])
		PathFileMASK=os.sep.join([dirMask,fileMask])
		PahtOutputRGB=os.sep.join([dirOutput,fileRGB])
		PahtOutputTIF=os.sep.join([dirOutput,fileTIF])
		PahtOutputMASK=os.sep.join([dirOutput,fileMask])
		shutil.copy2(PathFileRGB, PahtOutputRGB)
		shutil.copy2(PathFileTIF, PahtOutputTIF)
		shutil.copy2(PathFileMASK, PahtOutputMASK)
