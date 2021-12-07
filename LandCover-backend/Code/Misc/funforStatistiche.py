import sys
import csv
import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
if True:
    def statistiche(x, class_leg, mapFile, nomefile, mapping):

        unique, counts = np.unique(x, return_counts=True)
        #print(np.asarray((unique, counts)).T)

        #check coherence between output_classes file and the classification matrix
        listA = class_leg.Class.unique()
        listB = unique
        if set(listA)==set(listB):
            print('woooow')
        else:
            print("Warning: output_classes file not coherent with the classification matrix")

        #df1 = pd.DataFrame(statemp)
        df1 = pd.DataFrame([ [c,v] for c,v in zip (unique, counts)])   # AGGIUNTO DA FABRIZIO

        df1.columns =['Class', 'Values']

        val = class_leg[class_leg['Legend']=='other']['Class'].values[0]
        df1ok = df1[df1.Class != val]

        a = df1ok.Values

        out = np.true_divide(a,np.sum(a))
        outperc = np.multiply(out,100)
        df2 = pd.DataFrame(out)

        stat = df1ok
        #print('//2')
        stat.loc[:,'Frac'] = df2.values
        #print('//3')

        statfin = stat
        statfin = stat.join(class_leg, lsuffix='', rsuffix='_file')
        statfin = statfin.drop('Class_file',1)
        #print(statfin)
        p = str(datetime.date.today())
        print(p)

        #EUROSAT statistics

        statfin.to_csv(''.join([nomefile, "_EuroSAT", "_", p, ".csv"]), sep=';', index=False)
        print('EuroSAT statistics done')

        if mapping!=False:

            #LUCAS statistics
            mapFile['MAPPING_CLASS'] = mapFile['MAPPING_CLASS'].astype('Int64')
            mapFile['LEGEND'] = mapFile['LEGEND'].str.lower()
            mapFile['EUROSAT_LEGEND'] = mapFile['EUROSAT_LEGEND'].str.lower()

            #print(mapFile)
            #print('//')
            mapFileok= mapFile.set_index('EUROSAT_LEGEND').join(class_leg.set_index('Legend'), on='EUROSAT_LEGEND', how='left')
            #print(mapFileok)

            mapFileok['EUROSAT_CLASS'] = mapFileok['Class'].astype('Int64')
            mapFileok = mapFileok.drop('Class',1)

            new = stat.set_index('Class').join(mapFileok.set_index('EUROSAT_CLASS'), on='Class', how='left')
            #print(new)
            restmp = new.groupby(['MAPPING_CLASS']).sum()
            res = restmp.reset_index()
            #print(res)

            final = res.set_index('MAPPING_CLASS').join(mapFile[['MAPPING_CLASS','LEGEND']].set_index('MAPPING_CLASS'), how='left')
            final = final.drop_duplicates()
            #print(final)
            p = str(datetime.date.today())
            final.to_csv(''.join([nomefile, "_", mapping, "_", p, ".csv"]), sep=';')
            print( mapping + ' statistics done')

        return None


    def print_Mappa(x, class_leg, listofcolor, nomefile_mappa, mappa_eurosat, mapping, mapFile):

        num_class =len(class_leg)
        class_dict = class_leg.Legend.to_dict()
        result_Legend = str(class_dict)

        if len(listofcolor)<num_class:
            print("listofcolor")
            print(listofcolor)

            print("class_leg")
            print(class_leg)

            sys.exit("Error: list of color for colormap should be at least as long as the number of classes")

        if mappa_eurosat == True:

            cmap = mpl.colors.ListedColormap(listofcolor)
            cmap.set_over('0')
            cmap.set_under(str(num_class))

            bounds_list=list(np.arange(0.1, num_class+0.1, 1))
            bounds = [-1]+bounds_list

            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            fig=plt.figure(figsize = (18,18))

            plt.imshow(x,cmap=cmap,norm=norm, interpolation='none')
            plt.colorbar()
            plt.title(result_Legend)

            p = str(datetime.date.today())
            fig.savefig(''.join([nomefile_mappa, "_EuroSAT_", p, ".png"]))
        if mapping != False :
            print(mapFile)
            #if mapFile==False:
            if mapFile is None:
                sys.exit("Error: Mapping File required in input")
            else:
                selmyFile = mapFile[['MAPPING_CLASS','LEGEND']]
                selmyFile = selmyFile.drop_duplicates()

                mapFileok= mapFile.set_index('EUROSAT_LEGEND').join(class_leg.set_index('Legend'), on='EUROSAT_LEGEND', how='left')
                mapFileok['EUROSAT_CLASS'] = mapFileok['Class'].astype('Int64')
                mapFileok = mapFileok.drop('Class',1)

                ClassmyFile = mapFileok[['MAPPING_CLASS','EUROSAT_CLASS']]
                Classmy = ClassmyFile.dropna().astype(int)
                print("Classmy")
                print(Classmy)
                classmyind = Classmy.sort_values(by=['EUROSAT_CLASS'])

                print("listofcolor")
                print(listofcolor)
                print("classmyind.MAPPING_CLASS.tolist")
                print(classmyind.MAPPING_CLASS.tolist())
                listofcolornew  = [listofcolor[i] for i in classmyind.MAPPING_CLASS.tolist()]
                #print(listofcolornew)
                #['forestgreen', 'olive', 'magenta', 'darkorange', 'darkorange', 'forestgreen', 'forestgreen', 'darkorange', 'greenyellow', 'greenyellow']

                listofname = selmyFile.LEGEND.tolist()
                listoflegendnew  = [listofname[i] for i in classmyind.MAPPING_CLASS.tolist()]
                #print(listoflegendnew)

                cmap = mpl.colors.ListedColormap(listofcolornew)
                cmap.set_over('0')
                cmap.set_under(str(num_class))

                bounds_list=list(np.arange(0.1, num_class+0.1, 1))
                bounds = [-1]+bounds_list
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                fig = plt.figure(figsize = (18,18))

                plt.imshow(x,cmap=cmap,norm=norm, interpolation='none')
                plt.colorbar()
                plt.title(listoflegendnew)

                p = str(datetime.date.today())
                fig.savefig(''.join([nomefile_mappa, "_", mapping, "_", p, ".png"]))
        return None
