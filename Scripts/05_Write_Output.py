import tensorflow as tf
import numpy as np
import datetime
import rasterio
import xarray as xr

s_var=['snowmelt','snowmelt','snowmelt']
n_var=len(s_var)
IDs=np.reshape(np.array(list(range(0,88464))),(304,291))
x,y = int(np.where(IDs==24492)[0]),int(np.where(IDs==24492)[1])
xs = x-77
ys = y-44

xe = x + 11 * 19
ye = y + 11 * 21

xs55 = int(x*27/5.5)-54*7
ys55 = int(y*27/5.5)-54*4



class ANT:
    def __init__(self, model_mode, input_mode, year, calc_mode,doy):
        self.year=year
        self.model_mode=model_mode
        self.input_mode=input_mode
        self.model_name = model_mode+'.tf'
        self.calc_mode = calc_mode
        self.doy = doy
        self.model_outname='G:/My Drive/NeurIPS/Experiment/Models/'+model_mode  
        self.ANT_NPY_outname='G:/My Drive/NeurIPS/Experiment/Results_NPY/Test/'+model_mode+'_Year_'+str(year)+'_SSE.npy'
        self.out_tif_name='G:/My Drive/NeurIPS/Experiment/Results_NPY/Test/'+model_mode+'_Year_'+str(year)+'_SSE.tif'
        self.stand_mode=False
        self.model = tf.keras.models.load_model('G:/My Drive/NeurIPS/Experiment/Models/'+model_mode+'.tf',custom_objects=None, compile=False)

    def calc_ANT_NPY(self):
        ANT_27_fn = 'C:/Users/zh_hu/Documents/ARSM/BATCH/Output/Variables/snowmelt_Res27000_ALL_2011.npy'
        t_stamps = np.arange('2001-01-01', '2019-09-01', dtype='datetime64')
        data = np.load(ANT_27_fn)*86400

        data_sub = data[:,xs:xe,ys:ye]
        Ts, Hs, Ws = data_sub.shape
        del data_sub

        t_stamps=t_stamps.astype(datetime.datetime)

        temp_dem = np.load('G:/My Drive/Albedo_ANT/Training/ANT_TANDEM_Merge_Max_add_5500_Cubic_Extend_F.npy')
        temp_dem[temp_dem<0]=0
        temp_alb5 = np.load('G:/My Drive/Albedo_ANT/Alb_Char/ANT_Extent/MCD43_WSA_stack_export_5P_Near_Entend.npy') 
        
        output_img = np.zeros((Hs//11*54,Ws//11*54))

        #temp_alb = np.load('G:/My Drive/Albedo_ANT/Training/AP/Output/Alb/Merged/malb_Res5500_ALL_'+str(self.year)+'.npy')
        temp_alb = np.load('G:/My Drive/NeurIPS/Experiment/Results_NPY/Test/ALB_Year_'+str(self.year)+'_FNAN.npy')
        #temp_alb = np.moveaxis(temp_alb,0,2)
 
        temp_alb[np.where(np.isnan(temp_alb))]=np.nanmedian(temp_alb) #800
        temp_alb5[np.where(np.isnan(temp_alb5))]=800#np.nanmedian(temp_alb5) #800

        tloc_s = int(np.where(t_stamps == np.datetime64(str(self.year)+'-07-01'))[0])
        tloc_e = int(np.where(t_stamps == np.datetime64(str(self.year+1)+'-07-01'))[0])
        
        for i in range(Hs//11):
            for j in range(Ws//11):

                print('ij',i,j)
                I,J = (xs + i*11), (ys + j*11)
                raw = data[tloc_s:tloc_e,(I-2):(I+13),(J-2):(J+13)]

                LR_SMLT=np.zeros((len(range(tloc_s,tloc_e)),15,15,3))

                for b in range(3):
                    LR_SMLT[:,:,:,b]=raw
                del raw   

                I55,J55 = (xs55 + i*54), (ys55 + j*54)

                if self.input_mode == 'RACMO_single':
                    HR_out=self.model.predict(LR_SMLT, verbose=0)
                    del LR_SMLT
        
                elif self.input_mode == 'RACMO_ALB_DEM':
            
                    HR_MALB = temp_alb[:,(I55-10):(I55+64),(J55-10):(J55+64)]
                    HR_DEM = np.zeros(HR_MALB.shape)
                    for dp in range(HR_MALB.shape[0]):
                        HR_DEM[dp,:,:] = temp_dem[(I55-10):(I55+64),(J55-10):(J55+64)]


                    print('MALB Shape',HR_MALB.shape)
                    HR_out=self.model.predict([LR_SMLT,HR_MALB,HR_DEM], verbose=0)
                    del LR_SMLT,HR_MALB,HR_DEM

                elif self.input_mode == 'RACMO_ALB_DEM_C':
            
                    HR_MALB = temp_alb[:,(I55-10):(I55+64),(J55-10):(J55+64)]
                    HR_DEM = np.zeros(HR_MALB.shape)
                    HR_alb5 = np.zeros(HR_MALB.shape)
                    for dp in range(HR_MALB.shape[0]):
                        HR_DEM[dp,:,:] = temp_dem[(I55-10):(I55+64),(J55-10):(J55+64)]
                        HR_alb5[dp,:,:] = temp_alb5[0,(I55-10):(I55+64),(J55-10):(J55+64)]

                    print('MALB Shape',HR_MALB.shape)
                    HR_out=self.model.predict([LR_SMLT,HR_MALB,HR_DEM,HR_alb5], verbose=0)
                    del LR_SMLT,HR_MALB,HR_DEM

                elif self.input_mode == 'RACMO_ALB_DEM_CNA':
                    HR_MALB = temp_alb[:,(I55-10):(I55+64),(J55-10):(J55+64)]
                    HR_DEM = np.zeros(HR_MALB.shape)
                    HR_alb5 = np.zeros(HR_MALB.shape)
                    for dp in range(HR_MALB.shape[0]):
                        HR_DEM[dp,:,:] = temp_dem[(I55-10):(I55+64),(J55-10):(J55+64)]
                        HR_alb5[dp,:,:] = temp_alb5[0,(I55-10):(I55+64),(J55-10):(J55+64)]

                    print('SMLT Shape',LR_SMLT.shape)
                    print('MALB Shape',HR_MALB.shape)
                    print('DEM Shape',HR_DEM.shape)
                    print('ALB5 Shape',HR_alb5.shape)
                    HR_out=self.model.predict([LR_SMLT,HR_MALB,HR_DEM,HR_alb5], verbose=0)
                    del LR_SMLT,HR_MALB,HR_DEM

                else:
                    print('Unkown Mode')

                if self.calc_mode == 'DOY':
                    output_img[(i*54):(i*54+54),(j*54):(j*54+54)]=HR_out[self.doy,10:64,10:64]
                else:
                    output_img[(i*54):(i*54+54),(j*54):(j*54+54)]=np.sum(HR_out[:,10:64,10:64],axis=0)

        if self.calc_mode == 'DOY':
            outname_NPY_apply_mod = self.ANT_NPY_outname[:-4] + str(self.doy) + '.npy'
        else:
            outname_NPY_apply_mod = self.ANT_NPY_outname

        np.save(outname_NPY_apply_mod,output_img)
    

    def ANT_NPY_to_TIFF(self):
        TIFF = rasterio.open('C:/Users/zh_hu/Desktop/ARM5/Data/R5500.tif')
        if self.calc_mode == 'DOY':
            outname_NPY_apply_mod = self.ANT_NPY_outname[:-4] + str(self.doy) + '.npy'
        else:
            outname_NPY_apply_mod = self.ANT_NPY_outname

        out_img = np.load(outname_NPY_apply_mod)
        new_meta = TIFF.meta.copy()

        if self.calc_mode == 'DOY':
            outname_TIF_apply_mod = self.out_tif_name[:-4] + str(self.doy) + '.npy'
        else:
            outname_TIF_apply_mod = self.out_tif_name
        

        with rasterio.open(outname_TIF_apply_mod, "w", **new_meta) as dest:
            dest.write(out_img.reshape(1,out_img.shape[0], out_img.shape[1]))





















