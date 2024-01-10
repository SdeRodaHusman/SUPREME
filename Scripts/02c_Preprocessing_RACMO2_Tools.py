import xarray as xr
import os
import sys

import json
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

import datetime
import tensorflow as tf
import time
import os
import numpy as np

from scipy import ndimage
from pylab import *
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

import xarray as xr

class NC_to_TIFF:

    def __init__(self, nc_fn):
        self.nc = nc_fn

    def do_NC_to_TIFF(self, param, res, out_fn_var, exe=False):
        out_fn_proj=out_fn_var[:-4]+'_epsg3031.tif'
        nc_in=xr.open_dataset(self.nc)
        rlat_in=nc_in['rlat'].data
        rlon_in=nc_in['rlon'].data
        try:
            proj=(nc_in['rotated_pole'].attrs)['proj4_params']
        except:
            if res == 27000:
                proj = '-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0'
                print('No Projection Info: trying -m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0')
            elif res == 5500:
                proj = '-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=30.0'
                print('No Projection Info: trying -m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=30.0')
            else:
                sys.exit('[Warning] No Projection Info: Ending here!')

        nc_in.close()
    
        code_1 ='gdal_translate NETCDF:"{}":{} -a_ullr {} {} {} {} {}'.format(self.nc,param,min(rlon_in),max(rlat_in),max(rlon_in),min(rlat_in),out_fn_var)
        if exe:
            print('[Prosessing Start (GDAL)]: Concert NetCDF to GeoTIFF')
            os.system(code_1)
        print(code_1)
        print()
        time.sleep(10)
        
        code_2 ='gdalwarp -s_srs "{}" -t_srs "EPSG:3031" -tr {} -{} -r near {} {}'.format(proj,res,res,out_fn_var,out_fn_proj)
        if exe:
            print('[Prosessing Start (GDAL)]: Reproject to EPSG: 3031')
            os.system(code_2)
        print(code_2)
        print()

class AOI_clip:
     
    def __init__(self, grid_fn, tif_fn):
        self.grid = gpd.read_file(grid_fn)
        self.tif = xr.open_rasterio(tif_fn)
        
    def get_coor(self, index):
        gdf = self.grid
        gdf_sub = gdf[gdf['DN']==index]
        feature = [json.loads(gdf_sub.to_json())['features'][0]['geometry']]
        coors = feature [0]['coordinates'][0]
        x1 = (max([x[0] for x in coors])+min([x[0] for x in coors]))/2
        x2 = (max([x[1] for x in coors])+min([x[1] for x in coors]))/2
        
        return x1,x2    
    
    def get_xy(self, xs, ys):
        ds = self.tif 
        y=np.where(ds['y']==ds.sel(x=xs, y=ys, method="nearest")['y'])[0]
        x=np.where(ds['x']==ds.sel(x=xs, y=ys, method="nearest")['x'])[0]
        
        return int(x), int(y)




def latlon_to_xy(tif_in, locs):
    x0, y0 = tif_in.get_coor(locs['NW'])
    x1, y1 = tif_in.get_xy(x0,y0)
    
    x2, y2 = tif_in.get_coor(locs['SE'])
    x3, y3 = tif_in.get_xy(x2,y2)
    
    return tif_in.tif[:,y1:(y3+1),x1:(x3+1)]

def latlon_to_xy_loc(tif_in, locs):
    x0, y0 = tif_in.get_coor(locs['NW'])
    x1, y1 = tif_in.get_xy(x0,y0)
    
    x2, y2 = tif_in.get_coor(locs['SE'])
    x3, y3 = tif_in.get_xy(x2,y2)
    
    return y1, (y3+1), x1, (x3+1)

def filename_YM(fn):
    fn_name_base=os.path.basename(fn).split('_')
    return fn_name_base[2][1:]+fn_name_base[3][1:].zfill(2)

class Merge_NPY:
    def __init__(self, varname, parent_dir, aoi_nr, res,prefix):
        self.varname = varname
        self.parent_dir=parent_dir
        self.aoi_nr= aoi_nr
        self.res=res
        self.prefix = prefix

    def var_merge(self,out_filename, save_data_npy):
        var_path_label = os.path.join(self.parent_dir, self.varname, self.prefix+'AOI'+str(self.aoi_nr)) #t2m
        npy_list_label = glob.glob(var_path_label+'/*Res'+str(self.res)+'*.npy')
        npy_list_label_sorted = sorted(npy_list_label, key=filename_YM)

        for files in npy_list_label_sorted:
            print(files)

        print('-'*20)

        init_npy_label = np.load(npy_list_label_sorted[0])
        for fn_len in range(len(npy_list_label_sorted)-1):
            temp_npy_label= np.load(npy_list_label_sorted[fn_len+1])
            init_npy_label = np.concatenate((init_npy_label, temp_npy_label), axis=0)

        Labels_smlt = init_npy_label
        if save_data_npy == 'Y':
            out_fn=os.path.join(self.parent_dir,'Variables', out_filename)
            if os.path.isdir(os.path.join(self.parent_dir,'Variables')) == False:
                os.mkdir(os.path.join(self.parent_dir,'Variables'))
            np.save(out_fn, Labels_smlt)

    def var_merge_ANT(self,out_filename, save_data_npy):
        var_path_label = os.path.join(self.parent_dir, self.varname, self.prefix+'ANT') #t2m
        npy_list_label = glob.glob(var_path_label+'/*Res'+str(self.res)+'*.npy')
        npy_list_label_sorted = sorted(npy_list_label, key=filename_YM)

        for files in npy_list_label_sorted:
            print(files)

        print('-'*20)

        init_npy_label = np.load(npy_list_label_sorted[0])
        for fn_len in range(len(npy_list_label_sorted)-1):
            temp_npy_label= np.load(npy_list_label_sorted[fn_len+1])
            init_npy_label = np.concatenate((init_npy_label, temp_npy_label), axis=0)

        Labels_smlt = init_npy_label
        if save_data_npy == 'Y':
            out_fn=os.path.join(self.parent_dir,'Variables', out_filename)
            if os.path.isdir(os.path.join(self.parent_dir,'Variables')) == False:
                os.mkdir(os.path.join(self.parent_dir,'Variables'))
            np.save(out_fn, Labels_smlt)




class XY_creator:
    def __init__(self,Fdir):
        self.file_dir = Fdir

    def load_data(self, param_name, aoi_nr,res, mode):
        if mode=='partial':
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'_2001.npy')
        elif mode=='full':
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'_2011.npy')
        elif mode=='55cv':
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'F2.npy')
        else:
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'F.npy')
        
        raw_x = np.load(var_fn)
        return raw_x

    def create_X_RACMO(self, param_names,aoi_nr,res,mode):
        if mode=='full':
          x_processed = np.empty((6817, 11, 11, len(param_names)))
        else:  
          x_processed = np.empty((3652, 11, 11, len(param_names)))
        k=0
        for var_name in param_names:
          if var_name == 'snowmelt':
            x_processed[:,:,:,k]=self.load_data(var_name, aoi_nr,res,mode)*86400

          elif var_name == 'dir':
            U=self.load_data('u10m', aoi_nr,res,mode)
            V=self.load_data('v10m', aoi_nr,res,mode)
            for t in range(x_processed.shape[0]):
              wdir = np.mod(180+np.rad2deg(np.arctan2(U[t,:,:], V[t,:,:])),360)
              x_processed[t,:,:,k]=wdir

          else:
            x_processed[:,:,:,k]=self.load_data(var_name, aoi_nr,res,mode)
          k+=1

        return x_processed

    def create_GEO(self,aoi_nr,bands):

        lat_x = np.load(self.file_dir+'/lat/lat_5500_epsg3031_AOI_'+str(aoi_nr)+'.npy')
        lon_x = np.load(self.file_dir+'/lon/lon_5500_epsg3031_AOI_'+str(aoi_nr)+'.npy')
        hei_x = np.load(self.file_dir+'/height/height_5500_epsg3031_AOI_'+str(aoi_nr)+'.npy')
        mask2d_x = np.load(self.file_dir+'/mask2d/mask2d_5500_epsg3031_AOI_'+str(aoi_nr)+'.npy')

        mask2d_x[mask2d_x<0.5]=0 # 0.5
        mask2d_x[mask2d_x>=0.5]=1 # 0.5
          
        if bands ==4:
            geo = np.empty((54,54,4))
            geo[:,:,0] = lat_x 
            geo[:,:,1] = lon_x 
            geo[:,:,2] = hei_x 
            geo[:,:,3] = mask2d_x 
        else:
            geo = np.empty((54,54,2))
            geo[:,:,0] = hei_x 
            geo[:,:,1] = mask2d_x 

        return geo

def standardize(image_data):
    img=copy.deepcopy(image_data)
    for b in range(image_data.shape[-1]):
      img[:,:,:,b] -= np.mean(image_data[:,:,:,b], axis=0)
      img[:,:,:,b] /= np.std(image_data[:,:,:,b], axis=0)
    
    img[np.isnan(img)]=996
    
    return img

def Masking(x,ml, mode):
  
  masked = np.zeros(x.shape)
  
  if mode:
    for t in range(x.shape[0]):
      for b in range(x.shape[3]):
        masked[t,:,:,b]=x[t,:,:,b]*ml

  else:
    for t in range(x.shape[0]):
        masked[t,:,:]=x[t,:,:]*ml

  return masked


def Train_Dev_Filter(X,Y,Z,MO=['10','11','12','01','02','03','04','05','06'], AOI_NR=None, step=0):

  t_stamps = np.arange('2001-01-01', '2011-01-01', dtype='datetime64')
  
  # Month ONDJFM
  mo = np.array([str(i).split('-')[1] for i in t_stamps])
  mo_loc = np.where(np.in1d(mo, MO))[0]
  sl_loc=mo_loc

  # Year
  t_stamps_OM =  t_stamps[np.intersect1d(sl_loc, mo_loc)]

  #print(t_stamps_OM)
  yr = np.array([str(i).split('-')[0] for i in t_stamps_OM])

  mo_loc_train = np.intersect1d(sl_loc, mo_loc)[np.where(yr<'2007')]
  mo_loc_test = np.intersect1d(sl_loc, mo_loc)[np.where(yr>='2007')]

  train_x = X[mo_loc_train ,:,:,:]
  train_y = Y[mo_loc_train ,:,:]
  train_z = Z[mo_loc_train ,:,:,:]


  test_x = X[mo_loc_test,:,:,:]
  test_y = Y[mo_loc_test,:,:]
  test_z = Z[mo_loc_test,:,:,:]

  return train_x, train_y, train_z,  test_x,  test_y,  test_z #, mask

def overlap_train(file_dir,s_var, stand, step, topo, aoi_nr, MO=['10','11','12','01','02','03','04','05','06']):
    n_var=len(s_var)
    train_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    train_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    train_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))

    test_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    test_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    test_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))


    index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    t_stamps = np.arange('2001-01-01', '2011-01-01', dtype='datetime64')

    n_var=len(s_var)

    AP_X=np.zeros((len(t_stamps), 11*7,11*6,n_var))
    AP_Y=np.zeros((len(t_stamps),54*7,54*6))
    AP_Z=np.zeros((len(t_stamps),54*7,54*6,topo))
    Senario1 = XY_creator(file_dir)


    for AOI_NR in range(1,14): 
        rs=index[AOI_NR-1][0]+1
        cs=index[AOI_NR-1][1]+1

        AP_X[:,(11*rs):(11*rs+11),(11*cs):(11*cs+11),:] = Senario1.create_X_RACMO(s_var,AOI_NR,27000,'partial')
        AP_Y[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = Senario1.load_data('snowmelt', AOI_NR, 5500,'partial')*86400

        Senario1_GEO = Senario1.create_GEO(AOI_NR, topo)
        temp_z = np.zeros((int(len(t_stamps)),54,54,topo))

        for t in range(int(len(t_stamps))):
            temp_z[t,:,:,:]=Senario1_GEO[:,:,:]
        AP_Z[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54),:] = temp_z

        del temp_z

    if stand:
        AP_X=standardize(AP_X)
        AP_Z=standardize(AP_Z)

    for AOI_NR in aoi_nr:

        rs=index[AOI_NR-1][0]+1
        cs=index[AOI_NR-1][1]+1
  

        AOI_X = AP_X[:,(11*rs-2):(11*rs+11+2),(11*cs-2):(11*cs+11+2),:]
        AOI_Y = AP_Y[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10)]
        AOI_Z = AP_Z[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10),:]

        train_X, train_Y, train_Z,  test_X,  test_Y,  test_Z= Train_Dev_Filter(AOI_X,AOI_Y,AOI_Z, MO, AOI_NR, step)

 
        train_Y = Masking(train_Y, train_Z[0,:,:,3],False)
        test_Y = Masking(test_Y,test_Z[0,:,:,3],False)

        train_Xa = np.concatenate((train_Xa, train_X), axis=0)
        train_Ya = np.concatenate((train_Ya, train_Y), axis=0)
        train_Za = np.concatenate((train_Za, train_Z), axis=0)

        test_Xa = np.concatenate((test_Xa, test_X), axis=0)
        test_Ya = np.concatenate((test_Ya, test_Y), axis=0)
        test_Za = np.concatenate((test_Za, test_Z), axis=0)

    del AP_X, AOI_Y, AOI_Z

    return  train_Xa,  train_Ya,  train_Za,  test_Xa, test_Ya, test_Za

def overlap_ALL(s_var, stand, topo, Odir):

    n_var=len(s_var)
    index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    t_stamps = np.arange('2001-01-01', '2019-09-01', dtype='datetime64')
    n_var=len(s_var)

    AP_X=np.zeros((len(t_stamps), 11*7,11*6,n_var))
    AP_Z=np.zeros((len(t_stamps),54*7,54*6,topo))

    Senario1 = XY_creator(Odir)


    for AOI_NR in range(1,14): 
        rs=index[AOI_NR-1][0]+1
        cs=index[AOI_NR-1][1]+1

        Raw_Melt = np.zeros((6817, 11, 11, 3))
        RACMO_2001 = np.load(os.path.join(Odir, 'snowmelt_AOI_'+str(AOI_NR)+'_'+str(27000)+'_2001.npy'))
        RACMO_2011 = np.load(os.path.join(Odir, 'snowmelt_AOI_'+str(AOI_NR)+'_'+str(27000)+'_2011.npy'))

        for iter_band in range(3):
            Raw_Melt[0:3652,:,:,iter_band]=RACMO_2001
            Raw_Melt[3652:6817,:,:,iter_band]=RACMO_2011

        Cor_Melt = Raw_Melt *86400
        AP_X[:,(11*rs):(11*rs+11),(11*cs):(11*cs+11),:] = Cor_Melt
        #AP_X[:,(11*rs):(11*rs+11),(11*cs):(11*cs+11),:] = Senario1.create_X_RACMO(s_var,AOI_NR,27000, 'full')

        Senario1_GEO = Senario1.create_GEO(AOI_NR,topo)
        temp_z = np.zeros((int(len(t_stamps)),54,54,4))
        
        for t in range(int(len(t_stamps))):
          temp_z[t,:,:,:]=Senario1_GEO[:,:,:]
        AP_Z[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54),:] = temp_z

        del temp_z

    if stand:
        AP_X=standardize(AP_X)
        AP_Z=standardize(AP_Z)

    return AP_X, AP_Z

def Apply_NPY(model_name, model_in, s_var, stand=False, CSO=None,topo=4):
    index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    model = tf.keras.models.load_model('C:/Users/zh_hu/Documents/SR3M/Model/'+model_name,custom_objects=CSO, compile=False)
    
    t_stamps = np.arange('2001-01-01', '2019-09-01', dtype='datetime64')
    output_daily=np.zeros((len(t_stamps),54*5,54*4))
    output_monthly=np.zeros((54*5,54*4,18*12))
    output_yearly=np.zeros((54*5,54*4,18))

    years=[]
    months=[]

    for y in range(2001,2020):
        for m in range(1,13):
            years.append(y)
            months.append(m)


    for AOI_NR in range(1,14):

      rs=index[AOI_NR-1][0]+1
      cs=index[AOI_NR-1][1]+1

      AOI_X = np.load('C:/Users/zh_hu/Documents/SR3M/Data/apply/Input_AOI_'+str(AOI_NR)+'_Daily.npy')
      
      if model_in=='solo':
        model_input=AOI_X
      elif model_in=='mask':
        #model_input=[AOI_X, AOI_Z[:,:,:,3]]
        print('needZ')
      else:
        #model_input=[AOI_X, AOI_Z]
        print('needZ')

      PRED_ALL = model.predict(model_input)

      t_stamps=t_stamps.astype(datetime.datetime)
      t_stamps_months = np.array([i.month for i in t_stamps])
      t_stamps_years = np.array([i.year for i in t_stamps])

      rs=index[AOI_NR-1][0]
      cs=index[AOI_NR-1][1]

      out_dim = len(PRED_ALL.shape)

      if model_in=='mask':
        output_daily[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)]=PRED_ALL[:,10:64,10:64]
      elif out_dim == 4:
        output_daily[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)]=PRED_ALL[:,10:64,10:64,0]
      elif out_dim == 3:
        output_daily[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)]=PRED_ALL[:,10:64,10:64]
      else:
        print('Unknow Dim')
    

      k=0

      for y in range(2001,2019):
            for m in range(1,13):
                tloc=np.where((t_stamps_months==m) & (t_stamps_years==y))[0]
                if model_in=='mask':
                    output_monthly[(54*rs):(54*rs+54),(54*cs):(54*cs+54),k]=np.sum(PRED_ALL[tloc,10:64,10:64],axis=0)
                else:
                    output_monthly[(54*rs):(54*rs+54),(54*cs):(54*cs+54),k]=np.sum(PRED_ALL[tloc,10:64,10:64,0],axis=0)
                k+=1

      del PRED_ALL, AOI_X

      k=0
      for y in range(2001,2019):
        s=np.where((np.array(years)==y) & (np.array(months)==7))[0][0]
        e=np.where((np.array(years)==(y+1)) & (np.array(months)==7))[0][0]
        output_yearly[:,:,k]=np.sum(output_monthly[:,:,int(s):int(e)],axis=2)
        k+=1

      

    if model_name[-3:]=='.tf':
      model_name=model_name[:-3]

    np.save('C:/Users/zh_hu/Documents/SR3M/Data/eval/'+model_name+'_Daily.npy',output_daily)
    np.save('C:/Users/zh_hu/Documents/SR3M/Data/eval/'+model_name+'_Monthly.npy',output_monthly)
    np.save('C:/Users/zh_hu/Documents/SR3M/Data/eval/'+model_name+'_Yearly.npy',output_yearly)



def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def plot_month(model_mode):

    if os.path.isdir('C:/Users/zh_hu/Documents/SR3M/Plots/Monthly/'+model_mode)==False:
        os.mkdir('C:/Users/zh_hu/Documents/SR3M/Plots/Monthly/'+model_mode)

    output_UNET=np.load('C:/Users/zh_hu/Documents/SR3M/Data/eval/'+model_mode+'_Monthly.npy')
    output_R5M=np.load('C:/Users/zh_hu/Documents/SR3M/Data/Compare/output_R5_Monthly_Masked.npy')
    output_R27M=np.load('C:/Users/zh_hu/Documents/SR3M/Data/Compare/output_R27_Monthly.npy')
    QSCAT_M=xr.open_rasterio('C:/Users/zh_hu/Documents/SR3M/Data/Compare/QSCAT_AP_GRID2_Montly_199908_200906.tif')

    for t in range(12*18):
        date_t=datetime.datetime(2001,1,1) +  relativedelta(months=+t)

        fig = plt.figure(figsize=(24,16))
        fig.suptitle('Results @ Year '+str(date_t.year) + ' Month '+str(date_t.month), y=0.92, fontsize=28,weight=2)

        plt.subplot(2, 3, 1)
        plt.imshow(output_R5M[:,:,t], cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,200)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('RACMO 5.5 km',fontsize=15, color='black',weight=2, labelpad=10)

        plt.subplot(2, 3, 2)
        plt.imshow(output_R27M[:,:,t], cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,200)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('RACMO 27 km',fontsize=15, color='black',weight=2, labelpad=10)

        plt.subplot(2, 3, 3)
        try:
            plt.imshow(QSCAT_M[t+17], cmap = 'RdBu_r')
        except:
            plt.imshow(output_R27M[:,:,t]*0, cmap = 'RdBu_r')

        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,31)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('QSCAT 4.55 km',fontsize=15, color='black',weight=2, labelpad=10)


        plt.subplot(2, 3, 4)
        plt.imshow(output_UNET[:,:,t], cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,200)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('SRDRN 5.5 km',fontsize=15, color='black',weight=2, labelpad=10)


        plt.subplot(2, 3, 5)
        #plt.imshow(ndimage.median_filter(output_UNET[:,:,t],size=2), cmap = 'RdBu_r')
        plt.imshow(ndimage.median_filter(output_UNET[:,:,t],size=2), cmap = 'RdBu_r')
        #plt.imshow(ndimage.median_filter(lee_filter(output_UNET[:,:,t],3),size=2), cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,200)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('SRDRN Mfilter 5.5 km',fontsize=15, color='black',weight=2, labelpad=10)


        plt.savefig('C:/Users/zh_hu/Documents/SR3M/Plots/Monthly/'+model_mode+"/{}.png".format(t), bbox_inches="tight",transparent=True)
        fig.clear()
        plt.close(fig)

def plot_year(model_mode):
    QSCAT_Y=xr.open_rasterio('C:/Users/zh_hu/Documents/SR3M/Data/Compare/QSCAT_AP_GRID2.tif')
    output_R5=np.load('C:/Users/zh_hu/Documents/SR3M/Data/Compare/output_R5_Yearly_Masked.npy')
    output_R27=np.load('C:/Users/zh_hu/Documents/SR3M/Data/Compare/output_R27_Yearly.npy')

    output_UNET=np.load('C:/Users/zh_hu/Documents/SR3M/Data/eval/'+model_mode+'_Yearly.npy')

    if os.path.isdir('C:/Users/zh_hu/Documents/SR3M/Plots/Yearly/'+model_mode)==False:
        os.mkdir('C:/Users/zh_hu/Documents/SR3M/Plots/Yearly/'+model_mode)

    for t in range(18):
        fig = plt.figure(figsize=(24,16))
        fig.suptitle('Results @ Year '+str(t+2001), y=0.68, fontsize=28,weight=2)

        plt.subplot(1, 4, 1)
        plt.imshow(output_R5[:,:,t], cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,450)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('RACMO 5.5 km',fontsize=15, color='black',weight=2, labelpad=10)


        plt.subplot(1, 4, 2)
        plt.imshow(output_R27[:,:,t], cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,450)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('RACMO 27 km',fontsize=15, color='black',weight=2, labelpad=10)


        plt.subplot(1, 4, 3)
        plt.imshow(output_UNET[:,:,t], cmap = 'RdBu_r')
        #plt.imshow(output_UNET[:,:,t], cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,450)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('SRDRN 5.5 km',fontsize=15, color='black',weight=2, labelpad=10)

        plt.subplot(1, 4, 4)
        try:
            plt.imshow(QSCAT_Y[t+2], cmap = 'RdBu_r')
        except:
            plt.imshow(output_R27[:,:,t]*0, cmap = 'RdBu_r')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.clim(0,450)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('QSCAT 4.55 km',fontsize=15, color='black',weight=2, labelpad=10)

        plt.savefig('C:/Users/zh_hu/Documents/SR3M/Plots/Yearly/'+model_mode+"/{}.png".format(t), bbox_inches="tight",transparent=True)
        fig.clear()
        plt.close(fig)
