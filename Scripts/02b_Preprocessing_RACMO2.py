#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File        :   Data_Preprocessing.py
@Time        :   2021/12/14 13:12:01
@Author      :   Zhongyang Hu
@Version     :   1.0.0
@Contact     :   z.hu@uu.nl
@Publication :   
@Desc        :   ARSM Preprocessing: RACMO2 to Monthly Numpy Array
'''
import os
import xarray as xr
import numpy as np
import time
import argparse

#########################################################################################################################################################
##### ----------- 1. Config. from ARGV
#########################################################################################################################################################

print('Warning: GDAL version >=3.1.4 is required, checking the GDAL version below!')
os.system('gdalinfo --version')
print()

# setup the parser
parser = argparse.ArgumentParser(description='ARSM Preprocessing Modual 1: RACMO2 to Monthly Numpy Array')

parser.add_argument('parent_dir', type=str, help='Parent directory')
parser.add_argument('racmo_res', type=int, help='RACMO2 resolution (27000 or 5500)')
parser.add_argument('racmo_fn', type=str, help='RACMO2 filename')
parser.add_argument('clip_prefix', type=str, help='Prefix for saving clipped Data')
parser.add_argument('saving_ANT', type=str, help='Save unclipped data for ANT')
parser.add_argument('var_temporal', type=str, help='mono or multi-temporal (Y: multi, N: mono)')
parser.add_argument('checking_mode', type=str, help='Checking data (features) integrity [Y/N]')

# parse the args
args = parser.parse_args()
#----------------------------------------------------------------------------------------------------------------------------------------------------------

parent_dir=args.parent_dir
res=args.racmo_res
racmo_fn = args.racmo_fn
clip_prefix = args.clip_prefix
saving_ANT=args.saving_ANT
var_temporal = args.var_temporal
checking_mode = args.checking_mode

Param = racmo_fn.split('.')[0]
grid_fn = str(res)+'_GRID.shp'

tif_fn = os.path.join(parent_dir,'RACMO2',racmo_fn)
shp_fn = os.path.join(parent_dir,'GRID', grid_fn)

print('Tif found: ' + str(os.path.isfile(tif_fn)))
print('Shp found: ' + str(os.path.isfile(shp_fn)))

#########################################################################################################################################################
##### ----------- 2. Convert NetCDF to GeoTIFF
#########################################################################################################################################################

def NC_to_TIFF(fn, param, res, out_fn_var, exe=False):
    out_fn_proj=out_fn_var[:-4]+'_epsg3031.tif'
    nc_in=xr.open_dataset(fn)
    rlat_in=nc_in['rlat'].data
    rlon_in=nc_in['rlon'].data
    proj=(nc_in['rotated_pole'].attrs)['proj4_params']
    nc_in.close()
    
    code_1 ='gdal_translate NETCDF:"{}":{} -a_ullr {} {} {} {} {}'.format(fn,param,min(rlon_in),max(rlat_in),max(rlon_in),min(rlat_in),out_fn_var)
    if exe:
        print('[Prosessing Start (GDAL)]: Concert NetCDF to GeoTIFF')
        os.system(code_1)
    print(code_1)
    print()
        
    code_2 ='gdalwarp -s_srs "{}" -t_srs "EPSG:3031" -tr {} -{} -r near {} {}'.format(proj,res,res,out_fn_var,out_fn_proj)
    if exe:
        print('[Prosessing Start (GDAL)]: Reproject to EPSG: 3031')
        os.system(code_2)
    print(code_2)
    print()

os.chdir(parent_dir)
print('Change to Parent Directory: ' + parent_dir)

if os.path.isdir(os.path.join(parent_dir,'Output'))==False:
        os.mkdir(os.path.join(parent_dir,'Output'))

if os.path.isdir(os.path.join(parent_dir,'Output','Cache'))==False:
        os.mkdir(os.path.join(parent_dir,'Output','Cache'))

for k in range(1,13):
    
    if os.path.isdir(os.path.join(parent_dir,'Output',Param))==False:
        os.mkdir(os.path.join(parent_dir,'Output',Param))

    nc_in=xr.open_dataset(tif_fn)
    Dates=nc_in['time'].data

    ys = Dates[0].astype('datetime64[Y]').astype(int)+1970
    ye = Dates[-1].astype('datetime64[Y]').astype(int)+1970 +1

    for yr in range(ys, ye):

        if yr!=2019:
            me=13
        else:
            me=9

        for mo in range (1,me):
            
            out_fn_tif = os.path.join(parent_dir,'Output','Cache',Param+'_Res'+str(res)+'_Y'+str(yr)+'_M'+str(mo)+'.tif')
            f=Param+'_Res'+str(res)+'_Y'+str(yr)+'_M'+str(mo) +'_epsg3031'+'.tif'

            if os.path.isfile(os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy'))==False:
                if checking_mode == 'Y':
                    print(os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy')+' is missing!')
                    print('Start processing the missing file!')

                print(os.path.join(parent_dir,'Output',Param,os.path.basename(out_fn_tif [:-4]+'_epsg3031.tif')))
                if os.path.isfile(os.path.join(parent_dir,'Output','Cache',Param,os.path.basename(out_fn_tif [:-4]+'_epsg3031.tif')))==False and os.path.isfile(os.path.join(parent_dir,'Output',Param,os.path.basename(out_fn_tif [:-4]+'_epsg3031.tif')))==False:
        
                    Year_array=np.array([d.astype('datetime64[Y]').astype(int)+1970 for d in Dates])
                    Month_array = np.array([d.astype('datetime64[M]').astype(int) % 12 + 1 for d in Dates]) 
                    Dates_loc=np.where((Year_array == yr) & (Month_array == mo) )[0]
                    nc_in.close()

                    time_clip_out = os.path.join(parent_dir,'Output','Cache',Param+'_Res'+str(res)+'_Y'+str(yr)+'_M'+str(mo)+'.nc')
                    code_subtime = 'ncks -d time,{},{} {} {}'.format(int(Dates_loc.min()),int(Dates_loc.max()),tif_fn,time_clip_out)

                    print('[Prosessing Start (NCKS)]: Clip RACMO2 ' + str(res) + ' ' + Param + ' to Year: ' + str(yr) + ' Month: ' + str(mo))
                    print(code_subtime)
                    print()
                    os.system(code_subtime)

                    # Reproj 
        
                    NC_to_TIFF(time_clip_out,Param,res, out_fn_tif, True) # Debug 2
            
                    os.rename(out_fn_tif[:-4] +'_epsg3031.tif', os.path.join(parent_dir,'Output',Param,os.path.basename(out_fn_tif [:-4]+'_epsg3031.tif')))
                    time.sleep(1)
                    os.remove(time_clip_out)
                    time.sleep(1)
                    os.remove(out_fn_tif)
                    time.sleep(1)

#########################################################################################################################################################
##### ----------- 3. Clip GeoTIFF to  AOIs
#########################################################################################################################################################
import json
import geopandas as gpd
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import glob

class data_prep:
     
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

os.chdir(os.path.join(parent_dir,'Output',Param))
print('Change to Param Directory: ' + os.path.join(parent_dir,'Output',Param))
print()

# Make subdir
for i in range(1,14):
    if os.path.isdir(clip_prefix+'AOI'+str(i))==False:
        os.mkdir(clip_prefix+'AOI'+str(i))
        
files = glob.glob('*Res'+str(res)+'*.tif')    
f=files[0]


if res == 27000:
    GRID_27=data_prep(shp_fn,f)
    AP27={'NW':34117,'NE': 34127, 'SW': 37027, 'SE':37037}

    x_s, x_e, y_s, y_e = latlon_to_xy_loc(GRID_27,AP27)
    AOI_1 = (x_s-11*3), (x_e-11*3), (y_s-11*2), (y_e-11*2)

    AOI_2 = (x_s-11*2), (x_e-11*2), (y_s-11*2), (y_e-11*2)
    AOI_3 = (x_s-11*2), (x_e-11*2), (y_s-11), (y_e-11)

    AOI_4 = (x_s-11), (x_e-11), (y_s-11*2), (y_e-11*2)

    AOI_5 = (x_s-11), (x_e-11), (y_s-11), (y_e-11)
    AOI_6 = (x_s-11), (x_e-11), y_s, y_e
    AOI_7 = (x_s-11), (x_e-11), (y_s+11), (y_e+11)

    AOI_8 = x_s, x_e, (y_s-11), (y_e-11)
    AOI_9 = x_s, x_e, y_s, y_e
    AOI_10 = x_s, x_e, (y_s+11), (y_e+11)

    AOI_11 = (x_s+11), (x_e+11), (y_s-11), (y_e-11)
    AOI_12 = (x_s+11), (x_e+11), y_s, y_e
    AOI_13 = (x_s+11), (x_e+11), (y_s+11), (y_e+11)
    AOIs = [AOI_1, AOI_2, AOI_3, AOI_4, AOI_5, AOI_6, AOI_7, AOI_8, AOI_9, AOI_10,  AOI_11, AOI_12, AOI_13]

    for f in files:
        k=1
        GRID_27=data_prep(shp_fn,f)
        for aoi in AOIs:
            if os.path.isfile(os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy'))==False:
                AOI_IMG=GRID_27.tif[:,aoi[0]:aoi[1], aoi[2]: aoi[3]]
                clipped = AOI_IMG[:,:,:].values
                clip_out_fn=os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy')
                print('[Prosessing Start]: Save Numpy Array '+clip_out_fn)
                if os.path.isfile(clip_out_fn)==False:
                    np.save(clip_out_fn, clipped)
            else:
                print(os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy') + ' already exists!')

            k+=1

if res == 5500:
    GRID_55=data_prep(shp_fn,f)
    AP55={'NW':130726,'NE': 130779, 'SW': 157014, 'SE':157067}
    x_s, x_e, y_s, y_e = latlon_to_xy_loc(GRID_55,AP55)

    AOI_1 = (x_s-54*3), (x_e-54*3), (y_s-54*2), (y_e-54*2)

    AOI_2 = (x_s-54*2), (x_e-54*2), (y_s-54*2), (y_e-54*2)
    AOI_3 = (x_s-54*2), (x_e-54*2), (y_s-54), (y_e-54)

    AOI_4 = (x_s-54), (x_e-54), (y_s-54*2), (y_e-54*2)

    AOI_5 = (x_s-54), (x_e-54), (y_s-54), (y_e-54)
    AOI_6 = (x_s-54), (x_e-54), y_s, y_e
    AOI_7 = (x_s-54), (x_e-54), (y_s+54), (y_e+54)

    AOI_8 = x_s, x_e, (y_s-54), (y_e-54)
    AOI_9 = x_s, x_e, y_s, y_e
    AOI_10 = x_s, x_e, (y_s+54), (y_e+54)

    AOI_11 = (x_s+54), (x_e+54), (y_s-54), (y_e-54)
    AOI_12 = (x_s+54), (x_e+54), y_s, y_e
    AOI_13 = (x_s+54), (x_e+54), (y_s+54), (y_e+54)
    AOIs = [AOI_1, AOI_2, AOI_3, AOI_4, AOI_5, AOI_6, AOI_7, AOI_8, AOI_9, AOI_10,  AOI_11, AOI_12, AOI_13]

    for f in files:
        k=1
        GRID_55=data_prep(shp_fn,f)
        for aoi in AOIs:
            if os.path.isfile(os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy'))==False:
                AOI_IMG=GRID_55.tif[:,aoi[0]:aoi[1], aoi[2]: aoi[3]]
                clipped = AOI_IMG[:,:,:].values
                clip_out_fn=os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy')
                print('[Prosessing Start]: Save Numpy Array '+clip_out_fn)
                if os.path.isfile(clip_out_fn)==False:
                    np.save(clip_out_fn, clipped)
                k+=1
            else:
                print(os.path.join(parent_dir,'Output',Param,clip_prefix+'AOI'+str(k),f[:-4]+'_AOI_'+str(k)+'.npy') + ' already exists!')


if saving_ANT=='Y':

    if os.path.isdir(os.path.join(parent_dir,'Output',Param,clip_prefix+'ANT'))==False:
        os.mkdir(os.path.join(parent_dir,'Output',Param,clip_prefix+'ANT'))

    for f in files:

        if os.path.isfile(os.path.join(parent_dir,'Output',Param,clip_prefix+'ANT',f[:-4]+'_ANT'+'.npy'))==False:
            IMG=xr.open_rasterio(f).values
            clip_out_fn=os.path.join(parent_dir,'Output',Param,clip_prefix+'ANT',f[:-4]+'_ANT'+'.npy')
            print('[Prosessing Start]: Save Numpy Array '+clip_out_fn)
            if os.path.isfile(clip_out_fn)==False:
                np.save(clip_out_fn, IMG)
        else:
            print(os.path.join(parent_dir,'Output',Param,clip_prefix+'ANT',f[:-4]+'_ANT'+'.npy') + ' already exists!')


