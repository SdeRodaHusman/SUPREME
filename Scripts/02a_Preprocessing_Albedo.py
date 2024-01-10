import os
import tensorflow as tf
import numpy as np
import datetime


from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import scipy.ndimage as ndimage

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

for yr in range(2001,2020):

    temp_alb_2005 = np.load('G:/My Drive/Albedo_ANT/Training/AP/Output/Alb/Merged/'+'malb_Res5500_ALL_'+str(yr)+'.npy')
    temp_alb_2006 = np.load('G:/My Drive/Albedo_ANT/Training/AP/Output/Alb/Merged/'+'malb_Res5500_ALL_'+str(yr+1)+'.npy')

    date_temp_2005 = np.arange(str(yr)+'-01-01', str(yr+1)+'-01-01', dtype='datetime64')
    date_temp_2005 = date_temp_2005.astype(datetime.datetime)
    date_temp_months_2005 = np.array([i.month for i in date_temp_2005])  

    date_temp_2006 = np.arange(str(yr+1)+'-01-01', str(yr+2)+'-01-01', dtype='datetime64')
    date_temp_2006 = date_temp_2006.astype(datetime.datetime)
    date_temp_months_2006 = np.array([i.month for i in date_temp_2006])  
        
    temp_alb_2005[temp_alb_2005<0]=0
    temp_alb_2005[temp_alb_2005>1000]=1000

    temp_alb_2006[temp_alb_2006<0]=0
    temp_alb_2006[temp_alb_2006>1000]=1000

    alb_output = np.zeros((len(date_temp_2006), 1492, 1429))  
    alb_output_w = np.zeros((len(date_temp_2006), 1492, 1429))       
        
    date_idx_2005 = np.where(date_temp_months_2005>=7)[0]
    date_idx_2006 = np.where(date_temp_months_2006<7)[0]
        
    alb_output[0:len(date_idx_2005),:,:] = np.moveaxis(temp_alb_2005[:,:,date_idx_2005],2,0)
    alb_output[len(date_idx_2005)::,:,:] = np.moveaxis(temp_alb_2006[:,:,date_idx_2006],2,0)


    for i in range(365):
        if i<359:
            alb_output_w[i,:,:]= np.nanmedian(alb_output[i:(i+5),:,:],axis=0) 
        alb_output_w[i,:,:]=ndimage.median_filter(lee_filter(alb_output_w[i,:,:],3),size=2)

    np.save('G:/My Drive/NeurIPS/Experiment/Results_NPY/Test/ALB_Year_'+str(yr)+'_FNAN.npy',np.nan_to_num(alb_output_w))