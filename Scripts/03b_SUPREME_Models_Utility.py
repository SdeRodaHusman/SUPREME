import cv2
import copy
import os
import pandas as pd
import numpy as np



class XY_creator:
    def __init__(self,Fdir):
        self.file_dir = Fdir

    def load_data(self, param_name, aoi_nr,res, mode):
        if mode=='partial':
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'.npy')
        elif mode=='full':
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'_2011.npy')
        elif mode=='55cv':
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'F2.npy')
        else:
            var_fn = os.path.join(self.file_dir, param_name+'_AOI_'+str(aoi_nr)+'_'+str(res)+'F.npy')
        
        raw_x = np.load(var_fn)
        return raw_x

    def load_ssim(self, aoi_nr,res, mode):
        if mode=='partial':
            var_fn = os.path.join(self.file_dir, 'BT_SSIMS_E_scale100_stack_export_55_AP_AOI_'+str(aoi_nr)+'_Training.npy')
        elif mode=='full':
            var_fn = os.path.join(self.file_dir, 'BT_SSIMS_E_scale100_stack_export_55_AP_AOI_'+str(aoi_nr)+'_'+str(res)+'_2011.npy')
        elif mode=='55cv':
            var_fn = os.path.join(self.file_dir, 'BT_SSIMS_E_scale100_stack_export_55_AP_AOI_'+str(aoi_nr)+'_'+str(res)+'F2.npy')
        else:
            var_fn = os.path.join(self.file_dir, 'BT_SSIMS_E_scale100_stack_export_55_AP_AOI_'+str(aoi_nr)+'_'+str(res)+'F.npy')
        
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
        albc_median = np.load('G:/My Drive/Albedo_ANT/Alb_Char/Training/AP/Output/Alb/ALB_Char_NA_replaced_AOI'+str(aoi_nr)+'/MCD43_WSA_stack_export_Median_AP_AOI_'+str(aoi_nr)+'.npy')
        albc_stdv = np.load('G:/My Drive/Albedo_ANT/Alb_Char/Training/AP/Output/Alb/ALB_Char_NA_replaced_AOI'+str(aoi_nr)+'/MCD43_WSA_stack_export_Stdv_AP_AOI_'+str(aoi_nr)+'.npy')
        albc_5p =np.load('G:/My Drive/Albedo_ANT/Alb_Char/Training/AP/Output/Alb/ALB_Char_NA_replaced_AOI'+str(aoi_nr)+'/MCD43_WSA_stack_export_5P_AP_AOI_'+str(aoi_nr)+'.npy')

        mask2d_x[mask2d_x<0.5]=0 # 0.5
        mask2d_x[mask2d_x>=0.5]=1 # 0.5
          
        if bands ==4:
            geo = np.empty((54,54,4))
            geo[:,:,0] = lat_x 
            geo[:,:,1] = lon_x 
            geo[:,:,2] = hei_x 
            geo[:,:,3] = mask2d_x 
        elif bands==5:
            geo = np.empty((54,54,5))
            geo[:,:,0] = hei_x
            geo[:,:,1] = albc_median 
            geo[:,:,2] = albc_stdv
            geo[:,:,3] = mask2d_x 
            geo[:,:,4] = albc_5p
            

        else:
            geo = np.empty((54,54,2))
            geo[:,:,0] = hei_x 
            geo[:,:,1] = mask2d_x 

        return geo


def sum_fraction_N(img,i,j):
  w1 = np.array(list(range(0,55,5)))/10
  w2 = np.array(list(range(50,-5,-5)))/10

  Sub_54=img[(5*i-1): (5*i+5),(5*j-1): (5*j+5)]

  print((5*i-1), (5*i+4),(5*j-1), (5*j+4))

  Area_C = img[(5*i):(5*i+4), (5*j):(5*j+4)]
  print((5*i),(5*j+4), (5*j),(5*j+4))

  print(w1[0],w1[-1])
  print(w2[0],w2[-1])

  Area_T = img[(5*i-1),(5*j-1): (5*j+5)] * (np.array([w1[i],5.5,5.5,5.5,5.5,w2[i]])*w1[j]/(5.5*5.5))
  Area_B = img[(5*i+4),(5*j-1): (5*j+5)] * (np.array([w1[i],5.5,5.5,5.5,5.5,w2[i]])*w2[j]/(5.5*5.5))

  Area_L = img[(5*i): (5*i+4),(5*j)] * (w1[i] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))
  Area_R = img[(5*i): (5*i+4),(5*j+4)] * (w2[i] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))


  ws = Area_T.sum() + Area_B.sum() +Area_L.sum() + Area_R.sum() +Area_C.sum()
  print(ws)

  print(Area_C)

  return ws


def sum_fraction_N(img,i,j):
  w1 = np.array(list(range(0,55,5)))/10
  w2 = np.array(list(range(50,-5,-5)))/10

  Area_C = img[(5*i):(5*i+4), (5*j):(5*j+4)]

  if (i==0) & (j==0): # check
    Area_T = np.array([0])
    Area_B = img[(5*i+4),(5*j): (5*j+5)] * (np.array([5.5,5.5,5.5,5.5,w2[j]])*w2[i]/(5.5*5.5))

    Area_L = np.array([0])
    Area_R = img[(5*i): (5*i+4),(5*j+4)] * (np.array([5.5,5.5,5.5,5.5]) * w2[i] /(5.5*5.5))

  elif (i==0) & (j==10): # check

    Area_T = np.array([0])
    Area_B = img[(5*i+4),(5*j-1): (5*j+4)] * (np.array([w1[j],5.5,5.5,5.5,5.5])*w2[i]/(5.5*5.5))

    Area_L = img[(5*i): (5*i+4),(5*j)] * (w1[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))
    Area_R = np.array([0])


  elif (i==0) & (j!=0) & (j!=10): # check

    Area_T = np.array([0])
    Area_B = img[(5*i+4),(5*j-1): (5*j+5)] * (np.array([w1[j],5.5,5.5,5.5,5.5,w2[j]])*w2[i]/(5.5*5.5))

    Area_L = img[(5*i): (5*i+4),(5*j)] * (w1[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))
    Area_R = img[(5*i): (5*i+4),(5*j+4)] * (w2[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))


  elif (i!=0) & (j==0) & (i!=10):# check

    Area_T = img[(5*i-1),(5*j): (5*j+5)] * (np.array([5.5,5.5,5.5,5.5,w2[j]])*w1[i]/(5.5*5.5))
    Area_B = img[(5*i+4),(5*j): (5*j+5)] * (np.array([5.5,5.5,5.5,5.5,w2[j]])*w2[i]/(5.5*5.5))

    Area_L = np.array([0])
    Area_R = img[(5*i): (5*i+4),(5*j+4)] * (w2[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))



  elif (i==10) & (j==10): # check

    Area_T = img[(5*i-1),(5*j-1): (5*j+4)] * (np.array([w1[j],5.5,5.5,5.5,5.5])*w1[i]/(5.5*5.5))
    Area_B = np.array([0])

    Area_L = img[(5*i): (5*i+4),(5*j)] * (w1[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))
    Area_R = np.array([0])

  elif (i==10) & (j==0): # check

    Area_T = img[(5*i-1),(5*j): (5*j+5)] * (np.array([5.5,5.5,5.5,5.5,w2[j]])*w1[i]/(5.5*5.5))
    Area_B = np.array([0])

    Area_L = np.array([0])
    Area_R = img[(5*i): (5*i+4),(5*j+4)] * (w2[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))

  elif (i==10) & (j!=10) & (j!=0):

    Area_T = img[(5*i-1),(5*j-1): (5*j+5)] * (np.array([w1[j],5.5,5.5,5.5,5.5,w2[j]])*w1[i]/(5.5*5.5))
    Area_B = np.array([0])

    Area_L = img[(5*i): (5*i+4),(5*j)] * (w1[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))
    Area_R = img[(5*i): (5*i+4),(5*j+4)] * (w2[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))


  elif (i!=10) & (j==10) & (i!=0):

    Area_T = img[(5*i-1),(5*j-1): (5*j+4)] * (np.array([w1[j],5.5,5.5,5.5,5.5])*w1[i]/(5.5*5.5))
    Area_B = img[(5*i+4),(5*j-1): (5*j+4)] * (np.array([w1[j],5.5,5.5,5.5,5.5])*w2[i]/(5.5*5.5))

    Area_L = img[(5*i): (5*i+4),(5*j)] * (w1[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))
    Area_R = np.array([0])

  else:
    
    Area_T = img[(5*i-1),(5*j-1): (5*j+5)] * (np.array([w1[j],5.5,5.5,5.5,5.5,w2[j]])*w1[i]/(5.5*5.5))
    Area_B = img[(5*i+4),(5*j-1): (5*j+5)] * (np.array([w1[j],5.5,5.5,5.5,5.5,w2[j]])*w2[i]/(5.5*5.5))

    Area_L = img[(5*i): (5*i+4),(5*j)] * (w1[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))
    Area_R = img[(5*i): (5*i+4),(5*j+4)] * (w2[j] * np.array([5.5,5.5,5.5,5.5]) /(5.5*5.5))





  ws = (Area_T.sum() + Area_B.sum() +Area_L.sum() + Area_R.sum() +Area_C.sum())/(16+len(Area_T.flatten())+len(Area_B.flatten())+len(Area_L.flatten())+len(Area_R.flatten()))

  return ws

  
def C55to27(img):

  re = np.zeros((11,11))
  for i in range(11):
    for j in range(11):
      re[i,j]=sum_fraction_N(img,i,j)

  return re

def Bicubic(img,size):

  re = cv2.resize(img, size, interpolation = cv2.INTER_CUBIC)
  
  return re

def AreaINT(img,size):

  re = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
  
  return re


def C55to27_cv(img):

  re = cv2.resize(img, (11,11), interpolation = cv2.INTER_AREA)
  
  return re

def C55to27_cv_padding(img):

  re = cv2.resize(img, (15,15), interpolation = cv2.INTER_AREA)
  
  return re



def Train_Test_Filter(X,Y,SS,Z,SI,MO=['10','11','12','01','02','03','04','05','06'], AOI_NR=None, step=0):

  t_stamps = np.arange('2001-01-01', '2011-01-01', dtype='datetime64')
  
  # Month ONDJFM
  mo = np.array([str(i).split('-')[1] for i in t_stamps])
  mo_loc = np.where(np.in1d(mo, MO))[0]

  if SI:
      Similarity_df = pd.read_csv('G:/My Drive/Marion/Data/RACMO2/1D/Similarity_daily_AOI'+str(AOI_NR)+'.csv')
      # Similarity  
      sam = np.array(Similarity_df['SAM'])
      ssim = np.array(Similarity_df['SSIM']) 
      ss = np.array(Similarity_df['SS']) 
      #sl_loc = np.where((sam<0.63) | (ss<5.51) | (ssim>0.3))[0]
      sl_loc = np.where((sam<0.5) | (ss<5) | (ssim>0.5))[0]
  else:
      sl_loc=mo_loc

  # Year
  t_stamps_OM =  t_stamps[np.intersect1d(sl_loc, mo_loc)]

  #print(t_stamps_OM)
  yr = np.array([str(i).split('-')[0] for i in t_stamps_OM])

  mo_loc_train = np.intersect1d(sl_loc, mo_loc)[np.where(yr<'2007')]
  mo_loc_test = np.intersect1d(sl_loc, mo_loc)[np.where(yr>='2007')]

  train_x = X[mo_loc_train ,:,:,:]
  train_y = Y[mo_loc_train ,:,:]
  train_ss = SS[mo_loc_train ,:,:]
  train_z = Z[mo_loc_train ,:,:,:]


  test_x = X[mo_loc_test,:,:,:]
  test_y = Y[mo_loc_test,:,:]
  test_ss = SS[mo_loc_test,:,:]
  test_z = Z[mo_loc_test,:,:,:]

  #mask = np.zeros((54,54))
  #mask[train_Z[0,:,:,3]>0.5]=1

  return train_x, train_y, train_ss, train_z,  test_x,  test_y, test_ss, test_z #, mask



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

def standardize(image_data):
    img=copy.deepcopy(image_data)
    for b in range(image_data.shape[-1]):
      img[:,:,:,b] -= np.mean(image_data[:,:,:,b], axis=0)
      img[:,:,:,b] /= np.std(image_data[:,:,:,b], axis=0)
    
    img[np.isnan(img)]=996
    
    return img


def overlap_train(s_var, stand, step, topo, aoi_nr, SI, MO=['10','11','12','01','02','03','04','05','06']):
    n_var=len(s_var)
    train_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    train_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    train_SSa = np.zeros((0, 54+step*5*2, 54+step*5*2))
    train_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))

    test_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    test_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    test_SSa = np.zeros((0, 54+step*5*2, 54+step*5*2))
    test_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))


    index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    t_stamps = np.arange('2001-01-01', '2011-01-01', dtype='datetime64')

    n_var=len(s_var)

    AP_X=np.zeros((len(t_stamps), 11*7,11*6,n_var))
    AP_Y=np.zeros((len(t_stamps),54*7,54*6))
    AP_SS=np.zeros((len(t_stamps),54*7,54*6))
    AP_Z=np.zeros((len(t_stamps),54*7,54*6,topo))
    Senario1 = XY_creator('C:/Users/zh_hu/Documents/ARSM/BATCH/Output/Variables')


    for AOI_NR in range(1,14): 
        rs=index[AOI_NR-1][0]+1
        cs=index[AOI_NR-1][1]+1

        AP_X[:,(11*rs):(11*rs+11),(11*cs):(11*cs+11),:] = Senario1.create_X_RACMO(s_var,AOI_NR,27000,'partial')
        AP_Y[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = Senario1.load_data('snowmelt', AOI_NR, 5500,'partial')*86400
        #AP_SS[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = np.moveaxis(Senario1.load_data(gee_var, AOI_NR, 5500,'partial'),-1,0)
        AP_SS[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = np.moveaxis(Senario1.load_ssim(AOI_NR, 5500,'partial'),-1,0)

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
        AOI_SS = AP_SS[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10)]
        AOI_Z = AP_Z[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10),:]
        train_X, train_Y, train_SS, train_Z,  test_X,  test_Y, test_SS,  test_Z= Train_Test_Filter(AOI_X,AOI_Y, AOI_SS,AOI_Z, SI, MO, AOI_NR, step)

 
        train_Y = Masking(train_Y, train_Z[0,:,:,3],False)
        test_Y = Masking(test_Y,test_Z[0,:,:,3],False)

        train_Xa = np.concatenate((train_Xa, train_X), axis=0)
        train_Ya = np.concatenate((train_Ya, train_Y), axis=0)
        train_SSa = np.concatenate((train_SSa, train_SS), axis=0)
        train_Za = np.concatenate((train_Za, train_Z), axis=0)

        test_Xa = np.concatenate((test_Xa, test_X), axis=0)
        test_Ya = np.concatenate((test_Ya, test_Y), axis=0)
        test_SSa = np.concatenate((test_SSa, test_SS), axis=0)
        test_Za = np.concatenate((test_Za, test_Z), axis=0)

    del AP_X, AOI_Y, AOI_SS, AOI_Z

    return  train_Xa,  train_Ya, train_SSa,  train_Za,  test_Xa, test_Ya, test_SSa, test_Za


def overlap_train_alb_char(s_var, stand, step, topo, aoi_nr, SI, MO=['10','11','12','01','02','03','04','05','06']):
    n_var=len(s_var)
    train_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    train_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    train_SSa = np.zeros((0, 54+step*5*2, 54+step*5*2))
    train_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))

    test_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    test_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    test_SSa = np.zeros((0, 54+step*5*2, 54+step*5*2))
    test_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))


    index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    t_stamps = np.arange('2001-01-01', '2011-01-01', dtype='datetime64')

    n_var=len(s_var)

    AP_X=np.zeros((len(t_stamps), 11*7,11*6,n_var))
    AP_Y=np.zeros((len(t_stamps),54*7,54*6))
    AP_SS=np.zeros((len(t_stamps),54*7,54*6))
    AP_Z=np.zeros((len(t_stamps),54*7,54*6,topo))
    Senario1 = XY_creator('C:/Users/zh_hu/Documents/ARSM/BATCH/Output/Variables')


    for AOI_NR in range(1,14): 
        rs=index[AOI_NR-1][0]+1
        cs=index[AOI_NR-1][1]+1

        AP_X[:,(11*rs):(11*rs+11),(11*cs):(11*cs+11),:] = Senario1.create_X_RACMO(s_var,AOI_NR,27000,'partial')
        AP_Y[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = Senario1.load_data('snowmelt', AOI_NR, 5500,'partial')*86400
        AP_SS[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = np.moveaxis(Senario1.load_data('malb', AOI_NR, 5500,'partial'),-1,0)

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
        AOI_SS = AP_SS[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10)]
        AOI_Z = AP_Z[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10),:]
        train_X, train_Y, train_SS, train_Z,  test_X,  test_Y, test_SS,  test_Z= Train_Test_Filter(AOI_X,AOI_Y, AOI_SS,AOI_Z, SI, MO, AOI_NR, step)

 
        train_Y = Masking(train_Y, train_Z[0,:,:,3],False)
        test_Y = Masking(test_Y,test_Z[0,:,:,3],False)

        train_Xa = np.concatenate((train_Xa, train_X), axis=0)
        train_Ya = np.concatenate((train_Ya, train_Y), axis=0)
        train_SSa = np.concatenate((train_SSa, train_SS), axis=0)
        train_Za = np.concatenate((train_Za, train_Z), axis=0)

        test_Xa = np.concatenate((test_Xa, test_X), axis=0)
        test_Ya = np.concatenate((test_Ya, test_Y), axis=0)
        test_SSa = np.concatenate((test_SSa, test_SS), axis=0)
        test_Za = np.concatenate((test_Za, test_Z), axis=0)

    del AP_X, AOI_Y, AOI_SS, AOI_Z

    return  train_Xa,  train_Ya, train_SSa,  train_Za,  test_Xa, test_Ya, test_SSa, test_Za

def Train_Val_Test_Filter(X,Y,SS, Z, SI,MO=['10','11','12','01','02','03','04','05','06'], AOI_NR=None, step=0):

  t_stamps = np.arange('2001-01-01', '2019-09-01', dtype='datetime64')
  
  # Month ONDJFM
  mo = np.array([str(i).split('-')[1] for i in t_stamps])
  mo_loc = np.where(np.in1d(mo, MO))[0]

  if SI:
      Similarity_df = pd.read_csv('G:/My Drive/Marion/Data/RACMO2/1D/Similarity_daily_AOI'+str(AOI_NR)+'.csv')
      # Similarity  
      sam = np.array(Similarity_df['SAM'])
      ssim = np.array(Similarity_df['SSIM']) 
      ss = np.array(Similarity_df['SS']) 
      #sl_loc = np.where((sam<0.63) | (ss<5.51) | (ssim>0.3))[0]
      sl_loc = np.where((sam<0.5) | (ss<5) | (ssim>0.5))[0]
  else:
      sl_loc=mo_loc

  # Year
  t_stamps_OM =  t_stamps[np.intersect1d(sl_loc, mo_loc)]

  #print(t_stamps_OM)
  yr = np.array([str(i).split('-')[0] for i in t_stamps_OM])

  mo_loc_train = np.intersect1d(sl_loc, mo_loc)[np.where(yr<'2007')]
  mo_loc_val = np.intersect1d(sl_loc, mo_loc)[np.where((yr>='2007') & (yr<'2011'))]
  mo_loc_test = np.intersect1d(sl_loc, mo_loc)[np.where(yr>='2011')]

  train_x = X[mo_loc_train ,:,:,:]
  train_y = Y[mo_loc_train ,:,:]
  train_ss = SS[mo_loc_train ,:,:]
  train_z = Z[mo_loc_train ,:,:,:]

  val_x = X[mo_loc_val ,:,:,:]
  val_y = Y[mo_loc_val ,:,:]
  val_ss = SS[mo_loc_val ,:,:]
  val_z = Z[mo_loc_val ,:,:,:]

  test_x = X[mo_loc_test,:,:,:]
  test_y = Y[mo_loc_test,:,:]
  test_ss = SS[mo_loc_test,:,:]
  test_z = Z[mo_loc_test,:,:,:]

  #mask = np.zeros((54,54))
  #mask[train_Z[0,:,:,3]>0.5]=1

  return train_x, train_y, train_ss, train_z, val_x, val_y, val_ss, val_z,  test_x,  test_y, test_ss,  test_z #, mask

def overlap_evaluation(s_var, stand, step, topo, aoi_nr, SI, MO=['10','11','12','01','02','03','04','05','06']):
    n_var=len(s_var)

    train_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    train_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    train_SSa = np.zeros((0, 54+step*5*2, 54+step*5*2))
    train_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))

    val_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    val_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    val_SSa = np.zeros((0, 54+step*5*2, 54+step*5*2))
    val_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))

    test_Xa = np.zeros((0, 11+step*2, 11+step*2, n_var))
    test_Ya = np.zeros((0, 54+step*5*2, 54+step*5*2))
    test_SSa = np.zeros((0, 54+step*5*2, 54+step*5*2))
    test_Za = np.zeros((0, 54+step*5*2, 54+step*5*2, topo))


    index=[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
    t_stamps = np.arange('2001-01-01', '2019-09-01', dtype='datetime64')

    n_var=len(s_var)

    AP_X=np.zeros((len(t_stamps), 11*7,11*6,n_var))
    AP_Y=np.zeros((len(t_stamps),54*7,54*6))
    AP_SS=np.zeros((len(t_stamps),54*7,54*6))
    AP_Z=np.zeros((len(t_stamps),54*7,54*6,topo))
    Senario1 = XY_creator('C:/Users/zh_hu/Documents/ARSM/BATCH/Output/Variables')


    for AOI_NR in range(1,14): 
        rs=index[AOI_NR-1][0]+1
        cs=index[AOI_NR-1][1]+1

        AP_X[:,(11*rs):(11*rs+11),(11*cs):(11*cs+11),:] = Senario1.create_X_RACMO(s_var,AOI_NR,27000,'full')
        AP_Y[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = Senario1.load_data('snowmelt', AOI_NR, 5500,'full')*86400
        AP_SS[:,(54*rs):(54*rs+54),(54*cs):(54*cs+54)] = np.moveaxis(Senario1.load_data('malb', AOI_NR, 5500,'full'),-1,0)

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
        AOI_SS = AP_SS[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10)]
        AOI_Z = AP_Z[:,(54*rs-10):(54*rs+54+10),(54*cs-10):(54*cs+54+10),:]

        train_X, train_Y, train_SS, train_Z, val_X, val_Y, val_SS, val_Z,  test_X,  test_Y, test_SS,  test_Z= Train_Val_Test_Filter(AOI_X,AOI_Y,AOI_SS,AOI_Z, SI, MO, AOI_NR, step)


 
        train_Y = Masking(train_Y, train_Z[0,:,:,3],False)
        test_Y = Masking(test_Y,test_Z[0,:,:,3],False)

        train_Xa = np.concatenate((train_Xa, train_X), axis=0)
        train_Ya = np.concatenate((train_Ya, train_Y), axis=0)
        train_SSa = np.concatenate((train_SSa, train_SS), axis=0)
        train_Za = np.concatenate((train_Za, train_Z), axis=0)

        val_Xa = np.concatenate((val_Xa, val_X), axis=0)
        val_Ya = np.concatenate((val_Ya, val_Y), axis=0)
        val_SSa = np.concatenate((val_SSa, val_SS), axis=0)
        val_Za = np.concatenate((val_Za, val_Z), axis=0)

        test_Xa = np.concatenate((test_Xa, test_X), axis=0)
        test_Ya = np.concatenate((test_Ya, test_Y), axis=0)
        test_SSa = np.concatenate((test_SSa, test_SS), axis=0)
        test_Za = np.concatenate((test_Za, test_Z), axis=0)

    del AP_X, AOI_Y, AOI_Z, AOI_SS

    return  train_Xa,  train_Ya, train_SSa, train_Za, val_Xa, val_Ya, val_SSa, val_Za,  test_Xa, test_Ya, test_SSa, test_Za