#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File        :   SRMwithLOSS.py
@Time        :   2022/07/05 13:19:01
@Author      :   Zhongyang Hu
@Version     :   1.0.0
@Contact     :   z.hu@uu.nl
@Publication :   
@Desc        :   SR Models and Loss functions
'''



import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_addons as tfa
from keras import backend as K 

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from skimage.metrics import structural_similarity

##############################################################################################
#####     LOSS 
##############################################################################################

def ssim(a,b):

  im1 = tf.expand_dims(a, axis=-1)
  im2 = tf.expand_dims(b, axis=-1)

  return tf.image.ssim(im1,im2, 5, filter_size=3)


def SSIM_Loss(y_true, y_pred):

    return 1 - tf.reduce_mean(ssim(y_true, y_pred))
    #return 1 - tf.reduce_mean(structural_similarity(y_true, y_pred, full=True))


def CMAE_loss(y_true, y_pred):

    return tf.reduce_mean(tf.keras.metrics.mean_absolute_error(y_true, y_pred))

def DEM_Loss(DEM_hr):
    def calc_loss(y_true, y_pred):
        fact = 10/(1+np.exp(-0.005*DEM_hr+6))
        return tf.reduce_mean(y_pred*fact)
    return calc_loss



def VGG_Loss(y_true, y_pred):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape = [74,74,3])
    vgg.trainable = False
    content_layers = 'block3_conv4'

    lossModel = tf.keras.models.Model([vgg.input],vgg.get_layer(content_layers).output,name = 'vgg_Layer')

    #Xt = preprocess_input(y_pred)
    #Yt = preprocess_input(y_true)
    y_pred=tf.expand_dims(y_pred, axis=-1)
    y_pred = tf.keras.layers.concatenate([y_pred,y_pred,y_pred],axis=-1)

    y_true=tf.expand_dims(y_true, axis=-1)
    y_true = tf.keras.layers.concatenate([y_true,y_true,y_true],axis=-1)
    
    vggX = lossModel.predict(y_true)
    vggY = lossModel.predict(y_pred)
    
    #return tf.reduce_mean(tf.square(vggY-vggX))
    return np.mean(np.square(np.array(vggY).flatten()-np.array(vggX).flatten()))

def Y_pool(x):
    p=tf.keras.layers.AveragePooling2D(5,padding='same')(x)
    p.trainable = False
    lr_pred = tf.keras.layers.concatenate([p,p,p],axis=-1)
    return lr_pred

def SMLT_Loss(input_lr):
    def calc_loss(y_true, y_pred):
        lr_es =  Y_pool(y_pred)
        return tf.reduce_mean(tf.square(input_lr-lr_es))
    return calc_loss

##############################################################################################
#####     SRResNet
##############################################################################################

def down_sampling_conv(input_img,Fact):
    Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(input_img)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1)    
    Conv1 =tf.keras.layers.Activation(Fact)(Conv1)
    Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(Conv1)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1) 
    output_img =  tf.math.add(input_img, Conv1)
    return output_img

def down_sampling_conv_val(input_img,Fact):
    Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(input_img)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1)    
    Conv1 =tf.keras.layers.Activation(Fact)(Conv1)
    Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(Conv1)
    Conv1 = tf.keras.layers.BatchNormalization()(Conv1) 
    return Conv1

def RCAB(x):

  H,W,C = x.shape[-3], x.shape[-2], x.shape[-1]
  res0 = x
  Conv1= tf.keras.layers.Conv2D(filters= C,kernel_size=1, strides=(1, 1), padding='valid', use_bias=True,activation='PReLU')(x)
  res1 = Conv1
  
  Conv2= tf.keras.layers.Conv2D(filters= C,kernel_size=1, strides=(1, 1), padding='valid', use_bias=True,activation='PReLU')(x)

  Pool1= tfa.layers.AdaptiveAveragePooling2D(1)(Conv2)
  sig= tf.keras.activations.sigmoid(Pool1)

  att=tf.math.multiply(sig,res1)
  add_res=tf.math.add( att, res0)

  return add_res

def CSAM(x):
  res = x
  CM_Conv3D = tf.keras.layers.Conv3D(1, (3,3,3), activation='sigmoid', padding='same', input_shape=x.shape[2:])(x)
  out =  tf.math.multiply(CM_Conv3D, x)
  out = tf.math.add(out,res)
  out = tf.reshape(out, (-1, res.shape[-4],res.shape[-3],res.shape[-2]))
  return out

def U_Conv(x):
    Fact = 'PReLU'
    U_Conv1 = tf.keras.layers.Conv2D(256, 3, padding = 'same')(x)  
    U_Conv1 = tf.nn.depth_to_space(U_Conv1, 2)
    U_Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv1) # 28   
    U_Conv1 =tf.keras.layers.Activation(Fact)(U_Conv1)  
    U_Conv1 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv1) # 26   
    U_Conv1 =tf.keras.layers.Activation(Fact)(U_Conv1)     

    U_Conv2 = tf.keras.layers.Conv2D(64*3*3, 3, padding = 'same')(U_Conv1)  
    U_Conv2 = tf.nn.depth_to_space(U_Conv2, 3)
    U_Conv2 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv2) # 76   
    U_Conv2 =tf.keras.layers.Activation(Fact)(U_Conv2)  
    U_Conv2 = tf.keras.layers.Conv2D(64, 3, padding = 'valid')(U_Conv2) # 74   
    Out_U_Conv =tf.keras.layers.Activation(Fact)(U_Conv2)  

    return Out_U_Conv




### --- Baseline

def SRResNet(LR_Melt):
    n_d = 16 
    Fact = 'PReLU'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # Downsampling
    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer
    
    # Upsampling
    Up_Conv = U_Conv(Conv_R)  

    # MLP (Simple)
    HR_Melt = tf.keras.layers.Conv2D(1, 1,  padding = 'same')(Up_Conv)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))                                       
 
    model = tf.keras.models.Model(inputs = LR_Melt, outputs=HR_Melt)

    return model



### --- MOD1: Simple Add
def SRResNet_Simple_Add(LR_Melt, HR_MALB, HR_DEM):
    n_d = 16 
    Fact = 'PReLU'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # Downsampling
    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer
    
    # Upsampling
    Up_Conv = U_Conv(Conv_R)   

    # Simple Add
    HR_MALB_E = tf.expand_dims(HR_MALB,-1)
    HR_MALB_C = tf.keras.layers.Conv2D(64, 3, padding = 'same')(HR_MALB_E)
    HR_MALB_CA = tf.keras.layers.Activation(Fact)(HR_MALB_C)
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_DEM_C = tf.keras.layers.Conv2D(64, 3, padding = 'same')(HR_DEM_E)
    HR_DEM_CA = tf.keras.layers.Activation(Fact)(HR_DEM_C)
    HR_Comb = tf.keras.layers.concatenate([Up_Conv, HR_MALB_CA, HR_DEM_CA], axis=-1, name = 'Physics_info') 
    # Simple Add Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(HR_Comb) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    HR_Melt = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))

    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM], outputs=[HR_Melt])
    return model


### --- MOD 2 
def SRResNet_PA(LR_Melt, HR_MALB, HR_DEM, HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # Downsampling
    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_R)  
    

    # Simple Add
    HR_MALB_E = tf.expand_dims(HR_MALB,-1)
    HR_MALB_C = tf.keras.layers.Conv2D(64, 3, padding = 'same')(HR_MALB_E)
    HR_MALB_CA = tf.keras.layers.Activation(Fact)(HR_MALB_C)
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_DEM_C = tf.keras.layers.Conv2D(64, 3, padding = 'same')(HR_DEM_E)
    HR_DEM_CA = tf.keras.layers.Activation(Fact)(HR_DEM_C)
    HR_Comb = tf.keras.layers.concatenate([Up_Conv,HR_MALB_CA,HR_DEM_CA], axis=-1, name = 'Physics_info') 
    # Simple Add Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(HR_Comb) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv)

    # MLP (Simple) 
    Out_Conv = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    Out_Conv= tf.reshape(Out_Conv,(-1, 74,74))

    # Physical Activation
    HRF = tf.math.divide(1.36,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.005,HR_DEM),1) ))))
    Corr_Out = tf.math.multiply(Out_Conv , HRF)
    HRF2 =tf.math.divide(1.0,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.05,HR_Alb5),37) ))))
    HR_Melt = tf.math.multiply(Corr_Out , HRF2)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))

    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM, HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 3 
def SRResNet_SCA(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # Downsampling
    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_R)    

    # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    HR_Melt = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM, HR_Alb5], outputs=[HR_Melt])

    return model

### --- MOD 3.1 
def SRResNet_SCA_Deeper(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # Downsampling
    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_R)    

    # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_DEM_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_DEM_E)
    HR_DEM_CA = tf.keras.layers.Activation(Fact)(HR_DEM_C)

    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_ALB_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_ALB_E)
    HR_ALB_CA = tf.keras.layers.Activation(Fact)(HR_ALB_C)

    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Alb5_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_Alb5_E)
    HR_Alb5_CA = tf.keras.layers.Activation(Fact)(HR_Alb5_C)

    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_CA, HR_ALB_CA, HR_Alb5_CA], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    HR_Melt = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM, HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 4 
def SRResNet_CC(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # CC Block
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Comb = tf.keras.layers.concatenate([HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'GEO_info') 
    Conv_G = HR_Comb
    # CC Downsampling
    for i in range(2):
      Conv_G = down_sampling_conv_val(Conv_G,Fact)
      Conv_G = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(Conv_G)
    # CC Attention
    Conv_GAL=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Conv_G)
    Conv_GA=tf.math.multiply(Conv_GAL, Conv_G)

    # Downsampling
    for j in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer

    # CC and Encoder
    Conv_RA = tf.math.add(Conv_GA, Conv_R)
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_RA)  

    # MLP (Simple)
    HR_Melt = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Up_Conv)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 5 
def SRResNet_SCA_PA(LR_Melt, HR_MALB, HR_DEM, HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # Downsampling
    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_R)    

    # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    Out_Conv = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    Out_Conv= tf.reshape(Out_Conv,(-1, 74,74))

    # Physical Activation
    HRF = tf.math.divide(1.36,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.005,HR_DEM),1) ))))
    Corr_Out = tf.math.multiply(Out_Conv , HRF)
    HRF2 =tf.math.divide(1.0,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.05,HR_Alb5),37) ))))
    HR_Melt = tf.math.multiply(Corr_Out , HRF2)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 5.1 
def SRResNet_SCA_PA_Deeper(LR_Melt, HR_MALB, HR_DEM, HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # Downsampling
    for i in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_R)    

    # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_DEM_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_DEM_E)
    HR_DEM_CA = tf.keras.layers.Activation(Fact)(HR_DEM_C)

    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_ALB_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_ALB_E)
    HR_ALB_CA = tf.keras.layers.Activation(Fact)(HR_ALB_C)

    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Alb5_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_Alb5_E)
    HR_Alb5_CA = tf.keras.layers.Activation(Fact)(HR_Alb5_C)

    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_CA, HR_ALB_CA, HR_Alb5_CA], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    Out_Conv = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    Out_Conv= tf.reshape(Out_Conv,(-1, 74,74))

    # Physical Activation
    HRF = tf.math.divide(1.36,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.005,HR_DEM),1) ))))
    Corr_Out = tf.math.multiply(Out_Conv , HRF)
    HRF2 =tf.math.divide(1.0,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.05,HR_Alb5),37) ))))
    HR_Melt = tf.math.multiply(Corr_Out , HRF2)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 6 
def SRResNet_CC_PA(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

    # CC Block
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Comb = tf.keras.layers.concatenate([HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'GEO_info') 
    Conv_G = HR_Comb
    # CC Downsampling
    for i in range(2):
      Conv_G = down_sampling_conv_val(Conv_G,Fact)
      Conv_G = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(Conv_G)
    # CC Attention
    Conv_GAL=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Conv_G)
    Conv_GA=tf.math.multiply(Conv_GAL, Conv_G)

    # Downsampling
    for j in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer

    # CC and Encoder
    Conv_RA = tf.math.add(Conv_GA, Conv_R)
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_RA)  

    # MLP (Simple)
    Out_Conv = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Up_Conv)
    Out_Conv= tf.reshape(Out_Conv,(-1, 74,74))

    # Physical Activation
    HRF = tf.math.divide(1.36,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.005,HR_DEM),1) ))))
    Corr_Out = tf.math.multiply(Out_Conv , HRF)
    HRF2 =tf.math.divide(1.0,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.05,HR_Alb5),37) ))))
    HR_Melt = tf.math.multiply(Corr_Out , HRF2)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 7 
def SRResNet_CC_SCA(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

     # CC Block
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Comb = tf.keras.layers.concatenate([HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'GEO_info') 
    Conv_G = HR_Comb
    # CC Downsampling
    for i in range(2):
      Conv_G = down_sampling_conv_val(Conv_G,Fact)
      Conv_G = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(Conv_G)
    # CC Attention
    Conv_GAL=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Conv_G)
    Conv_GA=tf.math.multiply(Conv_GAL, Conv_G)

    # Downsampling
    for j in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer

    # CC and Encoder
    Conv_RA = tf.math.add(Conv_GA, Conv_R)
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_RA)  

     # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    HR_Melt = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))

    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model

### --- MOD 7.1 
def SRResNet_CC_SCA_Deeper(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

     # CC Block
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Comb = tf.keras.layers.concatenate([HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'GEO_info') 
    Conv_G = HR_Comb
    # CC Downsampling
    for i in range(2):
      Conv_G = down_sampling_conv_val(Conv_G,Fact)
      Conv_G = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(Conv_G)
    # CC Attention
    Conv_GAL=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Conv_G)
    Conv_GA=tf.math.multiply(Conv_GAL, Conv_G)

    # Downsampling
    for j in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer

    # CC and Encoder
    Conv_RA = tf.math.add(Conv_GA, Conv_R)
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_RA)  

    # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_DEM_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_DEM_E)
    HR_DEM_CA = tf.keras.layers.Activation(Fact)(HR_DEM_C)

    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_ALB_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_ALB_E)
    HR_ALB_CA = tf.keras.layers.Activation(Fact)(HR_ALB_C)

    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Alb5_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_Alb5_E)
    HR_Alb5_CA = tf.keras.layers.Activation(Fact)(HR_Alb5_C)

    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_CA, HR_ALB_CA, HR_Alb5_CA], axis=-1, name = 'Physics_info')  
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    HR_Melt = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))

    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 8 
def SUPREME(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

     # CC Block
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Comb = tf.keras.layers.concatenate([HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'GEO_info') 
    Conv_G = HR_Comb
    # CC Downsampling
    for i in range(2):
      Conv_G = down_sampling_conv_val(Conv_G,Fact)
      Conv_G = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(Conv_G)
    # CC Attention
    Conv_GAL=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Conv_G)
    Conv_GA=tf.math.multiply(Conv_GAL, Conv_G)

    # Downsampling
    for j in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer

    # CC and Encoder
    Conv_RA = tf.math.add(Conv_GA, Conv_R)
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_RA)  

     # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    Out_Conv = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    Out_Conv= tf.reshape(Out_Conv,(-1, 74,74))

    # Physical Activation
    HRF = tf.math.divide(1.36,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.005,HR_DEM),1) ))))
    Corr_Out = tf.math.multiply(Out_Conv , HRF)
    HRF2 =tf.math.divide(1.0,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.05,HR_Alb5),37) ))))
    HR_Melt = tf.math.multiply(Corr_Out , HRF2)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model

### --- MOD 8.1 
def SUPREME_Deeper(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

     # CC Block
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Comb = tf.keras.layers.concatenate([HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'GEO_info') 
    Conv_G = HR_Comb
    # CC Downsampling
    for i in range(2):
      Conv_G = down_sampling_conv_val(Conv_G,Fact)
      Conv_G = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(Conv_G)
    # CC Attention
    Conv_GAL=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Conv_G)
    Conv_GA=tf.math.multiply(Conv_GAL, Conv_G)

    # Downsampling
    for j in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer

    # CC and Encoder
    Conv_RA = tf.math.add(Conv_GA, Conv_R)
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_RA)  

    # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_DEM_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_DEM_E)
    HR_DEM_CA = tf.keras.layers.Activation(Fact)(HR_DEM_C)

    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_ALB_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_ALB_E)
    HR_ALB_CA = tf.keras.layers.Activation(Fact)(HR_ALB_C)

    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Alb5_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_Alb5_E)
    HR_Alb5_CA = tf.keras.layers.Activation(Fact)(HR_Alb5_C)

    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_CA, HR_ALB_CA, HR_Alb5_CA], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 

    # MLP (Simple)
    Out_Conv = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    Out_Conv= tf.reshape(Out_Conv,(-1, 74,74))

    # Physical Activation
    HRF = tf.math.divide(1.36,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.005,HR_DEM),1) ))))
    Corr_Out = tf.math.multiply(Out_Conv , HRF)
    HRF2 =tf.math.divide(1.0,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.05,HR_Alb5),37) ))))
    HR_Melt = tf.math.multiply(Corr_Out , HRF2)
    HR_Melt= tf.reshape(HR_Melt,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model


### --- MOD 8.2 
def SUPREME_Deeper_NoAlb(LR_Melt, HR_MALB, HR_DEM,HR_Alb5):
    n_d = 16 
    Fact = 'PReLU'
    kernel_siz_RG=3
    padding_RG='same'

    # Pre-normalization
    Conv0 = tf.keras.layers.Conv2D(64, 3, padding = 'same')(LR_Melt)
    Conv_R =tf.keras.layers.Activation(Fact)(Conv0)
    Conv_GSC = Conv_R

     # CC Block
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Comb = tf.keras.layers.concatenate([HR_DEM_E, HR_ALB_E, HR_Alb5_E], axis=-1, name = 'GEO_info') 
    Conv_G = HR_Comb
    # CC Downsampling
    for i in range(2):
      Conv_G = down_sampling_conv_val(Conv_G,Fact)
      Conv_G = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(Conv_G)
    # CC Attention
    Conv_GAL=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Conv_G)
    Conv_GA=tf.math.multiply(Conv_GAL, Conv_G)

    # Downsampling
    for j in range(n_d):
      Conv_R = down_sampling_conv(Conv_R,Fact)
    Conv_R = tf.math.add(Conv_GSC, Conv_R) # Global skip layer

    # CC and Encoder
    Conv_RA = tf.math.add(Conv_GA, Conv_R)
    
    # Upsampling
    Up_Conv  = U_Conv(Conv_RA)  

    # SCA
    HR_DEM_E = tf.expand_dims(HR_DEM,-1)
    HR_DEM_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_DEM_E)
    HR_DEM_CA = tf.keras.layers.Activation(Fact)(HR_DEM_C)

    HR_ALB_E = tf.expand_dims(HR_MALB,-1)
    HR_ALB_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_ALB_E)
    HR_ALB_CA = tf.keras.layers.Activation(Fact)(HR_ALB_C)

    HR_Alb5_E = tf.expand_dims(HR_Alb5,-1)
    HR_Alb5_C = tf.keras.layers.Conv2D(32, 3, padding = 'same')(HR_Alb5_E)
    HR_Alb5_CA = tf.keras.layers.Activation(Fact)(HR_Alb5_C)

    GEOINFO = tf.keras.layers.concatenate([Up_Conv, HR_DEM_CA, HR_ALB_CA, HR_Alb5_CA], axis=-1, name = 'Physics_info') 
    CSAM_in = tf.keras.layers.Conv2D(filters= 64, kernel_size=kernel_siz_RG, strides=(1, 1), padding=padding_RG, use_bias=True)(GEOINFO)
    CSAM_in = tf.expand_dims(CSAM_in,-1)
    CSAM_out = CSAM(CSAM_in) 
    Conv_SA=tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(CSAM_out)
    Conv_GA2=tf.math.multiply(CSAM_out, Conv_SA)
    # SCA Add
    MLP_in=tf.math.add(Up_Conv, Conv_GA2)
    # SCA Mixture
    Combine_Conv = tf.keras.layers.Conv2D(64, 3, padding = 'same')(MLP_in) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(32, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Activation(Fact)(Combine_Conv) 
    Combine_Conv = tf.keras.layers.Conv2D(16, 3, padding = 'same')(Combine_Conv) # 74   
    Combine_Conv =tf.keras.layers.Ac\tivation(Fact)(Combine_Conv) 

    # MLP (Simple)
    Out_Conv = tf.keras.layers.Conv2D(1, 1, padding = 'same')(Combine_Conv)
    Out_Conv= tf.reshape(Out_Conv,(-1, 74,74))

    # Physical Activation
    HRF = tf.math.divide(1.36,(tf.math.add(1.0, tf.math.exp( tf.math.subtract(tf.math.multiply(0.005,HR_DEM),1) ))))
    Corr_Out = tf.math.multiply(Out_Conv , HRF)
    HR_Melt= tf.reshape(Corr_Out,(-1, 74,74))
    
    model = tf.keras.models.Model(inputs = [LR_Melt, HR_MALB, HR_DEM,HR_Alb5], outputs=[HR_Melt])

    return model



    