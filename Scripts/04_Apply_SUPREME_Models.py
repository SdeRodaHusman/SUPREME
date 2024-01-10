# C:/Users/zh_hu/Documents/Test/TF2/Scripts/python
# -*-coding:utf-8 -*-
'''
@File        :   Data_Preprocessing.py
@Time        :   2022/12/19 10:55:01
@Author      :   Zhongyang Hu
@Version     :   1.0.0
@Contact     :   z.hu@uu.nl
@Publication :   
@Desc        :   Using BASED for SUPREME Main Programme
'''
# ------ Python config and packages
import os
import math
import BASED_Config
import shutil
import tensorflow as tf
import numpy as np

from BASED_Preprocessing import RACMO_Tool
from BASED_Preprocessing import GEE_Preprocessing
from BASED_Preprocessing import RACMO2_Preprocessing_MonoTemporal

from BASED_Training import BASED_Models_MF as BASED_Models
from BASED_Training import BASED_ultility

from BASED_Application import BASED_ANT_Sophie as SAM
from BASED_Application import BASED_AP_APP
from BASED_Application import BASED_Plot_NPY

# ---------------------------------------------------------------


list_models = ['V18_SUPREME_Deeper_SSE']
for mm in list_models:
    input_mode = 'RACMO_ALB_DEM_CNA'
    for exe_year in range(2001, 2018):
        Benchmark = SAM.ANT(mm, input_mode, exe_year, 'Year', -9)
        Benchmark.calc_ANT_NPY()
        Benchmark.ANT_NPY_to_TIFF()
        del Benchmark
