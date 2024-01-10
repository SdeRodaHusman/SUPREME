##################################################################################################
###### ----------- Preprocessing RACMO
##################################################################################################

py_TF='C:/Users/zh_hu/Documents/Test/TF2/Scripts/python' 
Preprocessing_dir = 'C:/Users/zh_hu/Documents/SUPREME_test'


RACMO_outdir    =    Preprocessing_dir + '/Data'
file_dir        =    Preprocessing_dir + '/Variables'
train_dir       =    Preprocessing_dir + '/train'
dev_dir         =    Preprocessing_dir + '/dev'
fn              =   'C:/Users/zh_hu/Documents/SUPREME_test/Data/RACMO2/Height_latlon_XPEN055.nc'
checking_mode   =   'N'
shp_dir         =   'C:/Users/zh_hu/Documents/SUPREME_test/Data/GRID/'


##################################################################################################
# ##### ----------- Training
##################################################################################################
overall_parent_dir= 'C:/Users/zh_hu/Documents/SUPREME_test/Experiment/'

aoi_nr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
s_var=['snowmelt','snowmelt','snowmelt']
bs=16 #16
strides=2
md =  5e-4# 0.005 
emb_dim=0 # 48
saveh5=True


##################################################################################################
# ##### ----------- Application
##################################################################################################