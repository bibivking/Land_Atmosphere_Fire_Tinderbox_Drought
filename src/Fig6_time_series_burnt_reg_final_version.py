import os
import cartopy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.interpolate import griddata, interp1d
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, OCEAN
from common_utils import *

def plot_time_series_burn_unburn_region(FMI_file, Tmax_file, LAI_file, ALB_file, Qle_file,  VPD_file, SMtop_file,
                                 FMI_file_unburnt, Tmax_file_unburnt, LAI_file_unburnt, ALB_file_unburnt, Qle_file_unburnt, VPD_file_unburnt,SMtop_file_unburnt,
                                 daytime_FMI_file=None, daytime_FMI_file_unburnt=None,
                                 loc_lats=None, loc_lons=None, time_s=None, time_e=None):
    # Check whether mask right

    f_Tmax         = Dataset(Tmax_file, mode='r')
    f_LAI          = Dataset(LAI_file,  mode='r')
    f_ALB          = Dataset(ALB_file,  mode='r')
    f_Qle          = Dataset(Qle_file,  mode='r')
    f_VPD          = Dataset(VPD_file,  mode='r')
    f_SM           = Dataset(SMtop_file,mode='r')

    time           = f_Tmax.variables['time'][:]
    Tmax_mean_ctl  = f_Tmax.variables['Tmax_mean_ctl'][:]
    Tmax_mean_sen  = f_Tmax.variables['Tmax_mean_sen'][:]

    LAI_mean_ctl   = f_LAI.variables['LAI_mean_ctl'][:]
    LAI_std_ctl    = f_LAI.variables['LAI_std_ctl'][:]
    LAI_mean_sen   = f_LAI.variables['LAI_mean_sen'][:]
    LAI_std_sen    = f_LAI.variables['LAI_std_sen'][:]

    ALB_mean_ctl   = f_ALB.variables['Albedo_mean_ctl'][:]
    ALB_std_ctl    = f_ALB.variables['Albedo_std_ctl'][:]
    ALB_mean_sen   = f_ALB.variables['Albedo_mean_sen'][:]
    ALB_std_sen    = f_ALB.variables['Albedo_std_sen'][:]

    Qle_mean_ctl   = f_Qle.variables['Qle_mean_ctl'][:]
    Qle_std_ctl    = f_Qle.variables['Qle_std_ctl'][:]
    Qle_mean_sen   = f_Qle.variables['Qle_mean_sen'][:]
    Qle_std_sen    = f_Qle.variables['Qle_std_sen'][:]

    VPD_mean_ctl   = f_VPD.variables['VPD_mean_ctl'][:]
    VPD_std_ctl    = f_VPD.variables['VPD_std_ctl'][:]
    VPD_mean_sen   = f_VPD.variables['VPD_mean_sen'][:]
    VPD_std_sen    = f_VPD.variables['VPD_std_sen'][:]

    SMtop_mean_ctl = f_SM.variables['SMtop_mean_ctl'][:]
    SMtop_std_ctl  = f_SM.variables['SMtop_std_ctl'][:]
    SMtop_mean_sen = f_SM.variables['SMtop_mean_sen'][:]
    SMtop_std_sen  = f_SM.variables['SMtop_std_sen'][:]

    f_Tmax.close()
    f_LAI.close()
    f_ALB.close()
    f_Qle.close()
    f_VPD.close()
    f_SM.close()

    df_reg1                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen[0,:] - Tmax_mean_ctl[0,:]})
    df_reg1['LAI_mean_ctl']  = LAI_mean_ctl[0,:]
    df_reg1['LAI_mean_sen']  = LAI_mean_sen[0,:]
    df_reg1['ALB_mean_ctl']  = ALB_mean_ctl[0,:]
    df_reg1['ALB_mean_sen']  = ALB_mean_sen[0,:]
    df_reg1['Qle_mean_ctl']  = Qle_mean_ctl[0,:]
    df_reg1['Qle_mean_sen']  = Qle_mean_sen[0,:]
    df_reg1['SMtop_mean_ctl']= SMtop_mean_ctl[0,:]
    df_reg1['SMtop_mean_sen']= SMtop_mean_sen[0,:]
    df_reg1['LAI_low_ctl']   = LAI_mean_ctl[0,:] - LAI_std_ctl[0,:]
    df_reg1['LAI_high_ctl']  = LAI_mean_ctl[0,:] + LAI_std_ctl[0,:]
    df_reg1['LAI_low_sen']   = LAI_mean_sen[0,:] - LAI_std_sen[0,:]
    df_reg1['LAI_high_sen']  = LAI_mean_sen[0,:] + LAI_std_sen[0,:]
    df_reg1['ALB_low_ctl']   = ALB_mean_ctl[0,:] - ALB_std_ctl[0,:]
    df_reg1['ALB_high_ctl']  = ALB_mean_ctl[0,:] + ALB_std_ctl[0,:]
    df_reg1['ALB_low_sen']   = ALB_mean_sen[0,:] - ALB_std_sen[0,:]
    df_reg1['ALB_high_sen']  = ALB_mean_sen[0,:] + ALB_std_sen[0,:]
    df_reg1['Qle_low_ctl']   = Qle_mean_ctl[0,:] - Qle_std_ctl[0,:]
    df_reg1['Qle_high_ctl']  = Qle_mean_ctl[0,:] + Qle_std_ctl[0,:]
    df_reg1['Qle_low_sen']   = Qle_mean_sen[0,:] - Qle_std_sen[0,:]
    df_reg1['Qle_high_sen']  = Qle_mean_sen[0,:] + Qle_std_sen[0,:]
    df_reg1['SMtop_low_ctl'] = SMtop_mean_ctl[0,:] - SMtop_std_ctl[0,:]
    df_reg1['SMtop_high_ctl']= SMtop_mean_ctl[0,:] + SMtop_std_ctl[0,:]
    df_reg1['SMtop_low_sen'] = SMtop_mean_sen[0,:] - SMtop_std_sen[0,:]
    df_reg1['SMtop_high_sen']= SMtop_mean_sen[0,:] + SMtop_std_sen[0,:]
    df_reg1['Qle_diff']      = (Qle_mean_sen[0,:] - Qle_mean_ctl[0,:])
    df_reg1['VPD_diff']      = (VPD_mean_sen[0,:] - VPD_mean_ctl[0,:])*10.

    print("df_reg1", df_reg1)

    df_reg2                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen[1,:] - Tmax_mean_ctl[1,:]})
    df_reg2['LAI_mean_ctl']  = LAI_mean_ctl[1,:]
    df_reg2['LAI_mean_sen']  = LAI_mean_sen[1,:]
    df_reg2['ALB_mean_ctl']  = ALB_mean_ctl[1,:]
    df_reg2['ALB_mean_sen']  = ALB_mean_sen[1,:]
    df_reg2['Qle_mean_ctl']  = Qle_mean_ctl[1,:]
    df_reg2['Qle_mean_sen']  = Qle_mean_sen[1,:]
    df_reg2['SMtop_mean_ctl']= SMtop_mean_ctl[1,:]
    df_reg2['SMtop_mean_sen']= SMtop_mean_sen[1,:]
    df_reg2['LAI_low_ctl']   = LAI_mean_ctl[1,:] - LAI_std_ctl[1,:]
    df_reg2['LAI_high_ctl']  = LAI_mean_ctl[1,:] + LAI_std_ctl[1,:]
    df_reg2['LAI_low_sen']   = LAI_mean_sen[1,:] - LAI_std_sen[1,:]
    df_reg2['LAI_high_sen']  = LAI_mean_sen[1,:] + LAI_std_sen[1,:]
    df_reg2['ALB_low_ctl']   = ALB_mean_ctl[1,:] - ALB_std_ctl[1,:]
    df_reg2['ALB_high_ctl']  = ALB_mean_ctl[1,:] + ALB_std_ctl[1,:]
    df_reg2['ALB_low_sen']   = ALB_mean_sen[1,:] - ALB_std_sen[1,:]
    df_reg2['ALB_high_sen']  = ALB_mean_sen[1,:] + ALB_std_sen[1,:]
    df_reg2['Qle_low_ctl']   = Qle_mean_ctl[1,:] - Qle_std_ctl[1,:]
    df_reg2['Qle_high_ctl']  = Qle_mean_ctl[1,:] + Qle_std_ctl[1,:]
    df_reg2['Qle_low_sen']   = Qle_mean_sen[1,:] - Qle_std_sen[1,:]
    df_reg2['Qle_high_sen']  = Qle_mean_sen[1,:] + Qle_std_sen[1,:]
    df_reg2['SMtop_low_ctl'] = SMtop_mean_ctl[1,:] - SMtop_std_ctl[1,:]
    df_reg2['SMtop_high_ctl']= SMtop_mean_ctl[1,:] + SMtop_std_ctl[1,:]
    df_reg2['SMtop_low_sen'] = SMtop_mean_sen[1,:] - SMtop_std_sen[1,:]
    df_reg2['SMtop_high_sen']= SMtop_mean_sen[1,:] + SMtop_std_sen[1,:]
    df_reg2['Qle_diff']      = (Qle_mean_sen[1,:] - Qle_mean_ctl[1,:])
    df_reg2['VPD_diff']      = (VPD_mean_sen[1,:] - VPD_mean_ctl[1,:])*10.

    df_reg3                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen[2,:] - Tmax_mean_ctl[2,:]})
    df_reg3['LAI_mean_ctl']  = LAI_mean_ctl[2,:]
    df_reg3['LAI_mean_sen']  = LAI_mean_sen[2,:]
    df_reg3['ALB_mean_ctl']  = ALB_mean_ctl[2,:]
    df_reg3['ALB_mean_sen']  = ALB_mean_sen[2,:]
    df_reg3['Qle_mean_ctl']  = Qle_mean_ctl[2,:]
    df_reg3['Qle_mean_sen']  = Qle_mean_sen[2,:]
    df_reg3['SMtop_mean_ctl']= SMtop_mean_ctl[2,:]
    df_reg3['SMtop_mean_sen']= SMtop_mean_sen[2,:]
    df_reg3['LAI_low_ctl']   = LAI_mean_ctl[2,:] - LAI_std_ctl[2,:]
    df_reg3['LAI_high_ctl']  = LAI_mean_ctl[2,:] + LAI_std_ctl[2,:]
    df_reg3['LAI_low_sen']   = LAI_mean_sen[2,:] - LAI_std_sen[2,:]
    df_reg3['LAI_high_sen']  = LAI_mean_sen[2,:] + LAI_std_sen[2,:]
    df_reg3['ALB_low_ctl']   = ALB_mean_ctl[2,:] - ALB_std_ctl[2,:]
    df_reg3['ALB_high_ctl']  = ALB_mean_ctl[2,:] + ALB_std_ctl[2,:]
    df_reg3['ALB_low_sen']   = ALB_mean_sen[2,:] - ALB_std_sen[2,:]
    df_reg3['ALB_high_sen']  = ALB_mean_sen[2,:] + ALB_std_sen[2,:]
    df_reg3['Qle_low_ctl']   = Qle_mean_ctl[2,:] - Qle_std_ctl[2,:]
    df_reg3['Qle_high_ctl']  = Qle_mean_ctl[2,:] + Qle_std_ctl[2,:]
    df_reg3['Qle_low_sen']   = Qle_mean_sen[2,:] - Qle_std_sen[2,:]
    df_reg3['Qle_high_sen']  = Qle_mean_sen[2,:] + Qle_std_sen[2,:]
    df_reg3['SMtop_low_ctl'] = SMtop_mean_ctl[2,:] - SMtop_std_ctl[2,:]
    df_reg3['SMtop_high_ctl']= SMtop_mean_ctl[2,:] + SMtop_std_ctl[2,:]
    df_reg3['SMtop_low_sen'] = SMtop_mean_sen[2,:] - SMtop_std_sen[2,:]
    df_reg3['SMtop_high_sen']= SMtop_mean_sen[2,:] + SMtop_std_sen[2,:]
    df_reg3['Qle_diff']      = (Qle_mean_sen[2,:] - Qle_mean_ctl[2,:])
    df_reg3['VPD_diff']      = (VPD_mean_sen[2,:] - VPD_mean_ctl[2,:])*10.

    # ======================== Read unburnt ===============================
    f_Tmax_unburnt         = Dataset(Tmax_file_unburnt, mode='r')
    f_LAI_unburnt          = Dataset(LAI_file_unburnt,  mode='r')
    f_ALB_unburnt          = Dataset(ALB_file_unburnt,  mode='r')
    f_Qle_unburnt          = Dataset(Qle_file_unburnt,  mode='r')
    f_VPD_unburnt          = Dataset(VPD_file_unburnt,  mode='r')
    f_SM_unburnt           = Dataset(SMtop_file_unburnt,  mode='r')

    time                   = f_Tmax_unburnt.variables['time'][:]
    Tmax_mean_ctl_unburnt  = f_Tmax_unburnt.variables['Tmax_mean_ctl'][:]
    Tmax_mean_sen_unburnt  = f_Tmax_unburnt.variables['Tmax_mean_sen'][:]

    LAI_mean_ctl_unburnt   = f_LAI_unburnt.variables['LAI_mean_ctl'][:]
    LAI_std_ctl_unburnt    = f_LAI_unburnt.variables['LAI_std_ctl'][:]
    LAI_mean_sen_unburnt   = f_LAI_unburnt.variables['LAI_mean_sen'][:]
    LAI_std_sen_unburnt    = f_LAI_unburnt.variables['LAI_std_sen'][:]

    ALB_mean_ctl_unburnt   = f_ALB_unburnt.variables['Albedo_mean_ctl'][:]
    ALB_std_ctl_unburnt    = f_ALB_unburnt.variables['Albedo_std_ctl'][:]
    ALB_mean_sen_unburnt   = f_ALB_unburnt.variables['Albedo_mean_sen'][:]
    ALB_std_sen_unburnt    = f_ALB_unburnt.variables['Albedo_std_sen'][:]

    Qle_mean_ctl_unburnt   = f_Qle_unburnt.variables['Qle_mean_ctl'][:]
    Qle_std_ctl_unburnt    = f_Qle_unburnt.variables['Qle_std_ctl'][:]
    Qle_mean_sen_unburnt   = f_Qle_unburnt.variables['Qle_mean_sen'][:]
    Qle_std_sen_unburnt    = f_Qle_unburnt.variables['Qle_std_sen'][:]

    VPD_mean_ctl_unburnt   = f_VPD_unburnt.variables['VPD_mean_ctl'][:]
    VPD_std_ctl_unburnt    = f_VPD_unburnt.variables['VPD_std_ctl'][:]
    VPD_mean_sen_unburnt   = f_VPD_unburnt.variables['VPD_mean_sen'][:]
    VPD_std_sen_unburnt    = f_VPD_unburnt.variables['VPD_std_sen'][:]

    SMtop_mean_ctl_unburnt = f_SM_unburnt.variables['SMtop_mean_ctl'][:]
    SMtop_std_ctl_unburnt  = f_SM_unburnt.variables['SMtop_std_ctl'][:]
    SMtop_mean_sen_unburnt = f_SM_unburnt.variables['SMtop_mean_sen'][:]
    SMtop_std_sen_unburnt  = f_SM_unburnt.variables['SMtop_std_sen'][:]

    f_Tmax_unburnt.close()
    f_LAI_unburnt.close()
    f_ALB_unburnt.close()
    f_Qle_unburnt.close()
    f_VPD_unburnt.close()
    f_SM_unburnt.close()

    df_reg1_unburnt                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen_unburnt[0,:] - Tmax_mean_ctl_unburnt[0,:]})
    df_reg1_unburnt['LAI_mean_ctl']  = LAI_mean_ctl_unburnt[0,:]
    df_reg1_unburnt['LAI_mean_sen']  = LAI_mean_sen_unburnt[0,:]
    df_reg1_unburnt['ALB_mean_ctl']  = ALB_mean_ctl_unburnt[0,:]
    df_reg1_unburnt['ALB_mean_sen']  = ALB_mean_sen_unburnt[0,:]
    df_reg1_unburnt['Qle_mean_ctl']  = Qle_mean_ctl_unburnt[0,:]
    df_reg1_unburnt['Qle_mean_sen']  = Qle_mean_sen_unburnt[0,:]
    df_reg1_unburnt['SMtop_mean_ctl']  = SMtop_mean_ctl_unburnt[0,:]
    df_reg1_unburnt['SMtop_mean_sen']  = SMtop_mean_sen_unburnt[0,:]
    df_reg1_unburnt['LAI_low_ctl']   = LAI_mean_ctl_unburnt[0,:] - LAI_std_ctl_unburnt[0,:]
    df_reg1_unburnt['LAI_high_ctl']  = LAI_mean_ctl_unburnt[0,:] + LAI_std_ctl_unburnt[0,:]
    df_reg1_unburnt['LAI_low_sen']   = LAI_mean_sen_unburnt[0,:] - LAI_std_sen_unburnt[0,:]
    df_reg1_unburnt['LAI_high_sen']  = LAI_mean_sen_unburnt[0,:] + LAI_std_sen_unburnt[0,:]
    df_reg1_unburnt['ALB_low_ctl']   = ALB_mean_ctl_unburnt[0,:] - ALB_std_ctl_unburnt[0,:]
    df_reg1_unburnt['ALB_high_ctl']  = ALB_mean_ctl_unburnt[0,:] + ALB_std_ctl_unburnt[0,:]
    df_reg1_unburnt['ALB_low_sen']   = ALB_mean_sen_unburnt[0,:] - ALB_std_sen_unburnt[0,:]
    df_reg1_unburnt['ALB_high_sen']  = ALB_mean_sen_unburnt[0,:] + ALB_std_sen_unburnt[0,:]
    df_reg1_unburnt['Qle_low_ctl']   = Qle_mean_ctl_unburnt[0,:] - Qle_std_ctl_unburnt[0,:]
    df_reg1_unburnt['Qle_high_ctl']  = Qle_mean_ctl_unburnt[0,:] + Qle_std_ctl_unburnt[0,:]
    df_reg1_unburnt['Qle_low_sen']   = Qle_mean_sen_unburnt[0,:] - Qle_std_sen_unburnt[0,:]
    df_reg1_unburnt['Qle_high_sen']  = Qle_mean_sen_unburnt[0,:] + Qle_std_sen_unburnt[0,:]
    df_reg1_unburnt['SMtop_low_ctl'] = SMtop_mean_ctl_unburnt[0,:] - SMtop_std_ctl_unburnt[0,:]
    df_reg1_unburnt['SMtop_high_ctl']= SMtop_mean_ctl_unburnt[0,:] + SMtop_std_ctl_unburnt[0,:]
    df_reg1_unburnt['SMtop_low_sen'] = SMtop_mean_sen_unburnt[0,:] - SMtop_std_sen_unburnt[0,:]
    df_reg1_unburnt['SMtop_high_sen']= SMtop_mean_sen_unburnt[0,:] + SMtop_std_sen_unburnt[0,:]
    df_reg1_unburnt['Qle_diff']      = (Qle_mean_sen_unburnt[0,:] - Qle_mean_ctl_unburnt[0,:])
    df_reg1_unburnt['VPD_diff']      = (VPD_mean_sen_unburnt[0,:] - VPD_mean_ctl_unburnt[0,:])*10.

    print("df_reg1_unburnt", df_reg1_unburnt)

    df_reg2_unburnt                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen_unburnt[1,:] - Tmax_mean_ctl_unburnt[1,:]})
    df_reg2_unburnt['LAI_mean_ctl']  = LAI_mean_ctl_unburnt[1,:]
    df_reg2_unburnt['LAI_mean_sen']  = LAI_mean_sen_unburnt[1,:]
    df_reg2_unburnt['ALB_mean_ctl']  = ALB_mean_ctl_unburnt[1,:]
    df_reg2_unburnt['ALB_mean_sen']  = ALB_mean_sen_unburnt[1,:]
    df_reg2_unburnt['Qle_mean_ctl']  = Qle_mean_ctl_unburnt[1,:]
    df_reg2_unburnt['Qle_mean_sen']  = Qle_mean_sen_unburnt[1,:]
    df_reg2_unburnt['SMtop_mean_ctl']  = SMtop_mean_ctl_unburnt[1,:]
    df_reg2_unburnt['SMtop_mean_sen']  = SMtop_mean_sen_unburnt[1,:]
    df_reg2_unburnt['LAI_low_ctl']   = LAI_mean_ctl_unburnt[1,:] - LAI_std_ctl_unburnt[1,:]
    df_reg2_unburnt['LAI_high_ctl']  = LAI_mean_ctl_unburnt[1,:] + LAI_std_ctl_unburnt[1,:]
    df_reg2_unburnt['LAI_low_sen']   = LAI_mean_sen_unburnt[1,:] - LAI_std_sen_unburnt[1,:]
    df_reg2_unburnt['LAI_high_sen']  = LAI_mean_sen_unburnt[1,:] + LAI_std_sen_unburnt[1,:]
    df_reg2_unburnt['ALB_low_ctl']   = ALB_mean_ctl_unburnt[1,:] - ALB_std_ctl_unburnt[1,:]
    df_reg2_unburnt['ALB_high_ctl']  = ALB_mean_ctl_unburnt[1,:] + ALB_std_ctl_unburnt[1,:]
    df_reg2_unburnt['ALB_low_sen']   = ALB_mean_sen_unburnt[1,:] - ALB_std_sen_unburnt[1,:]
    df_reg2_unburnt['ALB_high_sen']  = ALB_mean_sen_unburnt[1,:] + ALB_std_sen_unburnt[1,:]
    df_reg2_unburnt['Qle_low_ctl']   = Qle_mean_ctl_unburnt[1,:] - Qle_std_ctl_unburnt[1,:]
    df_reg2_unburnt['Qle_high_ctl']  = Qle_mean_ctl_unburnt[1,:] + Qle_std_ctl_unburnt[1,:]
    df_reg2_unburnt['Qle_low_sen']   = Qle_mean_sen_unburnt[1,:] - Qle_std_sen_unburnt[1,:]
    df_reg2_unburnt['Qle_high_sen']  = Qle_mean_sen_unburnt[1,:] + Qle_std_sen_unburnt[1,:]
    df_reg2_unburnt['SMtop_low_ctl'] = SMtop_mean_ctl_unburnt[1,:] - SMtop_std_ctl_unburnt[1,:]
    df_reg2_unburnt['SMtop_high_ctl']= SMtop_mean_ctl_unburnt[1,:] + SMtop_std_ctl_unburnt[1,:]
    df_reg2_unburnt['SMtop_low_sen'] = SMtop_mean_sen_unburnt[1,:] - SMtop_std_sen_unburnt[1,:]
    df_reg2_unburnt['SMtop_high_sen']= SMtop_mean_sen_unburnt[1,:] + SMtop_std_sen_unburnt[1,:]
    df_reg2_unburnt['Qle_diff']      = (Qle_mean_sen_unburnt[1,:] - Qle_mean_ctl_unburnt[1,:])
    df_reg2_unburnt['VPD_diff']      = (VPD_mean_sen_unburnt[1,:] - VPD_mean_ctl_unburnt[1,:])*10.

    df_reg3_unburnt                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen_unburnt[2,:] - Tmax_mean_ctl_unburnt[2,:]})
    df_reg3_unburnt['LAI_mean_ctl']  = LAI_mean_ctl_unburnt[2,:]
    df_reg3_unburnt['LAI_mean_sen']  = LAI_mean_sen_unburnt[2,:]
    df_reg3_unburnt['ALB_mean_ctl']  = ALB_mean_ctl_unburnt[2,:]
    df_reg3_unburnt['ALB_mean_sen']  = ALB_mean_sen_unburnt[2,:]
    df_reg3_unburnt['Qle_mean_ctl']  = Qle_mean_ctl_unburnt[2,:]
    df_reg3_unburnt['Qle_mean_sen']  = Qle_mean_sen_unburnt[2,:]
    df_reg3_unburnt['SMtop_mean_ctl']  = SMtop_mean_ctl_unburnt[2,:]
    df_reg3_unburnt['SMtop_mean_sen']  = SMtop_mean_sen_unburnt[2,:]
    df_reg3_unburnt['LAI_low_ctl']   = LAI_mean_ctl_unburnt[2,:] - LAI_std_ctl_unburnt[2,:]
    df_reg3_unburnt['LAI_high_ctl']  = LAI_mean_ctl_unburnt[2,:] + LAI_std_ctl_unburnt[2,:]
    df_reg3_unburnt['LAI_low_sen']   = LAI_mean_sen_unburnt[2,:] - LAI_std_sen_unburnt[2,:]
    df_reg3_unburnt['LAI_high_sen']  = LAI_mean_sen_unburnt[2,:] + LAI_std_sen_unburnt[2,:]
    df_reg3_unburnt['ALB_low_ctl']   = ALB_mean_ctl_unburnt[2,:] - ALB_std_ctl_unburnt[2,:]
    df_reg3_unburnt['ALB_high_ctl']  = ALB_mean_ctl_unburnt[2,:] + ALB_std_ctl_unburnt[2,:]
    df_reg3_unburnt['ALB_low_sen']   = ALB_mean_sen_unburnt[2,:] - ALB_std_sen_unburnt[2,:]
    df_reg3_unburnt['ALB_high_sen']  = ALB_mean_sen_unburnt[2,:] + ALB_std_sen_unburnt[2,:]
    df_reg3_unburnt['Qle_low_ctl']   = Qle_mean_ctl_unburnt[2,:] - Qle_std_ctl_unburnt[2,:]
    df_reg3_unburnt['Qle_high_ctl']  = Qle_mean_ctl_unburnt[2,:] + Qle_std_ctl_unburnt[2,:]
    df_reg3_unburnt['Qle_low_sen']   = Qle_mean_sen_unburnt[2,:] - Qle_std_sen_unburnt[2,:]
    df_reg3_unburnt['Qle_high_sen']  = Qle_mean_sen_unburnt[2,:] + Qle_std_sen_unburnt[2,:]
    df_reg3_unburnt['SMtop_low_ctl'] = SMtop_mean_ctl_unburnt[2,:] - SMtop_std_ctl_unburnt[2,:]
    df_reg3_unburnt['SMtop_high_ctl']= SMtop_mean_ctl_unburnt[2,:] + SMtop_std_ctl_unburnt[2,:]
    df_reg3_unburnt['SMtop_low_sen'] = SMtop_mean_sen_unburnt[2,:] - SMtop_std_sen_unburnt[2,:]
    df_reg3_unburnt['SMtop_high_sen']= SMtop_mean_sen_unburnt[2,:] + SMtop_std_sen_unburnt[2,:]
    df_reg3_unburnt['Qle_diff']      = (Qle_mean_sen_unburnt[2,:] - Qle_mean_ctl_unburnt[2,:])
    df_reg3_unburnt['VPD_diff']      = (VPD_mean_sen_unburnt[2,:] - VPD_mean_ctl_unburnt[2,:])*10.

    if 1:
        # =================== Read in FMI index ===================
        f_FMI        = Dataset(FMI_file, mode='r')
        Time         = f_FMI.variables['time'][:]
        FMI_mean_ctl = f_FMI.variables['FMI_ctl_mean'][:]
        FMI_mean_sen = f_FMI.variables['FMI_sen_mean'][:]
        FMI_std_ctl  = f_FMI.variables['FMI_ctl_std'][:]
        FMI_std_sen  = f_FMI.variables['FMI_sen_std'][:]
        f_FMI.close()

        FMI_reg                  = pd.DataFrame({'FMI_diff_reg1': FMI_mean_sen[0,:] - FMI_mean_ctl[0,:]})
        FMI_reg['FMI_diff_reg2'] = FMI_mean_sen[1,:] - FMI_mean_ctl[1,:]
        FMI_reg['FMI_diff_reg3'] = FMI_mean_sen[2,:] - FMI_mean_ctl[2,:]

        # =================== Read unburnt FFDI index ===================
        f_FMI_unburnt        = Dataset(FMI_file_unburnt, mode='r')
        Time                 = f_FMI_unburnt.variables['time'][:]
        FMI_mean_ctl_unburnt = f_FMI_unburnt.variables['FMI_ctl_mean'][:]
        FMI_mean_sen_unburnt = f_FMI_unburnt.variables['FMI_sen_mean'][:]
        FMI_std_ctl_unburnt  = f_FMI_unburnt.variables['FMI_ctl_std'][:]
        FMI_std_sen_unburnt  = f_FMI_unburnt.variables['FMI_sen_std'][:]
        f_FMI_unburnt.close()

        FMI_reg_unburnt                  = pd.DataFrame({'FMI_diff_reg1': FMI_mean_sen_unburnt[0,:] - FMI_mean_ctl_unburnt[0,:]})
        FMI_reg_unburnt['FMI_diff_reg2'] = FMI_mean_sen_unburnt[1,:] - FMI_mean_ctl_unburnt[1,:]
        FMI_reg_unburnt['FMI_diff_reg3'] = FMI_mean_sen_unburnt[2,:] - FMI_mean_ctl_unburnt[2,:]

        ntime_FMI      = np.shape(FMI_mean_ctl)[0]
        time_steps_FMI = np.arange(92,92+len(Time),1)

        if daytime_FMI_file != None and daytime_FMI_file_unburnt != None:
            # =================== Read in daytime FMI index ===================
            f_daytime_FMI        = Dataset(daytime_FMI_file, mode='r')
            daytime_Time         = f_daytime_FMI.variables['time'][:]
            daytime_FMI_mean_ctl = f_daytime_FMI.variables['FMI_ctl_mean'][:]
            daytime_FMI_mean_sen = f_daytime_FMI.variables['FMI_sen_mean'][:]
            daytime_FMI_std_ctl  = f_daytime_FMI.variables['FMI_ctl_std'][:]
            daytime_FMI_std_sen  = f_daytime_FMI.variables['FMI_sen_std'][:]
            f_daytime_FMI.close()

            daytime_FMI_reg                  = pd.DataFrame({'FMI_diff_reg1': daytime_FMI_mean_sen[0,:] - daytime_FMI_mean_ctl[0,:]})
            daytime_FMI_reg['FMI_diff_reg2'] = daytime_FMI_mean_sen[1,:] - daytime_FMI_mean_ctl[1,:]
            daytime_FMI_reg['FMI_diff_reg3'] = daytime_FMI_mean_sen[2,:] - daytime_FMI_mean_ctl[2,:]

            # =================== Read unburnt FFDI index ===================
            f_daytime_FMI_unburnt        = Dataset(daytime_FMI_file_unburnt, mode='r')
            daytime_Time                 = f_daytime_FMI_unburnt.variables['time'][:]
            daytime_FMI_mean_ctl_unburnt = f_daytime_FMI_unburnt.variables['FMI_ctl_mean'][:]
            daytime_FMI_mean_sen_unburnt = f_daytime_FMI_unburnt.variables['FMI_sen_mean'][:]
            daytime_FMI_std_ctl_unburnt  = f_daytime_FMI_unburnt.variables['FMI_ctl_std'][:]
            daytime_FMI_std_sen_unburnt  = f_daytime_FMI_unburnt.variables['FMI_sen_std'][:]
            f_daytime_FMI_unburnt.close()

            daytime_FMI_reg_unburnt                  = pd.DataFrame({'FMI_diff_reg1': daytime_FMI_mean_sen_unburnt[0,:] - daytime_FMI_mean_ctl_unburnt[0,:]})
            daytime_FMI_reg_unburnt['FMI_diff_reg2'] = daytime_FMI_mean_sen_unburnt[1,:] - daytime_FMI_mean_ctl_unburnt[1,:]
            daytime_FMI_reg_unburnt['FMI_diff_reg3'] = daytime_FMI_mean_sen_unburnt[2,:] - daytime_FMI_mean_ctl_unburnt[2,:]
            time_steps_daytime_FMI = np.arange(1,1+len(daytime_Time),1)

    if 1:
        # =========== Fire date ===========
        fire_file         = Dataset(fire_path, mode='r')
        Burn_Date_tmp     = fire_file.variables['Burn_Date'][2:8,::-1,:]  # 2019-09 - 2020-02
        lat_fire          = fire_file.variables['lat'][::-1]
        lon_fire          = fire_file.variables['lon'][:]

        Burn_Date         = Burn_Date_tmp.astype(float)
        Burn_Date         = np.where(Burn_Date<=0, 99999, Burn_Date)

        Burn_Date[4:,:,:] = Burn_Date[4:,:,:]+365 # Add 365 to Jan-Feb 2020

        Burn_Date_min     = np.nanmin(Burn_Date, axis=0)

        Burn_Date_min     = np.where(Burn_Date_min>=99999, np.nan, Burn_Date_min)
        Burn_Date_min     = Burn_Date_min - 243 # start from Sep 2019

        lons_2D, lats_2D = np.meshgrid(lon_fire, lat_fire)

        mask_val         = np.zeros((3,np.shape(lons_2D)[0],np.shape(lons_2D)[1]),dtype=bool)

        for i in np.arange(3):
            mask_val[i,:,:]  = np.all(( lats_2D>loc_lats[i][0],lats_2D<loc_lats[i][1],
                                        lons_2D>loc_lons[i][0],lons_2D<loc_lons[i][1]), axis=0)

        Burn_Date_min_reg1 = np.where( mask_val[0,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg2 = np.where( mask_val[1,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg3 = np.where( mask_val[2,:,:], Burn_Date_min, np.nan)

        Burn_reg1_10th = np.nanpercentile(Burn_Date_min_reg1, 10)
        Burn_reg1_50th = np.nanpercentile(Burn_Date_min_reg1, 50)
        Burn_reg1_90th = np.nanpercentile(Burn_Date_min_reg1, 90)

        Burn_reg2_10th = np.nanpercentile(Burn_Date_min_reg2, 10)
        Burn_reg2_50th = np.nanpercentile(Burn_Date_min_reg2, 50)
        Burn_reg2_90th = np.nanpercentile(Burn_Date_min_reg2, 90)

        Burn_reg3_10th = np.nanpercentile(Burn_Date_min_reg3, 10)
        Burn_reg3_50th = np.nanpercentile(Burn_Date_min_reg3, 50)
        Burn_reg3_90th = np.nanpercentile(Burn_Date_min_reg3, 90)

        print('Burn_reg1_10th',Burn_reg1_10th)
        print('Burn_reg1_50th',Burn_reg1_50th)
        print('Burn_reg1_90th',Burn_reg1_90th)
        print('Burn_reg2_10th',Burn_reg2_10th)
        print('Burn_reg2_50th',Burn_reg2_50th)
        print('Burn_reg2_90th',Burn_reg2_90th)
        print('Burn_reg3_10th',Burn_reg3_10th)
        print('Burn_reg3_50th',Burn_reg3_50th)
        print('Burn_reg3_90th',Burn_reg3_90th)


    cleaner_dates = ["Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019", "Jan 2020", "Feb 2020",       ""]
    xtickslocs    = [         0,         30,         61,         91,       122,         153,     182 ]

    # ===================== Plotting =====================
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=[11,7], sharex=False,
                sharey=False, squeeze=True)
    # fig, axs = plt.subplots(nrows=3, ncols=3, figsize=[10,8], sharex=False,
    #             sharey=False, squeeze=True)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 12
    plt.rcParams['font.size']       = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    almost_black                    = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    time_steps = np.arange(len(time))

    # Tmax & FMI
    row = 0
    axs[row,0].axhline(y=0, color='grey', linestyle='--')
    axs[row,1].axhline(y=0, color='grey', linestyle='--')
    axs[row,2].axhline(y=0, color='grey', linestyle='--')

    axs[row,0].plot(df_reg1['Tmax_diff'].rolling(window=5).mean(), c = almost_black, ls='-', lw=1., alpha=1)
    axs[row,1].plot(df_reg2['Tmax_diff'].rolling(window=5).mean(), c = almost_black, ls='-', lw=1., alpha=1)
    axs[row,2].plot(df_reg3['Tmax_diff'].rolling(window=5).mean(), c = almost_black, ls='-', lw=1., alpha=1)

    # axs[row,0].plot(df_reg1_unburnt['Tmax_diff'].rolling(window=5).mean(), c = almost_black, ls=':', lw=1., alpha=1)
    # axs[row,1].plot(df_reg2_unburnt['Tmax_diff'].rolling(window=5).mean(), c = almost_black, ls=':', lw=1., alpha=1)
    # axs[row,2].plot(df_reg3_unburnt['Tmax_diff'].rolling(window=5).mean(), c = almost_black, ls=':', lw=1., alpha=1)

    # Plot FMI
    if 1:
        axs[row,0].plot(time_steps_FMI, FMI_reg['FMI_diff_reg1'].rolling(window=5).mean(), c = 'red', ls='-', lw=1., alpha=1)
        axs[row,1].plot(time_steps_FMI, FMI_reg['FMI_diff_reg2'].rolling(window=5).mean(), c = 'red', ls='-', lw=1., alpha=1)
        axs[row,2].plot(time_steps_FMI, FMI_reg['FMI_diff_reg3'].rolling(window=5).mean(), c = 'red', ls='-', lw=1., alpha=1)

        # axs[row,0].plot(time_steps_FMI, FMI_reg_unburnt['FMI_diff_reg1'].rolling(window=5).mean(), c = 'red', ls=':', lw=1., alpha=1)
        # axs[row,1].plot(time_steps_FMI, FMI_reg_unburnt['FMI_diff_reg2'].rolling(window=5).mean(), c = 'red', ls=':', lw=1., alpha=1)
        # axs[row,2].plot(time_steps_FMI, FMI_reg_unburnt['FMI_diff_reg3'].rolling(window=5).mean(), c = 'red', ls=':', lw=1., alpha=1)

        if daytime_FMI_file != None and daytime_FMI_file_unburnt != None:
            axs[row,0].plot(time_steps_daytime_FMI, daytime_FMI_reg['FMI_diff_reg1'].rolling(window=5).mean(), c = 'firebrick', ls='-', lw=1., alpha=1)
            axs[row,1].plot(time_steps_daytime_FMI, daytime_FMI_reg['FMI_diff_reg2'].rolling(window=5).mean(), c = 'firebrick', ls='-', lw=1., alpha=1)
            axs[row,2].plot(time_steps_daytime_FMI, daytime_FMI_reg['FMI_diff_reg3'].rolling(window=5).mean(), c = 'firebrick', ls='-', lw=1., alpha=1)

            # axs[row,0].plot(time_steps_daytime_FMI, daytime_FMI_reg_unburnt['FMI_diff_reg1'].rolling(window=5).mean(), c = 'firebrick', ls=':', lw=1., alpha=1)
            # axs[row,1].plot(time_steps_daytime_FMI, daytime_FMI_reg_unburnt['FMI_diff_reg2'].rolling(window=5).mean(), c = 'firebrick', ls=':', lw=1., alpha=1)
            # axs[row,2].plot(time_steps_daytime_FMI, daytime_FMI_reg_unburnt['FMI_diff_reg3'].rolling(window=5).mean(), c = 'firebrick', ls=':', lw=1., alpha=1)

    # Plot VPD
    if 0:
        axs[row,0].plot( df_reg1['VPD_diff'].rolling(window=5).mean(), c = 'red', ls='-', lw=1., alpha=1)
        axs[row,1].plot( df_reg2['VPD_diff'].rolling(window=5).mean(), c = 'red', ls='-', lw=1., alpha=1)
        axs[row,2].plot( df_reg3['VPD_diff'].rolling(window=5).mean(), c = 'red', ls='-', lw=1., alpha=1)

        axs[row,0].plot( df_reg1_unburnt['VPD_diff'].rolling(window=5).mean(), c = 'red', ls=':', lw=1., alpha=1)
        axs[row,1].plot( df_reg2_unburnt['VPD_diff'].rolling(window=5).mean(), c = 'red', ls=':', lw=1., alpha=1)
        axs[row,2].plot( df_reg3_unburnt['VPD_diff'].rolling(window=5).mean(), c = 'red', ls=':', lw=1., alpha=1)

    # LAI
    row = 1
    axs[row,0].fill_between(time_steps, df_reg1['LAI_low_ctl'].rolling(window=5).mean(), df_reg1['LAI_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,0].fill_between(time_steps, df_reg1['LAI_low_sen'].rolling(window=5).mean(), df_reg1['LAI_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)

    axs[row,0].plot(df_reg1['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
    axs[row,0].plot(df_reg1['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls='-', lw=1, alpha=1)

    # axs[row,0].plot(df_reg1_unburnt['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
    # axs[row,0].plot(df_reg1_unburnt['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls=':', lw=1, alpha=1)

    axs[row,1].fill_between(time_steps, df_reg2['LAI_low_ctl'].rolling(window=5).mean(), df_reg2['LAI_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,1].fill_between(time_steps, df_reg2['LAI_low_sen'].rolling(window=5).mean(), df_reg2['LAI_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)

    axs[row,1].plot(df_reg2['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
    axs[row,1].plot(df_reg2['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls='-', lw=1, alpha=1)

    # axs[row,1].plot(df_reg2_unburnt['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
    # axs[row,1].plot(df_reg2_unburnt['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls=':', lw=1, alpha=1)

    axs[row,2].fill_between(time_steps, df_reg3['LAI_low_ctl'].rolling(window=5).mean(), df_reg3['LAI_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,2].fill_between(time_steps, df_reg3['LAI_low_sen'].rolling(window=5).mean(), df_reg3['LAI_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)

    axs[row,2].plot(df_reg3['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
    axs[row,2].plot(df_reg3['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls='-', lw=1, alpha=1)

    # axs[row,2].plot(df_reg3_unburnt['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
    # axs[row,2].plot(df_reg3_unburnt['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls=':', lw=1, alpha=1)

    # ALB
    row = 2
    axs[row,0].fill_between(time_steps, df_reg1['ALB_low_ctl'].rolling(window=5).mean(), df_reg1['ALB_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,0].fill_between(time_steps, df_reg1['ALB_low_sen'].rolling(window=5).mean(), df_reg1['ALB_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)

    axs[row,0].plot(df_reg1['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
    axs[row,0].plot(df_reg1['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls='-', lw=1, alpha=1)

    # axs[row,0].plot(df_reg1_unburnt['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
    # axs[row,0].plot(df_reg1_unburnt['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls=':', lw=1, alpha=1)

    axs[row,1].fill_between(time_steps, df_reg2['ALB_low_ctl'].rolling(window=5).mean(), df_reg2['ALB_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,1].fill_between(time_steps, df_reg2['ALB_low_sen'].rolling(window=5).mean(), df_reg2['ALB_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)

    axs[row,1].plot(df_reg2['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
    axs[row,1].plot(df_reg2['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls='-', lw=1, alpha=1)

    # axs[row,1].plot(df_reg2_unburnt['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
    # axs[row,1].plot(df_reg2_unburnt['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls=':', lw=1, alpha=1)

    axs[row,2].fill_between(time_steps, df_reg3['ALB_low_ctl'].rolling(window=5).mean(), df_reg3['ALB_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,2].fill_between(time_steps, df_reg3['ALB_low_sen'].rolling(window=5).mean(), df_reg3['ALB_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)

    axs[row,2].plot(df_reg3['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
    axs[row,2].plot(df_reg3['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls='-', lw=1, alpha=1)

    # axs[row,2].plot(df_reg3_unburnt['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
    # axs[row,2].plot(df_reg3_unburnt['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls=':', lw=1, alpha=1)

    if 0:
        # accumulated ET
        row = 3
        wm2_mmd = 86400/2454000
        # axs[row,0].fill_between(time_steps, df_reg1['Qle_low_ctl'].rolling(window=5).mean(), df_reg1['Qle_high_ctl'].rolling(window=5).mean(),
        #                 color="green", edgecolor="none", alpha=0.2)
        # axs[row,0].fill_between(time_steps, df_reg1['Qle_low_sen'].rolling(window=5).mean(), df_reg1['Qle_high_sen'].rolling(window=5).mean(),
        #                 color="orange", edgecolor="none", alpha=0.35)

        axs[row,0].plot(df_reg1['Qle_mean_ctl'].cumsum()*wm2_mmd, label="ctl", c = "green", ls='-', lw=1, alpha=1)
        axs[row,0].plot(df_reg1['Qle_mean_sen'].cumsum()*wm2_mmd, label="exp", c = "orange", ls='-', lw=1, alpha=1)

        # axs[row,0].plot(df_reg1_unburnt['Qle_mean_ctl'].cumsum()*wm2_mmd, label="ctl", c = "green", ls=':', lw=1, alpha=1)
        # axs[row,0].plot(df_reg1_unburnt['Qle_mean_sen'].cumsum()*wm2_mmd, label="exp", c = "orange", ls=':', lw=1, alpha=1)

        # axs[row,1].fill_between(time_steps, df_reg2['Qle_low_ctl'].rolling(window=5).mean(), df_reg2['Qle_high_ctl'].rolling(window=5).mean(),
        #                 color="green", edgecolor="none", alpha=0.2)
        # axs[row,1].fill_between(time_steps, df_reg2['Qle_low_sen'].rolling(window=5).mean(), df_reg2['Qle_high_sen'].rolling(window=5).mean(),
        #                 color="orange", edgecolor="none", alpha=0.35)

        axs[row,1].plot(df_reg2['Qle_mean_ctl'].cumsum()*wm2_mmd, label="ctl", c = "green", ls='-', lw=1, alpha=1)
        axs[row,1].plot(df_reg2['Qle_mean_sen'].cumsum()*wm2_mmd, label="exp", c = "orange", ls='-', lw=1, alpha=1)

        # axs[row,1].plot(df_reg2_unburnt['Qle_mean_ctl'].cumsum()*wm2_mmd, label="ctl", c = "green", ls=':', lw=1, alpha=1)
        # axs[row,1].plot(df_reg2_unburnt['Qle_mean_sen'].cumsum()*wm2_mmd, label="exp", c = "orange", ls=':', lw=1, alpha=1)

        # axs[row,2].fill_between(time_steps, df_reg3['Qle_low_ctl'].rolling(window=5).mean(), df_reg3['Qle_high_ctl'].rolling(window=5).mean(),
        #                 color="green", edgecolor="none", alpha=0.2)
        # axs[row,2].fill_between(time_steps, df_reg3['Qle_low_sen'].rolling(window=5).mean(), df_reg3['Qle_high_sen'].rolling(window=5).mean(),
        #                 color="orange", edgecolor="none", alpha=0.35)

        axs[row,2].plot(df_reg3['Qle_mean_ctl'].cumsum()*wm2_mmd, label="ctl", c = "green", ls='-', lw=1, alpha=1)
        axs[row,2].plot(df_reg3['Qle_mean_sen'].cumsum()*wm2_mmd, label="exp", c = "orange", ls='-', lw=1, alpha=1)

        # axs[row,2].plot(df_reg3_unburnt['Qle_mean_ctl'].cumsum()*wm2_mmd, label="ctl", c = "green", ls=':', lw=1, alpha=1)
        # axs[row,2].plot(df_reg3_unburnt['Qle_mean_sen'].cumsum()*wm2_mmd, label="exp", c = "orange", ls=':', lw=1, alpha=1)

    if 0:
        # SM top 50cm
        row = 4
        axs[row,0].fill_between(time_steps, df_reg1['SMtop_low_ctl'].rolling(window=5).mean(), df_reg1['SMtop_high_ctl'].rolling(window=5).mean(),
                        color="green", edgecolor="none", alpha=0.2)
        axs[row,0].fill_between(time_steps, df_reg1['SMtop_low_sen'].rolling(window=5).mean(), df_reg1['SMtop_high_sen'].rolling(window=5).mean(),
                        color="orange", edgecolor="none", alpha=0.35)

        axs[row,0].plot(df_reg1['SMtop_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
        axs[row,0].plot(df_reg1['SMtop_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls='-', lw=1, alpha=1)

        axs[row,0].plot(df_reg1_unburnt['SMtop_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
        axs[row,0].plot(df_reg1_unburnt['SMtop_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", ls=':', lw=1, alpha=1)

        axs[row,1].fill_between(time_steps, df_reg2['SMtop_low_ctl'].rolling(window=5).mean(), df_reg2['SMtop_high_ctl'].rolling(window=5).mean(),
                        color="green", edgecolor="none", alpha=0.2)
        axs[row,1].fill_between(time_steps, df_reg2['SMtop_low_sen'].rolling(window=5).mean(), df_reg2['SMtop_high_sen'].rolling(window=5).mean(),
                        color="orange", edgecolor="none", alpha=0.35)

        axs[row,1].plot(df_reg2['SMtop_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
        axs[row,1].plot(df_reg2['SMtop_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls='-', lw=1, alpha=1)

        axs[row,1].plot(df_reg2_unburnt['SMtop_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
        axs[row,1].plot(df_reg2_unburnt['SMtop_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls=':', lw=1, alpha=1)

        axs[row,2].fill_between(time_steps, df_reg3['SMtop_low_ctl'].rolling(window=5).mean(), df_reg3['SMtop_high_ctl'].rolling(window=5).mean(),
                        color="green", edgecolor="none", alpha=0.2)
        axs[row,2].fill_between(time_steps, df_reg3['SMtop_low_sen'].rolling(window=5).mean(), df_reg3['SMtop_high_sen'].rolling(window=5).mean(),
                        color="orange", edgecolor="none", alpha=0.35)

        axs[row,2].plot(df_reg3['SMtop_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls='-', lw=1, alpha=1)
        axs[row,2].plot(df_reg3['SMtop_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls='-', lw=1, alpha=1)

        axs[row,2].plot(df_reg3_unburnt['SMtop_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", ls=':', lw=1, alpha=1)
        axs[row,2].plot(df_reg3_unburnt['SMtop_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange",ls=':', lw=1, alpha=1)

        # delta Qle
        ax1= axs[row,0].twinx()
        ax2= axs[row,1].twinx()
        ax3= axs[row,2].twinx()

        ax1.plot(df_reg1['Qle_diff'].rolling(window=5).mean(), c = almost_black, ls='-', lw=1., alpha=1)
        ax2.plot(df_reg2['Qle_diff'].rolling(window=5).mean(), c = almost_black, ls='-', lw=1., alpha=1)
        ax3.plot(df_reg3['Qle_diff'].rolling(window=5).mean(), c = almost_black, ls='-', lw=1., alpha=1)

        ax1.plot(df_reg1_unburnt['Qle_diff'].rolling(window=5).mean(), c = almost_black, ls=':', lw=1., alpha=1)
        ax2.plot(df_reg2_unburnt['Qle_diff'].rolling(window=5).mean(), c = almost_black, ls=':', lw=1., alpha=1)
        ax3.plot(df_reg3_unburnt['Qle_diff'].rolling(window=5).mean(), c = almost_black, ls=':', lw=1., alpha=1)

        # Set the y-axis limits for the secondary y-axis
        ax1.set_ylim(-35, 8)
        ax2.set_ylim(-35, 8)
        ax3.set_ylim(-35, 8)
        ax1.set_yticklabels([])#,labelright=True)
        ax2.set_yticklabels([])#,labelright=True)

        ax3.set_yticks([-35,-30,-25,-20,-15,-10,-5,0,5])#,labelright=True)
        ax3.set_yticklabels(['-35','-30','-25','-20','-15','-10','-5','0','5'], fontsize=10)#,labelright=True)
        ax3.set_ylabel("ΔLH (W m$\mathregular{^{-2}}$)", fontsize=12)

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.spines['top'].set_visible(False)

        ax1.spines['left'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax3.spines['left'].set_visible(False)

        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)


    # for i in np.arange(5):
    for i in np.arange(3):
        axs[i,0].axvline(x=Burn_reg1_10th, color='coral', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,0].axvline(x=Burn_reg1_50th, color='red', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,0].axvline(x=Burn_reg1_90th, color='brown', linestyle='-',linewidth=1.5, alpha=0.8)

        axs[i,1].axvline(x=Burn_reg2_10th, color='coral', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,1].axvline(x=Burn_reg2_50th, color='red', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,1].axvline(x=Burn_reg2_90th, color='brown', linestyle='-',linewidth=1.5, alpha=0.8)

        axs[i,2].axvline(x=Burn_reg3_10th, color='coral', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,2].axvline(x=Burn_reg3_50th, color='red', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,2].axvline(x=Burn_reg3_90th, color='brown', linestyle='-',linewidth=1.5, alpha=0.8)

    Tmax_bot_val   = -2.8
    Tmax_up_val    = 1.1
    Tmax_levels    = [-2,-1,0,1]
    Tmax_labels    = ['-2','-1','0','1']

    #Tmax_bot_val   = -50
    #Tmax_up_val    = 50.
    #Tmax_levels    = [-50,-40,-30,-20,-10,0,10,20,30,40,50]
    #Tmax_labels    = ['-50','-40','-30','-20','-10','0','10','20','30','40','50']

    LAI_bot_val    = 0
    LAI_up_val     = 6
    LAI_levels     = [0,2,4,6]
    LAI_labels     = ['0','2','4','6']

    Albedo_bot_val = 0.06
    Albedo_up_val  = 0.17
    Albedo_levels  = [0.06,0.08,0.10,0.12,0.14,0.16]
    Albedo_labels  = ['0.06','0.08','0.10','0.12','0.14','0.16']

    if 1:
        # acculumate ET
        Qle_bot_val    = 0
        Qle_up_val     = 370
        Qle_levels     = [0,50,100,150,200,250,300,350]
        Qle_labels     = ['0','50','100','150','200','250','300','350']
    if 1:
        SM_bot_val     = 0.1
        SM_up_val      = 0.34
        SM_levels      = [0.1,0.2,0.3]
        SM_labels      = ['0.1','0.2','0.3']

    for j in np.arange(3):
        axs[0,j].set_ylim(Tmax_bot_val,Tmax_up_val)
        axs[1,j].set_ylim(LAI_bot_val,LAI_up_val)
        axs[2,j].set_ylim(Albedo_bot_val,Albedo_up_val)
        # axs[3,j].set_ylim(Qle_bot_val,Qle_up_val)
        # axs[4,j].set_ylim(SM_bot_val,SM_up_val)
        # for i in np.arange(5):
        for i in np.arange(3):
            # if i == 4:
            if i == 2:
                axs[i,j].set_xticks(xtickslocs)
                axs[i,j].set_xticklabels(cleaner_dates,rotation=25)
            else:
                axs[i,j].set_xticks(xtickslocs)
                axs[i,j].set_xticklabels([],rotation=25)

            axs[i,j].set_xlim(0,152)

        if j == 0:
            axs[0,j].set_yticks(Tmax_levels)
            axs[0,j].set_yticklabels(Tmax_labels)

            axs[1,j].set_yticks(LAI_levels)
            axs[1,j].set_yticklabels(LAI_labels)

            axs[2,j].set_yticks(Albedo_levels)
            axs[2,j].set_yticklabels(Albedo_labels)

            # axs[3,j].set_yticks(Qle_levels)
            # axs[3,j].set_yticklabels(Qle_labels)

            # axs[4,j].set_yticks(SM_levels)
            # axs[4,j].set_yticklabels(SM_labels)
        else:
            axs[0,j].set_yticks(Tmax_levels)
            axs[0,j].set_yticklabels([])

            axs[1,j].set_yticks(LAI_levels)
            axs[1,j].set_yticklabels([])

            axs[2,j].set_yticks(Albedo_levels)
            axs[2,j].set_yticklabels([])

            # axs[3,j].set_yticks(Qle_levels)
            # axs[3,j].set_yticklabels([])

            # axs[4,j].set_yticks(SM_levels)
            # axs[4,j].set_yticklabels([])

    # Set top titles
    axs[0,0].set_title("North")
    axs[0,1].set_title("Central")
    axs[0,2].set_title("South")

    # Set left labels
    axs[0,0].set_ylabel("ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C), ΔFMI (-)", fontsize=12)
    # axs[0,0].set_ylabel("ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C), ΔVPD (hPa)", fontsize=12)
    axs[1,0].set_ylabel("LAI (m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$)", fontsize=12)
    axs[2,0].set_ylabel("$α$ (-)", fontsize=12)
    # axs[3,0].set_ylabel("ET$\mathregular{_{accu}}$ (mm)", fontsize=12)
    # axs[4,0].set_ylabel("θ$\mathregular{_{0.5m}}$ (m$\mathregular{^{3}}$ m$\mathregular{^{-3}}$)", fontsize=12)

    axs[0,0].text(0.02, 0.95, "(e)", transform=axs[0,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,1].text(0.02, 0.95, "(f)", transform=axs[0,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,2].text(0.02, 0.95, "(g)", transform=axs[0,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,0].text(0.02, 0.95, "(h)", transform=axs[1,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,1].text(0.02, 0.95, "(i)", transform=axs[1,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,2].text(0.02, 0.95, "(j)", transform=axs[1,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,0].text(0.02, 0.95, "(k)", transform=axs[2,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,1].text(0.02, 0.95, "(l)", transform=axs[2,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,2].text(0.02, 0.95, "(m)", transform=axs[2,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[3,0].text(0.02, 0.95, "(j)", transform=axs[3,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[3,1].text(0.02, 0.95, "(k)", transform=axs[3,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[3,2].text(0.02, 0.95, "(l)", transform=axs[3,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[4,0].text(0.02, 0.95, "(m)", transform=axs[4,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[4,1].text(0.02, 0.95, "(n)", transform=axs[4,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # axs[4,2].text(0.02, 0.95, "(o)", transform=axs[4,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # ax.legend()
    # fig.tight_layout()

    # plt.savefig('./plots/burnt_unburnt_reg_time_series',dpi=300)

    plt.savefig('./plots/Fig6_burnt_reg_time_series',dpi=300)

def plot_time_series_burn_region(FMI_file, Tmax_file, LAI_file, ALB_file, Qle_file, loc_lats=None, loc_lons=None, time_s=None, time_e=None):
    # Check whether mask right

    f_Tmax         = Dataset(Tmax_file, mode='r')
    f_LAI          = Dataset(LAI_file,  mode='r')
    f_ALB          = Dataset(ALB_file,  mode='r')
    f_Qle          = Dataset(Qle_file,  mode='r')

    time           = f_Tmax.variables['time'][:]
    Tmax_mean_ctl  = f_Tmax.variables['Tmax_mean_ctl'][:]
    Tmax_mean_sen  = f_Tmax.variables['Tmax_mean_sen'][:]

    LAI_mean_ctl   = f_LAI.variables['LAI_mean_ctl'][:]
    LAI_std_ctl    = f_LAI.variables['LAI_std_ctl'][:]
    LAI_mean_sen   = f_LAI.variables['LAI_mean_sen'][:]
    LAI_std_sen    = f_LAI.variables['LAI_std_sen'][:]

    ALB_mean_ctl   = f_ALB.variables['Albedo_mean_ctl'][:]
    ALB_std_ctl    = f_ALB.variables['Albedo_std_ctl'][:]
    ALB_mean_sen   = f_ALB.variables['Albedo_mean_sen'][:]
    ALB_std_sen    = f_ALB.variables['Albedo_std_sen'][:]

    Qle_mean_ctl   = f_Qle.variables['Qle_mean_ctl'][:]
    Qle_std_ctl    = f_Qle.variables['Qle_std_ctl'][:]
    Qle_mean_sen   = f_Qle.variables['Qle_mean_sen'][:]
    Qle_std_sen    = f_Qle.variables['Qle_std_sen'][:]

    f_Tmax.close()
    f_LAI.close()
    f_ALB.close()
    f_Qle.close()


    df_reg1                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen[0,:] - Tmax_mean_ctl[0,:]})
    df_reg1['LAI_mean_ctl']  = LAI_mean_ctl[0,:]
    df_reg1['LAI_mean_sen']  = LAI_mean_sen[0,:]
    df_reg1['ALB_mean_ctl']  = ALB_mean_ctl[0,:]
    df_reg1['ALB_mean_sen']  = ALB_mean_sen[0,:]
    df_reg1['Qle_mean_ctl']  = Qle_mean_ctl[0,:]
    df_reg1['Qle_mean_sen']  = Qle_mean_sen[0,:]
    df_reg1['LAI_low_ctl']   = LAI_mean_ctl[0,:] - LAI_std_ctl[0,:]
    df_reg1['LAI_high_ctl']  = LAI_mean_ctl[0,:] + LAI_std_ctl[0,:]
    df_reg1['LAI_low_sen']   = LAI_mean_sen[0,:] - LAI_std_sen[0,:]
    df_reg1['LAI_high_sen']  = LAI_mean_sen[0,:] + LAI_std_sen[0,:]
    df_reg1['ALB_low_ctl']   = ALB_mean_ctl[0,:] - ALB_std_ctl[0,:]
    df_reg1['ALB_high_ctl']  = ALB_mean_ctl[0,:] + ALB_std_ctl[0,:]
    df_reg1['ALB_low_sen']   = ALB_mean_sen[0,:] - ALB_std_sen[0,:]
    df_reg1['ALB_high_sen']  = ALB_mean_sen[0,:] + ALB_std_sen[0,:]
    df_reg1['Qle_low_ctl']   = Qle_mean_ctl[0,:] - Qle_std_ctl[0,:]
    df_reg1['Qle_high_ctl']  = Qle_mean_ctl[0,:] + Qle_std_ctl[0,:]
    df_reg1['Qle_low_sen']   = Qle_mean_sen[0,:] - Qle_std_sen[0,:]
    df_reg1['Qle_high_sen']  = Qle_mean_sen[0,:] + Qle_std_sen[0,:]

    print("df_reg1", df_reg1)

    df_reg2                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen[1,:] - Tmax_mean_ctl[1,:]})
    df_reg2['LAI_mean_ctl']  = LAI_mean_ctl[1,:]
    df_reg2['LAI_mean_sen']  = LAI_mean_sen[1,:]
    df_reg2['ALB_mean_ctl']  = ALB_mean_ctl[1,:]
    df_reg2['ALB_mean_sen']  = ALB_mean_sen[1,:]
    df_reg2['Qle_mean_ctl']  = Qle_mean_ctl[1,:]
    df_reg2['Qle_mean_sen']  = Qle_mean_sen[1,:]
    df_reg2['LAI_low_ctl']   = LAI_mean_ctl[1,:] - LAI_std_ctl[1,:]
    df_reg2['LAI_high_ctl']  = LAI_mean_ctl[1,:] + LAI_std_ctl[1,:]
    df_reg2['LAI_low_sen']   = LAI_mean_sen[1,:] - LAI_std_sen[1,:]
    df_reg2['LAI_high_sen']  = LAI_mean_sen[1,:] + LAI_std_sen[1,:]
    df_reg2['ALB_low_ctl']   = ALB_mean_ctl[1,:] - ALB_std_ctl[1,:]
    df_reg2['ALB_high_ctl']  = ALB_mean_ctl[1,:] + ALB_std_ctl[1,:]
    df_reg2['ALB_low_sen']   = ALB_mean_sen[1,:] - ALB_std_sen[1,:]
    df_reg2['ALB_high_sen']  = ALB_mean_sen[1,:] + ALB_std_sen[1,:]
    df_reg2['Qle_low_ctl'] = Qle_mean_ctl[1,:] - Qle_std_ctl[1,:]
    df_reg2['Qle_high_ctl']= Qle_mean_ctl[1,:] + Qle_std_ctl[1,:]
    df_reg2['Qle_low_sen'] = Qle_mean_sen[1,:] - Qle_std_sen[1,:]
    df_reg2['Qle_high_sen']= Qle_mean_sen[1,:] + Qle_std_sen[1,:]

    df_reg3                  = pd.DataFrame({'Tmax_diff': Tmax_mean_sen[2,:] - Tmax_mean_ctl[2,:]})
    df_reg3['LAI_mean_ctl']  = LAI_mean_ctl[2,:]
    df_reg3['LAI_mean_sen']  = LAI_mean_sen[2,:]
    df_reg3['ALB_mean_ctl']  = ALB_mean_ctl[2,:]
    df_reg3['ALB_mean_sen']  = ALB_mean_sen[2,:]
    df_reg3['Qle_mean_ctl']  = Qle_mean_ctl[2,:]
    df_reg3['Qle_mean_sen']  = Qle_mean_sen[2,:]
    df_reg3['LAI_low_ctl']   = LAI_mean_ctl[2,:] - LAI_std_ctl[2,:]
    df_reg3['LAI_high_ctl']  = LAI_mean_ctl[2,:] + LAI_std_ctl[2,:]
    df_reg3['LAI_low_sen']   = LAI_mean_sen[2,:] - LAI_std_sen[2,:]
    df_reg3['LAI_high_sen']  = LAI_mean_sen[2,:] + LAI_std_sen[2,:]
    df_reg3['ALB_low_ctl']   = ALB_mean_ctl[2,:] - ALB_std_ctl[2,:]
    df_reg3['ALB_high_ctl']  = ALB_mean_ctl[2,:] + ALB_std_ctl[2,:]
    df_reg3['ALB_low_sen']   = ALB_mean_sen[2,:] - ALB_std_sen[2,:]
    df_reg3['ALB_high_sen']  = ALB_mean_sen[2,:] + ALB_std_sen[2,:]
    df_reg3['Qle_low_ctl'] = Qle_mean_ctl[2,:] - Qle_std_ctl[2,:]
    df_reg3['Qle_high_ctl']= Qle_mean_ctl[2,:] + Qle_std_ctl[2,:]
    df_reg3['Qle_low_sen'] = Qle_mean_sen[2,:] - Qle_std_sen[2,:]
    df_reg3['Qle_high_sen']= Qle_mean_sen[2,:] + Qle_std_sen[2,:]

    if 1:
        # Read in FFDI index
        FMI_file     = Dataset(FMI_file, mode='r')
        Time         = FMI_file.variables['time'][:]
        FMI_mean_ctl = FMI_file.variables['FMI_ctl_mean'][:]
        FMI_mean_sen = FMI_file.variables['FMI_sen_mean'][:]
        FMI_std_ctl  = FMI_file.variables['FMI_ctl_std'][:]
        FMI_std_sen  = FMI_file.variables['FMI_sen_std'][:]
        FMI_file.close()

        df_reg1['FMI_mean_ctl']  = FMI_mean_ctl[0,:]
        df_reg1['FMI_mean_sen']  = FMI_mean_sen[0,:]
        df_reg1['FMI_diff']      = FMI_mean_sen[0,:] - FMI_mean_ctl[0,:]
        # df_reg1['FMI_low_ctl']   = FMI_mean_ctl[0,:] - FMI_std_ctl[0,:]
        # df_reg1['FMI_high_ctl']  = FMI_mean_ctl[0,:] + FMI_std_ctl[0,:]
        # df_reg1['FMI_low_sen']   = FMI_mean_sen[0,:] - FMI_std_sen[0,:]
        # df_reg1['FMI_high_sen']  = FMI_mean_sen[0,:] + FMI_std_sen[0,:]

        print("df_reg1", df_reg1)

        df_reg2['FMI_mean_ctl']  = FMI_mean_ctl[1,:]
        df_reg2['FMI_mean_sen']  = FMI_mean_sen[1,:]
        df_reg2['FMI_diff']      = FMI_mean_sen[1,:] - FMI_mean_ctl[1,:]
        # df_reg2['FMI_low_ctl']   = FMI_mean_ctl[1,:] - FMI_std_ctl[1,:]
        # df_reg2['FMI_high_ctl']  = FMI_mean_ctl[1,:] + FMI_std_ctl[1,:]
        # df_reg2['FMI_low_sen']   = FMI_mean_sen[1,:] - FMI_std_sen[1,:]
        # df_reg2['FMI_high_sen']  = FMI_mean_sen[1,:] + FMI_std_sen[1,:]


        df_reg3['FMI_mean_ctl']  = FMI_mean_ctl[2,:]
        df_reg3['FMI_mean_sen']  = FMI_mean_sen[2,:]
        df_reg3['FMI_diff']      = FMI_mean_sen[2,:] - FMI_mean_ctl[2,:]
        # df_reg3['FMI_low_ctl']   = FMI_mean_ctl[2,:] - FMI_std_ctl[2,:]
        # df_reg3['FMI_high_ctl']  = FMI_mean_ctl[2,:] + FMI_std_ctl[2,:]
        # df_reg3['FMI_low_sen']   = FMI_mean_sen[2,:] - FMI_std_sen[2,:]
        # df_reg3['FMI_high_sen']  = FMI_mean_sen[2,:] + FMI_std_sen[2,:]


        ntime_FMI      = np.shape(FMI_mean_ctl)[0]
        time_steps_FMI = np.arange(92,92+len(Time),1)


    if 1:

        # =========== Fire date ===========
        fire_file         = Dataset(fire_path, mode='r')
        Burn_Date_tmp     = fire_file.variables['Burn_Date'][2:8,::-1,:]  # 2019-09 - 2020-02
        lat_fire          = fire_file.variables['lat'][::-1]
        lon_fire          = fire_file.variables['lon'][:]

        Burn_Date         = Burn_Date_tmp.astype(float)
        Burn_Date         = np.where(Burn_Date<=0, 99999, Burn_Date)

        Burn_Date[4:,:,:] = Burn_Date[4:,:,:]+365 # Add 365 to Jan-Feb 2020

        Burn_Date_min     = np.nanmin(Burn_Date, axis=0)

        Burn_Date_min     = np.where(Burn_Date_min>=99999, np.nan, Burn_Date_min)
        Burn_Date_min     = Burn_Date_min - 243 # start from Sep 2019

        lons_2D, lats_2D = np.meshgrid(lon_fire, lat_fire)

        mask_val         = np.zeros((3,np.shape(lons_2D)[0],np.shape(lons_2D)[1]),dtype=bool)

        for i in np.arange(3):
            mask_val[i,:,:]  = np.all(( lats_2D>loc_lats[i][0],lats_2D<loc_lats[i][1],
                                        lons_2D>loc_lons[i][0],lons_2D<loc_lons[i][1]), axis=0)

        Burn_Date_min_reg1 = np.where( mask_val[0,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg2 = np.where( mask_val[1,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg3 = np.where( mask_val[2,:,:], Burn_Date_min, np.nan)

        Burn_reg1_10th = np.nanpercentile(Burn_Date_min_reg1, 10)
        Burn_reg1_50th = np.nanpercentile(Burn_Date_min_reg1, 50)
        Burn_reg1_90th = np.nanpercentile(Burn_Date_min_reg1, 90)

        Burn_reg2_10th = np.nanpercentile(Burn_Date_min_reg2, 10)
        Burn_reg2_50th = np.nanpercentile(Burn_Date_min_reg2, 50)
        Burn_reg2_90th = np.nanpercentile(Burn_Date_min_reg2, 90)

        Burn_reg3_10th = np.nanpercentile(Burn_Date_min_reg3, 10)
        Burn_reg3_50th = np.nanpercentile(Burn_Date_min_reg3, 50)
        Burn_reg3_90th = np.nanpercentile(Burn_Date_min_reg3, 90)

        print('Burn_reg1_10th',Burn_reg1_10th)
        print('Burn_reg1_50th',Burn_reg1_50th)
        print('Burn_reg1_90th',Burn_reg1_90th)
        print('Burn_reg2_10th',Burn_reg2_10th)
        print('Burn_reg2_50th',Burn_reg2_50th)
        print('Burn_reg2_90th',Burn_reg2_90th)
        print('Burn_reg3_10th',Burn_reg3_10th)
        print('Burn_reg3_50th',Burn_reg3_50th)
        print('Burn_reg3_90th',Burn_reg3_90th)


    cleaner_dates = ["Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019", "Jan 2020", "Feb 2020",       ""]
    xtickslocs    = [         0,         30,         61,         91,       122,         153,     182 ]

    # ===================== Plotting =====================
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=[10,10], sharex=False,
                sharey=False, squeeze=True)
    # fig, axs = plt.subplots(nrows=3, ncols=3, figsize=[10,8], sharex=False,
    #             sharey=False, squeeze=True)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 12
    plt.rcParams['font.size']       = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    almost_black                    = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    time_steps = np.arange(len(time))

    # Tmax & FMI
    row = 0
    axs[row,0].axhline(y=0, color='grey', linestyle='--')
    axs[row,0].plot(df_reg1['Tmax_diff'].rolling(window=5).mean(), c = almost_black, lw=1., alpha=1)

    axs[row,1].axhline(y=0, color='grey', linestyle='--')
    axs[row,1].plot(df_reg2['Tmax_diff'].rolling(window=5).mean(), c = almost_black, lw=1., alpha=1)

    axs[row,2].axhline(y=0, color='grey', linestyle='--')
    axs[row,2].plot(df_reg3['Tmax_diff'].rolling(window=5).mean(), c = almost_black, lw=1., alpha=1)

    if 1:
        axs[row,0].plot(time_steps_FMI, df_reg1['FMI_diff'].rolling(window=5).mean(), c = 'red', lw=1., alpha=1)
        axs[row,1].plot(time_steps_FMI, df_reg2['FMI_diff'].rolling(window=5).mean(), c = 'red', lw=1., alpha=1)
        axs[row,2].plot(time_steps_FMI, df_reg3['FMI_diff'].rolling(window=5).mean(), c = 'red', lw=1., alpha=1)

    # LAI
    row = 1
    axs[row,0].fill_between(time_steps, df_reg1['LAI_low_ctl'].rolling(window=5).mean(), df_reg1['LAI_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,0].fill_between(time_steps, df_reg1['LAI_low_sen'].rolling(window=5).mean(), df_reg1['LAI_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,0].plot(df_reg1['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,0].plot(df_reg1['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    axs[row,1].fill_between(time_steps, df_reg2['LAI_low_ctl'].rolling(window=5).mean(), df_reg2['LAI_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,1].fill_between(time_steps, df_reg2['LAI_low_sen'].rolling(window=5).mean(), df_reg2['LAI_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,1].plot(df_reg2['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,1].plot(df_reg2['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    axs[row,2].fill_between(time_steps, df_reg3['LAI_low_ctl'].rolling(window=5).mean(), df_reg3['LAI_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,2].fill_between(time_steps, df_reg3['LAI_low_sen'].rolling(window=5).mean(), df_reg3['LAI_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,2].plot(df_reg3['LAI_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,2].plot(df_reg3['LAI_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    # ALB
    row = 2
    axs[row,0].fill_between(time_steps, df_reg1['ALB_low_ctl'].rolling(window=5).mean(), df_reg1['ALB_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,0].fill_between(time_steps, df_reg1['ALB_low_sen'].rolling(window=5).mean(), df_reg1['ALB_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,0].plot(df_reg1['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,0].plot(df_reg1['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    axs[row,1].fill_between(time_steps, df_reg2['ALB_low_ctl'].rolling(window=5).mean(), df_reg2['ALB_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,1].fill_between(time_steps, df_reg2['ALB_low_sen'].rolling(window=5).mean(), df_reg2['ALB_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,1].plot(df_reg2['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,1].plot(df_reg2['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    axs[row,2].fill_between(time_steps, df_reg3['ALB_low_ctl'].rolling(window=5).mean(), df_reg3['ALB_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,2].fill_between(time_steps, df_reg3['ALB_low_sen'].rolling(window=5).mean(), df_reg3['ALB_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,2].plot(df_reg3['ALB_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,2].plot(df_reg3['ALB_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    # Qle
    row = 3
    axs[row,0].fill_between(time_steps, df_reg1['Qle_low_ctl'].rolling(window=5).mean(), df_reg1['Qle_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,0].fill_between(time_steps, df_reg1['Qle_low_sen'].rolling(window=5).mean(), df_reg1['Qle_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,0].plot(df_reg1['Qle_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,0].plot(df_reg1['Qle_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    axs[row,1].fill_between(time_steps, df_reg2['Qle_low_ctl'].rolling(window=5).mean(), df_reg2['Qle_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,1].fill_between(time_steps, df_reg2['Qle_low_sen'].rolling(window=5).mean(), df_reg2['Qle_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,1].plot(df_reg2['Qle_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,1].plot(df_reg2['Qle_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    axs[row,2].fill_between(time_steps, df_reg3['Qle_low_ctl'].rolling(window=5).mean(), df_reg3['Qle_high_ctl'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.2)
    axs[row,2].fill_between(time_steps, df_reg3['Qle_low_sen'].rolling(window=5).mean(), df_reg3['Qle_high_sen'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.35)
    axs[row,2].plot(df_reg3['Qle_mean_ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=1, alpha=1)
    axs[row,2].plot(df_reg3['Qle_mean_sen'].rolling(window=5).mean(), label="exp", c = "orange", lw=1, alpha=1)

    for i in np.arange(4):
    # for i in np.arange(3):
        axs[i,0].axvline(x=Burn_reg1_10th, color='coral', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,0].axvline(x=Burn_reg1_50th, color='red', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,0].axvline(x=Burn_reg1_90th, color='brown', linestyle='-',linewidth=1.5, alpha=0.8)

        axs[i,1].axvline(x=Burn_reg2_10th, color='coral', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,1].axvline(x=Burn_reg2_50th, color='red', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,1].axvline(x=Burn_reg2_90th, color='brown', linestyle='-',linewidth=1.5, alpha=0.8)

        axs[i,2].axvline(x=Burn_reg3_10th, color='coral', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,2].axvline(x=Burn_reg3_50th, color='red', linestyle='-',linewidth=1.5, alpha=0.8)
        axs[i,2].axvline(x=Burn_reg3_90th, color='brown', linestyle='-',linewidth=1.5, alpha=0.8)

    Tmax_bot_val   = -0.6
    Tmax_up_val    = 1.
    Tmax_levels    = [-0.5,0,0.5,1]
    Tmax_labels    = ['-0.5','0.0','0.5','1.0']

    LAI_bot_val    = 0
    LAI_up_val     = 6
    LAI_levels     = [0,2,4,6]
    LAI_labels     = ['0','2','4','6']

    Albedo_bot_val = 0.06
    Albedo_up_val  = 0.17
    Albedo_levels  = [0.06,0.08,0.10,0.12,0.14,0.16]
    Albedo_labels  = ['0.06','0.08','0.10','0.12','0.14','0.16']

    Qle_bot_val    = 0
    Qle_up_val     = 200
    Qle_levels     = [0,50,100,150,200]
    Qle_labels     = ['0','50','100','150','200']

    for j in np.arange(3):
        axs[0,j].set_ylim(Tmax_bot_val,Tmax_up_val)
        axs[1,j].set_ylim(LAI_bot_val,LAI_up_val)
        axs[2,j].set_ylim(Albedo_bot_val,Albedo_up_val)
        axs[3,j].set_ylim(Qle_bot_val,Qle_up_val)
        for i in np.arange(4):
        # for i in np.arange(3):
            if i == 3:
            # if i == 2:
                axs[i,j].set_xticks(xtickslocs)
                axs[i,j].set_xticklabels(cleaner_dates,rotation=25)
            else:
                axs[i,j].set_xticks(xtickslocs)
                axs[i,j].set_xticklabels([],rotation=25)

            axs[i,j].set_xlim(0,152)

        if j == 0:
            axs[0,j].set_yticks(Tmax_levels)
            axs[0,j].set_yticklabels(Tmax_labels)

            axs[1,j].set_yticks(LAI_levels)
            axs[1,j].set_yticklabels(LAI_labels)

            axs[2,j].set_yticks(Albedo_levels)
            axs[2,j].set_yticklabels(Albedo_labels)

            axs[3,j].set_yticks(Qle_levels)
            axs[3,j].set_yticklabels(Qle_labels)
        else:
            axs[0,j].set_yticks(Tmax_levels)
            axs[0,j].set_yticklabels([])

            axs[1,j].set_yticks(LAI_levels)
            axs[1,j].set_yticklabels([])

            axs[2,j].set_yticks(Albedo_levels)
            axs[2,j].set_yticklabels([])

            axs[3,j].set_yticks(Qle_levels)
            axs[3,j].set_yticklabels([])

    # Set top titles
    axs[0,0].set_title("North")
    axs[0,1].set_title("Central")
    axs[0,2].set_title("South")

    # Set left labels
    axs[0,0].set_ylabel("ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C), ΔFMI (-)", fontsize=12)
    axs[1,0].set_ylabel("LAI (m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$)", fontsize=12)
    axs[2,0].set_ylabel("$α$ (-)", fontsize=12)
    axs[3,0].set_ylabel("LH (W m$\mathregular{^{-2}}$)", fontsize=12)

    axs[0,0].text(0.02, 0.95, "(a)", transform=axs[0,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,1].text(0.02, 0.95, "(b)", transform=axs[0,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,2].text(0.02, 0.95, "(c)", transform=axs[0,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,0].text(0.02, 0.95, "(d)", transform=axs[1,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,1].text(0.02, 0.95, "(e)", transform=axs[1,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,2].text(0.02, 0.95, "(f)", transform=axs[1,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,0].text(0.02, 0.95, "(g)", transform=axs[2,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,1].text(0.02, 0.95, "(h)", transform=axs[2,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,2].text(0.02, 0.95, "(i)", transform=axs[2,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[3,0].text(0.02, 0.95, "(j)", transform=axs[3,0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[3,1].text(0.02, 0.95, "(k)", transform=axs[3,1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[3,2].text(0.02, 0.95, "(l)", transform=axs[3,2].transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # ax.legend()
    # fig.tight_layout()

    plt.savefig('./plots/burnt_reg_time_series_version2.png',dpi=300)

if __name__ == "__main__":

    # plot burnt region time series
    case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
    case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

    file_name      = 'LIS.CABLE.201701-202002.nc'
    fire_path      = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
    wrf_path       = "/scratch/w97/mm3972/model/NUWRF/Tinderbox_drght_LAI_ALB/output/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"

    land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
    land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"


    time_s         = datetime(2019,9,1,0,0,0,0)
    time_e         = datetime(2020,3,1,0,0,0,0)

    #                   North ,        Central,       South
    loc_lats       = [[-32,-28.5],   [-34.5,-32.5], [-38,-34.5]]
    loc_lons       = [[151.5,153.5], [149.5,151.5], [146.5,151]]

    if 1:
        Tmax_file        = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_Tmax.nc"
        LAI_file         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_LAI.nc"
        ALB_file         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_Albedo.nc"
        Qle_file         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_Qle.nc"
        VPD_file         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_VPD.nc"
        SMtop_file       = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_SMtop.nc"
        FMI_file         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/max_FMI_time_series_201912_202002.nc"
        daytime_FMI_file = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/daytime_FMI_time_series_201909_202002.nc"

        Tmax_file_unburnt        = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_Tmax_unburnt.nc"
        LAI_file_unburnt         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_LAI_unburnt.nc"
        ALB_file_unburnt         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_Albedo_unburnt.nc"
        Qle_file_unburnt         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_Qle_unburnt.nc"
        VPD_file_unburnt         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_VPD_unburnt.nc"
        SMtop_file_unburnt       = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_SMtop_unburnt.nc"
        FMI_file_unburnt         = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/max_FMI_time_series_201912_202002_unburnt.nc"
        daytime_FMI_file_unburnt = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/daytime_FMI_time_series_201909_202002_unburnt.nc"
        # plot_time_series_burn_region(FMI_file, Tmax_file, LAI_file, ALB_file, Qle_file, loc_lats=loc_lats, loc_lons=loc_lons, time_s=time_s, time_e=time_e)

        plot_time_series_burn_unburn_region(FMI_file, Tmax_file, LAI_file, ALB_file, Qle_file, VPD_file,SMtop_file,
                                            FMI_file_unburnt, Tmax_file_unburnt, LAI_file_unburnt, ALB_file_unburnt, Qle_file_unburnt, VPD_file_unburnt,SMtop_file_unburnt,
                                            daytime_FMI_file, daytime_FMI_file_unburnt,
                                            loc_lats=loc_lats, loc_lons=loc_lons, time_s=time_s, time_e=time_e)
