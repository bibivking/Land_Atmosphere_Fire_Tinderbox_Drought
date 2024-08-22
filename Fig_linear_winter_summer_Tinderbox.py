#!/usr/bin/env python
"""
Produce the Australia map with the label of EucFACE site - Figure 1
"""

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"


#!/usr/bin/python

import sys
import cartopy
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from scipy import stats
import scipy.ndimage as ndimage
from scipy.interpolate import griddata, interp1d
from netCDF4 import Dataset,num2date
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
from common_utils import *

def read_LIS_var(file_name, land_ctl_path, land_sen_path, var_name, loc_lat, loc_lon, lat_names, lon_names, time_ss, time_es):

    if var_name in ["Tmax","Tmin","TDR"]:
        land_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        land_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp - 273.15
        Sen_tmp       = Sen_tmp - 273.15
    elif var_name in ["VegTmax","VegTmin","VegTDR"]:
        land_ctl_files= [land_ctl_path+'VegT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'VegT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp - 273.15
        Sen_tmp       = Sen_tmp - 273.15
    elif var_name in ["SurfTmax","SurfTmin","SurfTDR"]:
        land_ctl_files= [land_ctl_path+'AvgSurfT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'AvgSurfT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp - 273.15
        Sen_tmp       = Sen_tmp - 273.15
    elif var_name in ["Rnet",]:
        land_ctl_files= [land_ctl_path+'Lwnet_tavg/'+file_name]
        land_sen_files= [land_sen_path+'Lwnet_tavg/'+file_name]
        time, Ctl_Lwnet_tmp = read_var_multi_file(land_ctl_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Lwnet_tmp = read_var_multi_file(land_sen_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_ctl_files= [land_ctl_path+'Swnet_tavg/'+file_name]
        land_sen_files= [land_sen_path+'Swnet_tavg/'+file_name]
        time, Ctl_Swnet_tmp = read_var_multi_file(land_ctl_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Swnet_tmp = read_var_multi_file(land_sen_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp = Ctl_Lwnet_tmp+Ctl_Swnet_tmp
        Sen_tmp = Sen_Lwnet_tmp+Sen_Swnet_tmp
    elif var_name in ["EF",]:
        land_ctl_files      = [land_ctl_path+'Lwnet_tavg/'+file_name]
        land_sen_files      = [land_sen_path+'Lwnet_tavg/'+file_name]
        time, Ctl_Lwnet_tmp = read_var_multi_file(land_ctl_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Lwnet_tmp = read_var_multi_file(land_sen_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_ctl_files      = [land_ctl_path+'Swnet_tavg/'+file_name]
        land_sen_files      = [land_sen_path+'Swnet_tavg/'+file_name]
        time, Ctl_Swnet_tmp = read_var_multi_file(land_ctl_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Swnet_tmp = read_var_multi_file(land_sen_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_ctl_files      = [land_ctl_path+'Qle_tavg/'+file_name]
        land_sen_files      = [land_sen_path+'Qle_tavg/'+file_name]
        time, Ctl_Qle_tmp   = read_var_multi_file(land_ctl_files, 'Qle_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Qle_tmp   = read_var_multi_file(land_sen_files, 'Qle_tavg', loc_lat, loc_lon, lat_names, lon_names)
        ctl_Rnet            = Ctl_Lwnet_tmp + Ctl_Swnet_tmp
        sen_Rnet            = Sen_Lwnet_tmp + Sen_Swnet_tmp
        Ctl_tmp             = np.where(abs(ctl_Rnet)>0.01, Ctl_Qle_tmp/ctl_Rnet,np.nan)
        Sen_tmp             = np.where(abs(sen_Rnet)>0.01, Sen_Qle_tmp/sen_Rnet,np.nan)
    elif var_name in ["SM_top50cm",]:
        land_ctl_files = [land_ctl_path+'SoilMoist_inst/'+file_name]
        land_sen_files = [land_sen_path+'SoilMoist_inst/'+file_name]
        time, Ctl_temp = read_var_multi_file(land_ctl_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_temp = read_var_multi_file(land_sen_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        # [.022, .058, .154, .409, 1.085, 2.872]
        Ctl_tmp    = Ctl_temp[:,0,:,:]*0.022 + Ctl_temp[:,1,:,:]*0.058 + Ctl_temp[:,2,:,:]*0.154 + Ctl_temp[:,3,:,:]*0.266
        Sen_tmp    = Sen_temp[:,0,:,:]*0.022 + Sen_temp[:,1,:,:]*0.058 + Sen_temp[:,2,:,:]*0.154 + Sen_temp[:,3,:,:]*0.266
    elif var_name in ['VPD','VPDmax','VPDmin']:
        tair_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        tair_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        qair_ctl_files= [land_ctl_path+'Qair_f_inst/'+file_name]
        qair_sen_files= [land_sen_path+'Qair_f_inst/'+file_name]
        pres_ctl_files= ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']
        pres_sen_files= ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']

        time, Tair_ctl    = read_var_multi_file(tair_ctl_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Tair_sen    = read_var_multi_file(tair_sen_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_ctl    = read_var_multi_file(qair_ctl_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_sen    = read_var_multi_file(qair_sen_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time_wrf, Pres_ctl_tmp= read_var_multi_file(pres_ctl_files, "slp", loc_lat, loc_lon, lat_names, lon_names)
        time_wrf, Pres_sen_tmp= read_var_multi_file(pres_sen_files, "slp", loc_lat, loc_lon, lat_names, lon_names)

        time_in = []
        time_out= []
        for t in time_wrf:
            time_in.append(t.total_seconds())
        for t in time:
            time_out.append(t.total_seconds())

        f_ctl             = interp1d(np.array(time_in), Pres_ctl_tmp[:], kind='linear',fill_value='extrapolate', axis=0)
        f_sen             = interp1d(np.array(time_in), Pres_sen_tmp[:],kind='linear', fill_value='extrapolate', axis=0)
        Pres_ctl          = f_ctl(np.array(time_out))
        Pres_sen          = f_sen(np.array(time_out))
        Ctl_tmp           = qair_to_vpd(Qair_ctl, Tair_ctl, Pres_ctl)
        Sen_tmp           = qair_to_vpd(Qair_sen, Tair_sen, Pres_sen)
    else:
        land_ctl_files= [land_ctl_path+var_name+'/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+var_name+'/LIS.CABLE.201701-202002.nc']
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)

    nlat           = np.shape(Ctl_tmp)[1]
    nlon           = np.shape(Ctl_tmp)[2]

    if 'max' in var_name:
        # average of daily max
        ctl_in       = time_clip_to_day_max(time,Ctl_tmp, time_ss, time_es)
        sen_in       = time_clip_to_day_max(time,Sen_tmp, time_ss, time_es)
    elif 'min' in var_name:
        # average of daily min
        ctl_in       = time_clip_to_day_min(time,Ctl_tmp, time_ss, time_es)
        sen_in       = time_clip_to_day_min(time,Sen_tmp, time_ss, time_es)
    elif 'TDR' in var_name:
        # average of daily min
        ctl_in_max   = time_clip_to_day_max(time,Ctl_tmp, time_ss, time_es)
        sen_in_max   = time_clip_to_day_max(time,Sen_tmp, time_ss, time_es)
        ctl_in_min   = time_clip_to_day_min(time,Ctl_tmp, time_ss, time_es)
        sen_in_min   = time_clip_to_day_min(time,Sen_tmp, time_ss, time_es)
        ctl_in       = ctl_in_max - ctl_in_min
        sen_in       = sen_in_max - sen_in_min
    else:
        ctl_in       = time_clip_to_day(time,Ctl_tmp, time_ss, time_es)
        sen_in       = time_clip_to_day(time,Sen_tmp, time_ss, time_es)
    var_diff  = sen_in - ctl_in

    if var_name in ['WaterTableD_tavg','WatTable']:
        var_diff     = var_diff/1000.
    if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        var_diff   = var_diff*3600*24
    if var_name in ['Qair_f_inst']:
        var_diff     = var_diff*1000
    if var_name in ['GPP_tavg','NPP_tavg']:
        s2d        = 3600*24.          # s-1 to d-1
        GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
        var_diff     = var_diff*GPP_scale

    return var_diff

def linear_regress(x_values, y_values):

    # Fit a linear regression model to the data
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)

    x_series = np.linspace(np.min(x_values),np.max(x_values),300)

    # Calculate the predicted y-values for the linear regression model
    y_pred = slope * x_series + intercept

    return slope, x_series, y_pred

def linear_winter_summer(     land_ctl_path, land_sen_path, time_ss=None,
                              time_es=None, lat_names="lat", lon_names="lon",loc_lat=None,
                              loc_lon=None, wrf_path=None,  message=None):

    '''
    plot a single spatial map
    '''

    # read lat and lon outs
    f_lc      = Dataset(land_ctl_path+'LIS.CABLE.201912-201912.d01.nc',  mode='r')
    landcover = f_lc.variables['Landcover_inst'][0,:,:]
    land_type = [2,5,6,9,14]

    Tmax_diff_1718   = read_LIS_var("LIS.CABLE.201712-201802.nc", land_ctl_path, land_sen_path, 'Tmax', loc_lat, loc_lon, lat_names, lon_names, time_ss[0], time_es[0])
    Albedo_diff_1718 = read_LIS_var("LIS.CABLE.201712-201802.nc", land_ctl_path, land_sen_path, 'Albedo_inst', loc_lat, loc_lon, lat_names, lon_names, time_ss[0], time_es[0])
    LAI_diff_1718    = read_LIS_var("LIS.CABLE.201712-201802.nc", land_ctl_path, land_sen_path, 'LAI_inst', loc_lat, loc_lon, lat_names, lon_names, time_ss[0], time_es[0])
    Tmax_diff_1819   = read_LIS_var("LIS.CABLE.201812-201902.nc", land_ctl_path, land_sen_path, 'Tmax', loc_lat, loc_lon, lat_names, lon_names, time_ss[1], time_es[1])
    Albedo_diff_1819 = read_LIS_var("LIS.CABLE.201812-201902.nc", land_ctl_path, land_sen_path, 'Albedo_inst', loc_lat, loc_lon, lat_names, lon_names, time_ss[1], time_es[1])
    LAI_diff_1819    = read_LIS_var("LIS.CABLE.201812-201902.nc", land_ctl_path, land_sen_path, 'LAI_inst', loc_lat, loc_lon, lat_names, lon_names, time_ss[1], time_es[1])
    Tmax_diff_1920   = read_LIS_var("LIS.CABLE.201912-202002.nc", land_ctl_path, land_sen_path, 'Tmax', loc_lat, loc_lon, lat_names, lon_names, time_ss[2], time_es[2])
    Albedo_diff_1920 = read_LIS_var("LIS.CABLE.201912-202002.nc", land_ctl_path, land_sen_path, 'Albedo_inst', loc_lat, loc_lon, lat_names, lon_names, time_ss[2], time_es[2])
    LAI_diff_1920    = read_LIS_var("LIS.CABLE.201912-202002.nc", land_ctl_path, land_sen_path, 'LAI_inst', loc_lat, loc_lon, lat_names, lon_names, time_ss[2], time_es[2])

    print('np.shape(Tmax_diff_1718)',np.shape(Tmax_diff_1718))
    Tmax_diff   = np.concatenate((Tmax_diff_1718, Tmax_diff_1819, Tmax_diff_1920), axis=0)
    Albedo_diff = np.concatenate((Albedo_diff_1718, Albedo_diff_1819, Albedo_diff_1920), axis=0)
    LAI_diff    = np.concatenate((LAI_diff_1718, LAI_diff_1819, LAI_diff_1920), axis=0)

    print('np.shape(Tmax_diff)',np.shape(Tmax_diff))

    Tmax_diff   = np.where(Tmax_diff==-9999, np.nan, Tmax_diff)
    Albedo_diff = np.where(Albedo_diff==-9999, np.nan, Albedo_diff)
    LAI_diff    = np.where(LAI_diff==-9999, np.nan, LAI_diff)

    # Forest
    mask_forest    = (landcover==2)

    # Non Forest
    mask_nonforest = (landcover>2) & (landcover<14)

    ntime                 = len(Tmax_diff[:,0,0])
    mask_forest_ext       = np.expand_dims(mask_forest, axis=0)
    mask_nonforest_ext    = np.expand_dims(mask_nonforest, axis=0)
    print("np.shape(mask_forest_ext)",np.shape(mask_forest_ext))

    mask_forest_3D        = np.repeat(mask_forest_ext, ntime, axis=0)
    mask_nonforest_3D     = np.repeat(mask_nonforest_ext, ntime, axis=0)
    print("np.shape(mask_forest_3D)",np.shape(mask_forest_3D))

    # forest
    Tmax_diff_forest      = np.where(mask_forest_3D, Tmax_diff, np.nan)
    Albedo_diff_forest    = np.where(mask_forest_3D, Albedo_diff, np.nan)
    LAI_diff_forest       = np.where(mask_forest_3D, LAI_diff, np.nan)

    # non forest
    Tmax_diff_nonforest   = np.where(mask_nonforest_3D, Tmax_diff, np.nan)
    Albedo_diff_nonforest = np.where(mask_nonforest_3D, Albedo_diff, np.nan)
    LAI_diff_nonforest    = np.where(mask_nonforest_3D, LAI_diff, np.nan)

    # convert to 1 D
    Tmax_diff_1D             = Tmax_diff.reshape(-1)
    Albedo_diff_1D           = Albedo_diff.reshape(-1)
    LAI_diff_1D              = LAI_diff.reshape(-1)

    Tmax_diff_forest_1D      = Tmax_diff_forest.reshape(-1)
    Albedo_diff_forest_1D    = Albedo_diff_forest.reshape(-1)
    LAI_diff_forest_1D       = LAI_diff_forest.reshape(-1)

    Tmax_diff_nonforest_1D   = Tmax_diff_nonforest.reshape(-1)
    Albedo_diff_nonforest_1D = Albedo_diff_nonforest.reshape(-1)
    LAI_diff_nonforest_1D    = LAI_diff_nonforest.reshape(-1)

    # remove nan values
    Tmax_diff_1D_nonnan             = Tmax_diff_1D[~np.isnan(Tmax_diff_1D)]
    Albedo_diff_1D_nonnan           = Albedo_diff_1D[~np.isnan(Albedo_diff_1D)]
    LAI_diff_1D_nonnan              = LAI_diff_1D[~np.isnan(LAI_diff_1D)]

    Tmax_diff_forest_1D_nonnan      = Tmax_diff_forest_1D[~np.isnan(Tmax_diff_forest_1D)]
    Albedo_diff_forest_1D_nonnan    = Albedo_diff_forest_1D[~np.isnan(Albedo_diff_forest_1D)]
    LAI_diff_forest_1D_nonnan       = LAI_diff_forest_1D[~np.isnan(LAI_diff_forest_1D)]

    Tmax_diff_nonforest_1D_nonnan   = Tmax_diff_nonforest_1D[~np.isnan(Tmax_diff_nonforest_1D)]
    Albedo_diff_nonforest_1D_nonnan = Albedo_diff_nonforest_1D[~np.isnan(Albedo_diff_nonforest_1D)]
    LAI_diff_nonforest_1D_nonnan    = LAI_diff_nonforest_1D[~np.isnan(LAI_diff_nonforest_1D)]

    print("np.shape(LAI_diff_1D_nonnan)",np.shape(LAI_diff_1D_nonnan))
    print("np.shape(LAI_diff_forest_1D_nonnan)",np.shape(LAI_diff_forest_1D_nonnan))
    print("np.shape(LAI_diff_nonforest_1D_nonnan)",np.shape(LAI_diff_nonforest_1D_nonnan))

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[8,5],sharex=False,
                sharey=False, squeeze=True)

    # plt.subplots_adjust(wspace=-0.23, hspace=0.105)

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

    texts = ["(a)","(b)",
             "(c)","(d)",
             "(e)","(f)"]

    plot1 = axs[0].scatter(LAI_diff,    Tmax_diff, color='none', edgecolors='lightgrey',  s=1,
                            marker="o", alpha=0.05, label='ctl') #edgecolor='none', c='red'

    plot2 = axs[1].scatter(Albedo_diff, Tmax_diff, color='none', edgecolors='lightgrey',  s=1,
                            marker="o", alpha=0.05, label='ctl')

    slope_Tmax_LAI, x_series_Tmax_LAI, y_pred_Tmax_LAI          = linear_regress(LAI_diff_1D_nonnan, Tmax_diff_1D_nonnan)
    slope_Tmax_Albedo, x_series_Tmax_Albedo, y_pred_Tmax_Albedo = linear_regress(Albedo_diff_1D_nonnan, Tmax_diff_1D_nonnan)

    slope_Tmax_LAI_forest, x_series_Tmax_LAI_forest, y_pred_Tmax_LAI_forest       = \
                        linear_regress(LAI_diff_forest_1D_nonnan, Tmax_diff_forest_1D_nonnan)
    slope_Tmax_Albedo_forest, x_series_Tmax_Albedo_forest, y_pred_Tmax_Albedo_forest = \
                        linear_regress(Albedo_diff_forest_1D_nonnan, Tmax_diff_forest_1D_nonnan)

    slope_Tmax_LAI_nonforest, x_series_Tmax_LAI_nonforest, y_pred_Tmax_LAI_nonforest       = \
                        linear_regress(LAI_diff_nonforest_1D_nonnan, Tmax_diff_nonforest_1D_nonnan)
    slope_Tmax_Albedo_nonforest, x_series_Tmax_Albedo_nonforest, y_pred_Tmax_Albedo_nonforest = \
                        linear_regress(Albedo_diff_nonforest_1D_nonnan, Tmax_diff_nonforest_1D_nonnan)

    # Plot the linear fitting line
    plot1= axs[0].plot(x_series_Tmax_LAI, y_pred_Tmax_LAI, color='black', linestyle='solid')
    plot1= axs[0].plot(x_series_Tmax_LAI_forest, y_pred_Tmax_LAI_forest, color='forestgreen', linestyle='solid')
    plot1= axs[0].plot(x_series_Tmax_LAI_nonforest, y_pred_Tmax_LAI_nonforest, color='orange', linestyle='solid')
    plot2= axs[1].plot(x_series_Tmax_Albedo, y_pred_Tmax_Albedo, color='black', linestyle='solid')
    plot2= axs[1].plot(x_series_Tmax_Albedo_forest, y_pred_Tmax_Albedo_forest, color='forestgreen', linestyle='solid')
    plot2= axs[1].plot(x_series_Tmax_Albedo_nonforest, y_pred_Tmax_Albedo_nonforest, color='orange', linestyle='solid')

    print(f'slope\n {slope_Tmax_LAI}\n{slope_Tmax_LAI_forest}\n{slope_Tmax_LAI_nonforest}')
    print(f'slope\n {slope_Tmax_Albedo}\n{slope_Tmax_Albedo_forest}\n{slope_Tmax_Albedo_nonforest}')

    axs[0].text(0.02, 0.15, f'slope\n {slope_Tmax_LAI:.2g}\n{slope_Tmax_LAI_forest:.2g}\n{slope_Tmax_LAI_nonforest:.2g}',
               transform=axs[0].transAxes, fontsize=9, verticalalignment='top', bbox=props)
    axs[1].text(0.02, 0.15, f'slope\n {slope_Tmax_Albedo:.2g}\n{slope_Tmax_Albedo_forest:.2g}\n{slope_Tmax_Albedo_nonforest:.2g}',
               transform=axs[1].transAxes, fontsize=9, verticalalignment='top', bbox=props)

    axs[0].set_title("ΔT$\mathregular{_{max}}$ vs ΔLAI")
    axs[1].set_title("ΔT$\mathregular{_{max}}$ vs Δ$α$")

    # Apply tight layout
    # plt.tight_layout()
    plt.savefig('./plots/scatter_linear_summer_'+message + '.png',dpi=300)

if __name__ == "__main__":


    # ======================= Option =======================
    region = "SE Aus" #"SE Aus" #"CORDEX" #"SE Aus"

    if region == "Aus":
        loc_lat    = [-44,-10]
        loc_lon    = [112,154]
    elif region == "SE Aus":
        loc_lat    = [-40,-23]
        loc_lon    = [134,155]
    elif region == "CORDEX":
        loc_lat    = [-52.36,3.87]
        loc_lon    = [89.25,180]

    #######################################################
    # Decks to run:
    #    plot a single map
    #######################################################
    if 1:
        '''
        Test WRF-CABLE LIS output
        '''

        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/wrfout_d01_2019-12-01_01:00:00"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"
        atmo_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/WRF_output/"
        atmo_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/"

        time_ss    = [  datetime(2017,12,1,0,0,0,0),
                        datetime(2018,12,1,0,0,0,0),
                        datetime(2019,12,1,0,0,0,0)]

        time_es    = [ datetime(2018,3,1,0,0,0,0),
                       datetime(2019,3,1,0,0,0,0),
                       datetime(2020,3,1,0,0,0,0)]

        message    = "Summer"

        linear_winter_summer(land_ctl_path, land_sen_path, time_ss=time_ss, time_es=time_es, lat_names="lat",
                            lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, message=message)
