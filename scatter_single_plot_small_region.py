#!/usr/bin/python

'''
Plot spitial map of land diagnosis and parameters from LIS-CABLE
1. per time step
2. time period average
cp /g/data/w97/mm3972/scripts/Groundwater_Atmosphere_Heatwave_Drought/src/Fig8_scatter_deltaT_WTD_PFT.py scatter_single_plot.py
'''

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as mticker
from convert_units import get_land_var_scale, get_land_var_range_diff
from common_utils import *


def qair_to_vpd(qair, tair, press):
    '''
    calculate vpd
    '''
    DEG_2_KELVIN = 273.15
    PA_TO_KPA    = 0.001
    PA_TO_HPA    = 0.01

    # convert back to Pa
    press        /= PA_TO_HPA
    tair         -= DEG_2_KELVIN

    # saturation vapor pressure
    es = 100.0 * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # vapor pressure
    ea = (qair * press) / (0.622 + (1.0 - 0.622) * qair)

    vpd = (es - ea) * PA_TO_KPA
    vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd

def mask_by_pft(land_path,case_name):

    print("In mask_by_pft Var")

    file_path  = land_path + case_name + '/LIS_output/Landcover_inst/LIS.CABLE.201701-201912.nc'
    f          = Dataset(file_path, mode='r')
    Time       = nc.num2date(f.variables['time'][:],f.variables['time'].units,
                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    time       = UTC_to_AEST(Time) - datetime(2000,1,1,0,0,0)
    Var        = f.variables['Landcover_inst']
    var        = time_clip_to_day(time,Var,time_s,time_e)
    # var        = np.where(var == iveg, var, np.nan) # awap_t == current for awap_t in AWAP_time

    print(var)

    return var

def read_data(land_path,case_name,var_name,pft,time_s,time_e,loc_lat=None,loc_lon=None,
              lat_name=None,lon_name=None):

    # ============ Read data ============
    if var_name =='Tmax':
        file_path = [land_path + case_name + '/LIS_output/Tair_f_inst/LIS.CABLE.201701-201912.nc']
        time, Var = read_var_multi_file(file_path, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        var       = time_clip_to_day_max(time,Var,time_s,time_e)
    elif var_name =='Tmin':
        file_path = [land_path + case_name + '/LIS_output/Tair_f_inst/LIS.CABLE.201701-201912.nc']
        time, Var = read_var_multi_file(file_path, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        var       = time_clip_to_day_min(time,Var,time_s,time_e)
    elif var_name =='VPDmax':
        file_path  = [land_path + case_name + '/LIS_output/Tair_f_inst/LIS.CABLE.201701-201912.nc']
        time, Tair = read_var_multi_file(file_path, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        file_path  = [land_path + case_name + '/LIS_output/Qair_f_inst/LIS.CABLE.201701-201912.nc']
        time, Qair = read_var_multi_file(file_path, "Qair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        Var        = qair_to_vpd(Qair, Tair, 1000.)
        var        = time_clip_to_day_max(time,Var,time_s,time_e)
    elif var_name =='VPDmin':
        file_path  = [land_path + case_name + '/LIS_output/Tair_f_inst/LIS.CABLE.201701-201912.nc']
        time, Tair = read_var_multi_file(file_path, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        file_path  = [land_path + case_name + '/LIS_output/Qair_f_inst/LIS.CABLE.201701-201912.nc']
        time, Qair = read_var_multi_file(file_path, "Qair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        Var        = qair_to_vpd(Qair, Tair, 1000.)
        var        = time_clip_to_day_min(time,Var,time_s,time_e)
    else:
        file_path = [land_path + case_name + '/LIS_output/' + var_name + '/LIS.CABLE.201701-201912.nc']
        time, Var = read_var_multi_file(file_path, var_name, loc_lat, loc_lon, lat_name, lon_name)
        var       = time_clip_to_day(time,Var,time_s,time_e)

    # f         = Dataset(file_path, mode='r')
    # Time      = nc.num2date(f.variables['time'][:],f.variables['time'].units,
    #             only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    # time      = UTC_to_AEST(Time) - datetime(2000,1,1,0,0,0)
    # Var       = f.variables[var_name]
    # print("np.shape(Var)",np.shape(Var))
    # var       = time_clip_to_day_max(time,Var,time_s,time_e)
    # print("np.shape(var)",np.shape(var))

    t_s       = time_s - datetime(2000,1,1,0,0,0)
    t_e       = time_e - datetime(2000,1,1,0,0,0)

    time_3D   = np.zeros(np.shape(var))

    j=0
    for i in np.arange(t_s.days, t_e.days):
        # print("j=",j,"i=",i)
        time_3D[j,:,:] = i
        j += 1

    var_shrink = np.reshape(var,-1)
    time_shrink= np.reshape(time_3D,-1)
    var_1D     = var_shrink[~ np.isnan(var_shrink)]
    time_1D    = time_shrink[~ np.isnan(var_shrink)]

    df         = pd.DataFrame(var_1D, columns=['var'])
    df['time'] = time_1D

    if pft is not None:
        mask_pft   = mask_by_pft(land_path,case_name)
        pft_shrink = np.reshape(mask_pft,-1)
        pft_1D     = pft_shrink[~ np.isnan(var_shrink)]
        df['pft']  = pft_1D

    print(df)

    return df

def plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=None,loc_lon=None,
                           lat_name=None, lon_name=None, message=None):

    # ============= read data ================
    if len(var_names) ==1:
        df_ctl = read_data(land_path,case_names[0],var_names[0],pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)
        df_sen = read_data(land_path,case_names[1],var_names[0],pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)
    else:
        df_ctl = read_data(land_path,case_names[0],var_names[0],pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)
        df_sen = read_data(land_path,case_names[1],var_names[0],pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)
        df_ctl_2 = read_data(land_path,case_names[0],var_names[1],pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)
        df_sen_2 = read_data(land_path,case_names[1],var_names[1],pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)
        df_ctl['var2']  = df_ctl_2['var'][:]
        df_sen['var2']  = df_sen_2['var'][:]

    # ============ Setting for plotting ============
    cmap     = plt.cm.BrBG #YlOrBr #coolwarm_r

    markers  = ["o","o","o","^","s"]
    mrk_sz   = 1.5

    # fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=[12,8],sharex=True, sharey=True, squeeze=True) #
    fig, ax = plt.subplots(figsize=[20, 10])
    # plt.subplots_adjust(wspace=0.0, hspace=0.0)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
    if pft is not None:
        mask_1D = df_ctl['pft'] == pft
        if len(var_names) ==1:
            sct     = ax.scatter(df_ctl[mask_1D]['time'], df_sen[mask_1D]['var']-df_ctl[mask_1D]['var'],  color='none', edgecolors='red',  s=9,
                            marker=markers[0], alpha=0.05, cmap=cmap, label='ctl') #edgecolor='none', c='red'
        else:
            sct     = ax.scatter(df_ctl[mask_1D]['var2'], df_sen[mask_1D]['var']-df_ctl[mask_1D]['var'],  color='none', edgecolors='red',  s=9,
                            marker=markers[0], alpha=0.05, cmap=cmap, label='ctl') #edgecolor='none', c='red'
    else:

        if len(var_names) ==1:
            # plot scatters
            sct         = ax.scatter(df_ctl['time'], df_sen['var']-df_ctl['var'], color='none', edgecolors='red', s=9, marker=markers[0], alpha=0.05, cmap=cmap, label='ctl')
            # plot 0 values
            sct         = ax.axhline(y=0, color="black", lw=1.0, alpha=1, linestyle='-')
            # plot median values
            t_s         = time_s - datetime(2000,1,1,0,0,0,0)
            t_e         = time_e - datetime(2000,1,1,0,0,0,0)
            time_series = np.arange(t_s.days,t_e.days,1)
            dctl        = df_ctl.groupby(by=['time']).median()
            dsen        = df_sen.groupby(by=['time']).median()
            sct         = ax.plot(time_series, dsen-dctl, c='blue', lw=1.0, alpha=1, label='median')
        else:
            sct         = ax.scatter(df_ctl['var2'], df_sen['var']-df_ctl['var'], color='none', edgecolors='red', s=9, marker=markers[0], alpha=0.05, cmap=cmap, label='ctl')
            sct         = ax.axhline(y=0, color="black", lw=1.0, alpha=1, linestyle='-')
    if pft is not None:
        fig.savefig("./plots/scatter_"+message+"_pft="+str(pft)+"_Tmax",bbox_inches='tight')
    else:
        fig.savefig("./plots/scatter_"+message,bbox_inches='tight')

if __name__ == "__main__":

    # =============================== Operation ================================
    region = "SE Aus" #"SE Aus" #"CORDEX" #"SE Aus"

    if region == "Aus":
        loc_lat    = [-44,-10]
        loc_lon    = [112,154]
    elif region == "SE Aus":
        loc_lat    = [-40,-25]
        loc_lon    = [135,155]
    elif region == "CORDEX":
        loc_lat    = [-52.36,3.87]
        loc_lon    = [89.25,180]

    PFT        = False
    lat_name   = "lat"
    lon_name   = "lon"

    # =========================== Plot =============================

    if 1:
        land_path = '/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/'
        case_names= ['drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2',
                     'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB']

        pft       = None #2

        # pft      = ["BEF","crop","shrub","grass","barren"]
        # iveg_num = [2, 9, 5, 6, 14]

        # small region
        loc_lat    = [-33,-29]
        loc_lon    = [147,149]

        time_s  = datetime(2017,12,1,0,0,0,0)
        time_e  = datetime(2018,3,1,0,0,0,0)
        var_names  = ['Tmax','VPDmax']
        message    = "201718_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmin','VPDmin'] #'Tair_f_inst'
        message    = "201718_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmax','FWsoil_tavg']
        message    = "201718_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmin','FWsoil_tavg'] #'Tair_f_inst'
        message    = "201718_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)


        time_s  = datetime(2018,12,1,0,0,0,0)
        time_e  = datetime(2019,3,1,0,0,0,0)
        var_names  = ['Tmax','VPDmax']
        message    = "201819_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmin','VPDmin'] #'Tair_f_inst'
        message    = "201819_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmax','FWsoil_tavg']
        message    = "201819_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmin','FWsoil_tavg'] #'Tair_f_inst'
        message    = "201819_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        time_s  = datetime(2019,12,1,0,0,0,0)
        time_e  = datetime(2020,2,1,0,0,0,0)
        var_names  = ['Tmax','VPDmax']
        message    = "201920_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmin','VPDmin'] #'Tair_f_inst'
        message    = "201920_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmax','FWsoil_tavg']
        message    = "201920_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        var_names  = ['Tmin','FWsoil_tavg'] #'Tair_f_inst'
        message    = "201920_summer_small_region_"+var_names[0]+"_vs_"+var_names[1]
        plot_spatial_land_days(land_path,case_names,var_names,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)
