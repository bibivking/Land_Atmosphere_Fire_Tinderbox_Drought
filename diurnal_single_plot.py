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

def read_data(land_path,case_name,var_name,pft,time_s,time_e,loc_lat=None,loc_lon=None,
              lat_name=None,lon_name=None):

    # ============ Read data ============

    file_path = [land_path + case_name + '/LIS_output/' + var_name + '/LIS.CABLE.201701-201912.nc']
    time, Var = read_var_multi_file(file_path, var_name, loc_lat, loc_lon, lat_name, lon_name)
    time, Lat = read_var(file_path[0], lat_name, loc_lat, loc_lon, lat_name, lon_name)
    time, Lon = read_var(file_path[0], lon_name, loc_lat, loc_lon, lat_name, lon_name)

    # plt.contourf(Lat)
    # plt.show()
    # plt.contourf(Lon)
    # plt.show()

    sec_3D    = np.zeros(np.shape(Var))
    day_3D    = np.zeros(np.shape(Var))
    lat_3D    = np.zeros(np.shape(Var))
    lon_3D    = np.zeros(np.shape(Var))
    
    for i,t in enumerate(time):
        print("t.seconds",t.seconds,"t.days",t.days)
        sec_3D[i,:,:]  = t.seconds
        day_3D[i,:,:]  = t.days
        lat_3D[i,:,:]  = Lat
        lon_3D[i,:,:]  = Lon

    var_shrink = np.reshape(Var,-1)
    sec_shrink = np.reshape(sec_3D,-1)
    day_shrink = np.reshape(day_3D,-1)
    lat_shrink = np.reshape(lat_3D,-1)
    lon_shrink = np.reshape(lon_3D,-1)

    var_1D     = var_shrink[~ np.isnan(var_shrink)]
    sec_1D     = sec_shrink[~ np.isnan(var_shrink)]
    day_1D     = day_shrink[~ np.isnan(var_shrink)]
    lat_1D     = lat_shrink[~ np.isnan(var_shrink)]
    lon_1D     = lon_shrink[~ np.isnan(var_shrink)]

    df         = pd.DataFrame(var_1D, columns=['var'])
    df['sec']  = sec_1D
    df['day']  = day_1D
    df['lat']  = lat_1D
    df['lon']  = lon_1D

    if pft is not None:
        mask_pft   = mask_by_pft(land_path,case_name)
        pft_shrink = np.reshape(mask_pft,-1)
        pft_1D     = pft_shrink[~ np.isnan(var_shrink)]
        df['pft']  = pft_1D

    
    t_s   = time_s - datetime(2000,1,1,0,0,0)
    t_e   = time_e - datetime(2000,1,1,0,0,0)

    mask_days = np.all([[df["day"].values >= t_s.days], [df["day"].values < t_e.days]],axis=0)
    print("np.shape(mask_days)",np.shape(mask_days))
    df        = df[mask_days[0]]

    print(df)
    
    return df

def plot_spatial_land_days(land_path,case_names,var_name,pft,time_s,time_e,loc_lat=None,loc_lon=None,
                           lat_name=None, lon_name=None, message=None):

    # ============= read data ================
    df_ctl = read_data(land_path,case_names[0],var_name,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)
    df_sen = read_data(land_path,case_names[1],var_name,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                        lat_name=lat_name, lon_name=lon_name)

    # D_ctl     = df_ctl.groupby(by=['hour','lat','lon']).mean()
    # D_sen     = df_sen.groupby(by=['hour','lat','lon']).mean()
    ctl_diurnal = np.zeros(24)
    sen_diurnal = np.zeros(24)
    i = 0
    for h in np.arange(1800,84600+3600,3600):
        ctl_diurnal[i] = np.nanmean(df_ctl[df_ctl["sec"]==h]['var'].values)
        sen_diurnal[i] = np.nanmean(df_sen[df_sen["sec"]==h]['var'].values)
        i = i + 1 
    print("ctl_diurnal",ctl_diurnal)

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
        sct     = ax.scatter(df_ctl[mask_1D]['time'], df_sen[mask_1D]['var']-df_ctl[mask_1D]['var'],  color='none', edgecolors='red',  s=9, 
                            marker=markers[0], alpha=0.05, cmap=cmap, label='ctl') #edgecolor='none', c='red'
    else: 
        sct     = ax.plot(np.arange(0,24), ctl_diurnal-273.25, color='red', s=9, marker=markers[0], cmap=cmap, label='ctl')
        sct     = ax.plot(np.arange(0,24), sen_diurnal-273.25, color='blue', s=9, marker=markers[0], cmap=cmap, label='sen')

    if pft is not None:
        fig.savefig("./plots/diurnal_"+message+"_pft="+str(pft)+"_Tmax",bbox_inches='tight')
    else:
        fig.savefig("./plots/diurnal_"+message+"_Tmax",bbox_inches='tight')

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

    # # small region
    # loc_lat    = [-33,-29]
    # loc_lon    = [147,149]

    # east coast
    loc_lat    = [-33,-27]
    loc_lon    = [152,154]
    region     = "east_coast"


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
        
        message   = "2019_preHW_8_12Jan_"+region #"2018_growth_season"
        time_s    = datetime(2019,1,8,0,0,0,0)
        time_e    = datetime(2019,1,13,0,0,0,0)
        var_name  = 'Tair_f_inst'
        plot_spatial_land_days(land_path,case_names,var_name,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)


        message   = "2019_HW_14_18Jan_"+region #"2018_growth_season"
        time_s    = datetime(2019,1,14,0,0,0,0)
        time_e    = datetime(2019,1,19,0,0,0,0)
        var_name  = 'Tair_f_inst'
        plot_spatial_land_days(land_path,case_names,var_name,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

        message   = "2019_HW_22_26Jan_"+region #"2018_growth_season"
        time_s    = datetime(2019,1,22,0,0,0,0)
        time_e    = datetime(2019,1,27,0,0,0,0)
        var_name  = 'Tair_f_inst'
        plot_spatial_land_days(land_path,case_names,var_name,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)
  

        message   = "2019_postHW_28_1Feb_"+region #"2018_growth_season"
        time_s    = datetime(2019,1,28,0,0,0,0)
        time_e    = datetime(2019,2,2,0,0,0,0)
        var_name  = 'Tair_f_inst'
        plot_spatial_land_days(land_path,case_names,var_name,pft,time_s,time_e,loc_lat=loc_lat,loc_lon=loc_lon,
                               lat_name=lat_name, lon_name=lon_name, message=message)

