
import os
import sys
import cartopy
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata, interp1d
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, OCEAN
from wrf import (getvar, to_np, vertcross, CoordPair,
                 get_cartopy, latlon_coords, ALL_TIMES)
from common_utils import *


def read_LIS_diff(var_name,file_name,land_ctl_path,land_sen_path, lat_names, lon_names, loc_lat=None, loc_lon=None, time_s=None,time_e=None):

    print("plotting "+var_name)

    if var_name in ["Tmax","Tmin","TDR"]:
        land_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        land_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
    elif var_name in ["VegTmax","VegTmin","VegTDR"]:
        land_ctl_files= [land_ctl_path+'VegT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'VegT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
    elif var_name in ["SurfTmax","SurfTmin","SurfTDR"]:
        land_ctl_files= [land_ctl_path+'AvgSurfT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'AvgSurfT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
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
        land_ctl_files= [land_ctl_path+var_name+'/'+file_name]
        land_sen_files= [land_sen_path+var_name+'/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)

    if 'max' in var_name:
        # average of daily max
        ctl_in       = spatial_var_max(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var_max(time,Sen_tmp,time_s,time_e)
    elif 'min' in var_name:
        # average of daily min
        ctl_in       = spatial_var_min(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var_min(time,Sen_tmp,time_s,time_e)
    elif 'TDR' in var_name:
        # average of daily min
        ctl_in_max   = spatial_var_max(time,Ctl_tmp,time_s,time_e)
        sen_in_max   = spatial_var_max(time,Sen_tmp,time_s,time_e)
        ctl_in_min   = spatial_var_min(time,Ctl_tmp,time_s,time_e)
        sen_in_min   = spatial_var_min(time,Sen_tmp,time_s,time_e)
        ctl_in       = ctl_in_max - ctl_in_min
        sen_in       = sen_in_max - sen_in_min
    else:
        ctl_in       = spatial_var(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var(time,Sen_tmp,time_s,time_e)

    if var_name in ['WaterTableD_tavg','WatTable']:
        ctl_in     = ctl_in/1000.
        sen_in     = sen_in/1000.
    if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        t_s        = time_s - datetime(2000,1,1,0,0,0,0)
        t_e        = time_e - datetime(2000,1,1,0,0,0,0)
        ctl_in     = ctl_in*3600*24*(t_e.days - t_s.days)
        sen_in     = sen_in*3600*24*(t_e.days - t_s.days)
    if var_name in ['Qair_f_inst']:
        ctl_in     = ctl_in*1000
        sen_in     = sen_in*1000
    if var_name in ['GPP_tavg','NPP_tavg']:
        t_s        = time_s - datetime(2000,1,1,0,0,0,0)
        t_e        = time_e - datetime(2000,1,1,0,0,0,0)
        s2d        = 3600*24.          # s-1 to d-1
        GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
        ctl_in     = ctl_in*GPP_scale*(t_e.days - t_s.days)
        sen_in     = sen_in*GPP_scale*(t_e.days - t_s.days)

    var_diff     = sen_in - ctl_in


    # Select clevs
    cmap  = plt.cm.BrBG
    if var_name in ['WaterTableD_tavg','WatTable']:
        clevs = [-4,-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3,4]
    elif var_name in ['GWwb_tavg','GWMoist']:
        clevs = [-0.05,-0.04,-0.03,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.03,0.04,0.05]
    elif  var_name in ["Qair_f_inst"]:
        clevs = [-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    elif var_name in ['SoilMoist_inst','SoilMoist',"SM_top50cm"]:
        clevs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
    elif var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        clevs = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,-5,5,10,20.,30,40,50,60,70,80,90,100]
    elif var_name in ["GPP_tavg","NPP_tavg",]:
        clevs = [-200,-190,-180,-170,-160,-150,-140,-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
                    -5,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50,60,70]
        cmap  = plt.cm.BrBG
    elif var_name in ["CanopInt_inst","SnowCover_inst"]:
        clevs = [-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2.]
    elif var_name in ["Qle_tavg","Qh_tavg","Qg_tavg","Rnet",]:
        clevs = [-140,-120,-100,-80,-60,-40,-20,-10,-5,5,10,20,40,60,80,100,120,140]
    elif var_name in ["Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst","Rnet"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Wind_f_inst",]:
        clevs = [-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4]
    elif var_name in ["Psurf_f_inst"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Tair_f_inst","Tmax","Tmin","VegT_tavg","VegTmax","VegTmin",
                        "AvgSurfT_tavg","SurfTmax","SurfTmin","SoilTemp_inst",'TDR','VegTDR','SurfTDR']:
        # clevs = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
        clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        cmap  = plt.cm.seismic
    elif var_name in ["Wind_f_inst",]:
        clevs = [-2.,-1.5,-1,-0.5,-0.1,0.1,0.5,1.,1.5,2.]
    elif var_name in ["FWsoil_tavg","SmLiqFrac_inst","SmFrozFrac_inst"]:
        clevs = [-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    elif var_name in ["LAI_inst"]:
        clevs = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = plt.cm.BrBG
    elif var_name in ["VPD","VPDmax","VPDmin",]:
        clevs = [-0.1,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        cmap  = plt.cm.BrBG
    elif var_name in ["Albedo_inst"]:
        clevs = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
        clevs_percentage =   [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = plt.cm.BrBG_r
    else:
        clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

    return var_diff, clevs, cmap

def plot_spatial_map_Tmax(fire_path, file_name, land_ctl_path, land_sen_path, time_ss=None, time_es=None,
                        lat_names="lat", lon_names="lon", loc_lat=None, loc_lon=None, reg_lats=None, reg_lons=None,
                        wrf_path=None, message=None, burn=0):

    '''
    plot LIS variables in burnt / unburnt / all regions
    '''

    # Read in WRF lat and lon
    wrf            = Dataset(wrf_path,  mode='r')
    lon_in         = wrf.variables['XLONG'][0,:,:]
    lat_in         = wrf.variables['XLAT'][0,:,:]

    # # Read Burned Date
    # #               1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 1
    # day_in_month   = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    # # =========== Read in data ===========
    # fire_file      = Dataset(fire_path, mode='r')
    # Burn_Date_tmp  = fire_file.variables['Burn_Date'][2:8,::-1,:]  # 2019-09 - 2020-02
    # lat_fire       = fire_file.variables['lat'][::-1]
    # lon_fire       = fire_file.variables['lon'][:]

    # Burn_Date      = Burn_Date_tmp.astype(float)
    # Burn_Date      = np.where(Burn_Date<=0, 99999, Burn_Date)

    # Burn_Date[4:,:,:] = Burn_Date[4:,:,:]+365 # Add 365 to Jan-Feb 2020

    # Burn_Date_min  = np.nanmin(Burn_Date, axis=0)
    # unique_values  = np.unique(Burn_Date)
    # print("Unique values:", unique_values)

    # Burn_Date_index = np.where(Burn_Date_min<=273,   1009, Burn_Date_min)
    # print('Burn_Date_index.count(1009)',np.count_nonzero(Burn_Date_index == 1009))
    # Burn_Date_index = np.where(Burn_Date_index<=304, 1010, Burn_Date_index)
    # print('Burn_Date_index.count(1010)',np.count_nonzero(Burn_Date_index == 1010))
    # Burn_Date_index = np.where(Burn_Date_index<=334, 1011, Burn_Date_index)
    # print('Burn_Date_index.count(1011)',np.count_nonzero(Burn_Date_index == 1011))
    # Burn_Date_index = np.where(Burn_Date_index<=365, 1012, Burn_Date_index)
    # print('Burn_Date_index.count(1012)',np.count_nonzero(Burn_Date_index == 1012))
    # Burn_Date_index = np.where(Burn_Date_index<=396, 1013, Burn_Date_index)
    # print('Burn_Date_index.count(1013)',np.count_nonzero(Burn_Date_index == 1013))
    # Burn_Date_index = np.where(Burn_Date_index<=425, 1014, Burn_Date_index)
    # print('Burn_Date_index.count(1014)',np.count_nonzero(Burn_Date_index == 1014))
    # Burn_Date_index = np.where(Burn_Date_index>=99999, np.nan, Burn_Date_index)
    # print('Burn_Date_index.count(np.nan)',np.count_nonzero( np.isnan(Burn_Date_index)))

    # print('Burn_Date_index.count(99999)',np.count_nonzero(Burn_Date_index == 99999))
    # # np.savetxt("Burn_Date_index.txt",Burn_Date_index,delimiter=",")

    # read in var
    Tmax_diff_Dec, clevs, cmap = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                lat_names, lon_names, loc_lat, loc_lon, time_s=time_ss[0], time_e=time_es[0])
    Tmax_diff_Jan, clevs, cmap = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                lat_names, lon_names, loc_lat, loc_lon, time_s=time_ss[1], time_e=time_es[1])
    Tmax_diff_Feb, clevs, cmap = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                lat_names, lon_names, loc_lat, loc_lon, time_s=time_ss[2], time_e=time_es[2])

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=[4,10],sharex=False,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

    plt.subplots_adjust(wspace=0., hspace=0.1)

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

    states= NaturalEarthFeature(category="cultural", scale="50m",
                                        facecolor="none",
                                        name="admin_1_states_provinces_shp")

    # =============== CHANGE HERE ===============

    for i in np.arange(3):

        axs[i].set_extent([135,155,-39,-24])
        axs[i].add_feature(states, linewidth=.5, edgecolor="black")
        axs[i].set_facecolor('lightgray')
        axs[i].coastlines(resolution="50m",linewidth=1)

        axs[i].tick_params(axis='x', direction='out')
        axs[i].tick_params(axis='y', direction='out')
        x_ticks = np.arange(135, 156, 5)
        y_ticks = np.arange(-40, -20, 5)
        axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)

    axs[0].set_xticklabels([])
    axs[0].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                    '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                    '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])
    axs[2].set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                        '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)
    axs[2].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                    '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])

    # cmap1 = plt.cm.Pastel2
    plot1 = axs[0].contourf(lon_in, lat_in, Tmax_diff_Dec, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
    plot2 = axs[1].contourf(lon_in, lat_in, Tmax_diff_Jan, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
    plot3 = axs[2].contourf(lon_in, lat_in, Tmax_diff_Feb, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
    cbar = plt.colorbar(plot3, ax=axs[0:3], ticklocation="right", pad=0.06, orientation="horizontal",
            aspect=30, shrink=0.9) # cax=cax,
    cbar.set_ticks([-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    cbar.set_ticklabels(['','-0.9','','-0.7','','-0.5','','-0.3','','-0.1', '0.1','','0.3','','0.5','','0.7','','0.9',''])
    cbar.ax.tick_params(labelsize=10,labelrotation=90)
    cbar.set_label('ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=14)# rotation=270,
    axs[0].text(0.02, 0.15, "(a)", transform=axs[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1].text(0.02, 0.15, "(b)", transform=axs[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2].text(0.02, 0.15, "(c)", transform=axs[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # draw the burn box
    if 0:
        reg_lats = [-39,-27]
        reg_lons = [146,154]
        for i in np.arange(3):
            axs[i].add_patch(Polygon([[reg_lons[0], reg_lats[0]], [reg_lons[1], reg_lats[0]],
                                        [reg_lons[1], reg_lats[1]], [reg_lons[0], reg_lats[1]]],
                                        closed=True,color=almost_black, fill=False,linewidth=0.8))


    axs[0].set_ylabel("Dec 2019", fontsize=12)
    axs[1].set_ylabel("Jan 2020", fontsize=12)
    axs[2].set_ylabel("Feb 2020", fontsize=12)

    for i in np.arange(3):
        axs[i].plot([ 149, 154], [-30, -30],     c=almost_black, lw=1.5, alpha = 1, linestyle="--", transform=ccrs.PlateCarree())
        axs[i].plot([ 148, 153], [-33, -33],     c=almost_black, lw=1.5, alpha = 1, linestyle="--", transform=ccrs.PlateCarree())
        axs[i].plot([ 146, 151], [-37.5, -37.5], c=almost_black, lw=1.5, alpha = 1, linestyle="--", transform=ccrs.PlateCarree())


    # fig.tight_layout()
    plt.savefig('./plots/Fig2_spatial_map_' +message + '.png',dpi=300)
    return

def plot_profile_wrf_wind(file_outs,message=None):

    # ****************** plotting ******************

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=[7,10],sharex=False, sharey=True, squeeze=True)
    plt.subplots_adjust(wspace=0.14, hspace=0.07)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 12
    plt.rcParams['font.size']       = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

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

    # ===== set colormap =====
    # color map
    color_map_1   = get_cmap("seismic")

    # Define the RGB values as a 2D array
    rgb_17colors= np.array([
                        [0.338024, 0.193310, 0.020377],
                        [0.458593, 0.264360, 0.031142],
                        [0.576471, 0.343483, 0.058055],
                        [0.686275, 0.446828, 0.133410],
                        [0.778547, 0.565859, 0.250288],
                        [0.847443, 0.705805, 0.422530],
                        [0.932872, 0.857209, 0.667820],
                        [0.964091, 0.917801, 0.795463],
                        [0.955517, 0.959016, 0.9570165],
                        [0.808689, 0.924414, 0.907882],
                        [0.627528, 0.855210, 0.820531],
                        [0.426990, 0.749942, 0.706882],
                        [0.265513, 0.633679, 0.599231],
                        [0.135871, 0.524337, 0.492964],
                        [0.023914, 0.418839, 0.387466],
                        [0.002153, 0.325721, 0.287274],
                        [0.000000, 0.235294, 0.188235]
                    ])

    rgb_21colors = np.array([
                [0.338024, 0.193310, 0.020377],
                [0.441369, 0.254210, 0.029604],
                [0.544714, 0.315110, 0.038831],
                [0.631373, 0.395156, 0.095732],
                [0.733333, 0.491119, 0.165706],
                [0.793310, 0.595848, 0.287197],
                [0.857286, 0.725798, 0.447136],
                [0.904575, 0.810458, 0.581699],
                [0.947020, 0.880584, 0.710880],
                [0.963629, 0.923799, 0.818531],
                [0.955517, 0.959016, 0.9570165],
                [0.822837, 0.927797, 0.912803],
                [0.714879, 0.890888, 0.864821],
                [0.583852, 0.837370, 0.798385],
                [0.461592, 0.774856, 0.729950],
                [0.311649, 0.666897, 0.629988],
                [0.183852, 0.569550, 0.538178],
                [0.087889, 0.479123, 0.447751],
                [0.003691, 0.390311, 0.358016],
                [0.001845, 0.312803, 0.273126],
                [0.000000, 0.235294, 0.188235]
            ])


    # Create a colormap from the RGB values
    cmap17 = plt.cm.colors.ListedColormap(rgb_17colors)
    cmap21 = plt.cm.colors.ListedColormap(rgb_21colors)

    # quiver scale
    scale = 1.

    # contour levels
    # levels1    = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
    levels1    = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
    levels_lai = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
    levels_alb = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
    texts = ["(d)","(g)","(j)",
             "(e)","(h)","(k)",
             "(f)","(i)","(l)"]
    # ****************** Read data *****************
    for i, file_out in enumerate(file_outs):

        row = int(i / 3)
        col = i % 3

        ncfile        = Dataset(file_out, mode='r')
        vertical      = ncfile.variables['level'][:]
        xy_loc        = ncfile.variables['lon'][:]

        th_day_diff   = ncfile.variables['th_day_diff'][:]
        ua_day_diff   = ncfile.variables['ua_day_diff'][:]
        wa_day_diff   = ncfile.variables['wa_day_diff'][:]

        lai_diff      = ncfile.variables['lai_diff'][:]
        alb_diff      = ncfile.variables['alb_diff'][:]
        pbl_day_ctl   = ncfile.variables['pbl_day_ctl'][:]
        pbl_day_sen   = ncfile.variables['pbl_day_sen'][:]

        # Water table depth height
        lai_hgt          = [0,100]
        lai_diff_2D      = np.zeros((2,len(lai_diff)))
        lai_diff_2D[0,:] = lai_diff
        lai_diff_2D[1,:] = lai_diff

        alb_hgt          = [-100,0]
        alb_diff_2D      = np.zeros((2,len(alb_diff)))
        alb_diff_2D[0,:] = alb_diff
        alb_diff_2D[1,:] = alb_diff

        # Set color for topography
        ax[row, col].set_facecolor('lightgray')

        # Set the lons boundaries of the transect
        if row == 2:
            ax[row, 0].set_xlim(149, 154)
            ax[row, 0].set_xticks([150, 152, 154])
            ax[row, 0].set_xticklabels(['150','152','154'],rotation=45)
            ax[row, 1].set_xlim(148, 153)
            ax[row, 1].set_xticks([148, 150, 152])
            ax[row, 1].set_xticklabels(['148','150','152'],rotation=45)
            ax[row, 2].set_xlim(146, 151)
            ax[row, 2].set_xticks([146, 148, 150])
            ax[row, 2].set_xticklabels(['146','148','150'],rotation=45)
        else:
            ax[row, 0].set_xlim(149, 154)
            ax[row, 0].set_xticks([150, 152, 154])
            ax[row, 0].set_xticklabels([])
            ax[row, 1].set_xlim(148, 153)
            ax[row, 1].set_xticks([148, 150, 152])
            ax[row, 1].set_xticklabels([])
            ax[row, 2].set_xlim(146, 151)
            ax[row, 2].set_xticks([146, 148, 150])
            ax[row, 2].set_xticklabels([])


        ax[row,col].text(0.02, 0.135, texts[i], transform=ax[row,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        # Plot variables
        contour   = ax[row, col].contourf(xy_loc, vertical, th_day_diff, levels=levels1, cmap=color_map_1, extend='both')
        cntr_lai  = ax[row, col].contourf(xy_loc, lai_hgt,  lai_diff_2D, levels=levels_lai, cmap=cmap21, extend='both')
        cntr_alb  = ax[row, col].contourf(xy_loc, alb_hgt,  alb_diff_2D, levels=levels_alb, cmap=cmap17, extend='both')
        line1     = ax[row, col].plot(xy_loc,pbl_day_ctl,ls="-", lw=0.5, color=almost_black)
        line2     = ax[row, col].plot(xy_loc,pbl_day_sen,ls="--", lw=0.5, color=almost_black)
        line3     = ax[row, col].axhline(y=0,   color=almost_black, lw=0.5, linestyle='-')
        line4     = ax[row, col].axhline(y=100, color=almost_black, lw=0.5, linestyle='-')

        # q         = ax[row, col].quiver(xy_loc[::30], vertical[::3], ua_day_diff[::3,::30],
        #                         wa_day_diff[::3,::30], angles='xy', scale_units='xy',
        #                         scale=scale, pivot='middle', color="lightgrey")
        ax[row, col].set_ylim(-100,3000)


    # # Set titles
    # ax[0,0].set_ylabel("Dec 2019")
    # ax[0,1].set_ylabel("Jan 2020")
    # ax[0,2].set_ylabel("Feb 2020")

    ax[0,0].set_title("North", fontsize=12)
    ax[0,1].set_title("Central", fontsize=12)
    ax[0,2].set_title("South", fontsize=12)

    ax[0,0].set_ylabel("Geopotential Height (m)", fontsize=12)
    ax[1,0].set_ylabel("Geopotential Height (m)", fontsize=12)
    ax[2,0].set_ylabel("Geopotential Height (m)", fontsize=12)

    # fig.text(0.01, 0.5, 'Geopotential Height (m)', va='center', rotation='vertical')

    # Add Tmax colorbar
    cbar  = plt.colorbar(contour, ax=ax, ticklocation="right", pad=0.02, orientation="vertical",
            aspect=70, shrink=1.0) # cax=cax,
    cbar.set_label('Δθ$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
    cbar.ax.tick_params(labelsize=12) # ,labelrotation=45
    cbar.set_ticks([-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
    # cbar.set_ticklabels(['-1.0','-0.9','-0.8','-0.7','-0.6','-0.5','-0.4','-0.3','-0.2','-0.1', '0.1',
    #                     '0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])
    cbar.set_ticklabels(['','-0.9','','-0.7','','-0.5','','-0.3','','-0.1', '0.1',
                        '','0.3','','0.5','','0.7','','0.9',''])

    # Add LAI and albedo colorbar
    position_lai  = fig.add_axes([0.08, 0.055, 0.362, 0.009]) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(cntr_lai, ax=ax, pad=0.125, cax=position_lai, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('ΔLAI (m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$)', loc='center',size=12)# rotation=270,
    cb_lai.ax.tick_params(labelsize=10,rotation=90)
    cb_lai.set_ticks([-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2])
    # cb_lai.set_ticklabels(['-2.0','-1.8','-1.6','-1.4','-1.2','-1.0','-0.8','-0.6','-0.4','-0.2','-0.05',
    #                         '0.05','0.2','0.4','0.6','0.8','1.0','1.2','1.4','1.6','1.8','2.0']) # cax=cax,
    cb_lai.set_ticklabels(['','-1.8','','-1.4','','-1.0','','-0.6','','-0.2','-0.05',
                            '0.05','0.2','','0.6','','1.0','','1.4','','1.8','']) # cax=cax,

    position_alb  = fig.add_axes([0.47, 0.055, 0.362, 0.009]) # [left, bottom, width, height]
    cb_alb        = fig.colorbar(cntr_alb, ax=ax, pad=0.125, cax=position_alb, orientation="horizontal", aspect=60, shrink=0.8)
    cb_alb.set_label('Δ$α$ (-)', loc='center',size=12)# rotation=270,
    cb_alb.ax.tick_params(labelsize=10,rotation=90)
    cb_alb.set_ticks([-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])
    # cb_alb.set_ticklabels(['-0.08','-0.07','-0.06','-0.05','-0.04','-0.03','-0.02','-0.01',
    #                         '0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08']) # cax=cax,

    cb_alb.set_ticklabels(['','-0.07','','-0.05','','-0.03','','-0.01',
                            '0.01','','0.03','','0.05','','0.07','']) # cax=cax,

    fig.savefig("./plots/Fig2_profile_wrf_new_"+message, bbox_inches='tight', pad_inches=0.3,dpi=300)

if __name__ == "__main__":

    loc_lat    = [-40,-23]
    loc_lon    = [134,155]

    '''
    Difference plot yearly

    '''
    case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
    case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

    fire_path      = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
    wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"
    land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
    land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

    burn           = 1
    file_name      = 'LIS.CABLE.201912-202002.nc'


    reg_lats       = [  [-32,-28.5],
                       [-34.5,-32.5],
                       [-38,-34.5]    ]

    reg_lons       = [  [151.5,153.5],
                       [149.5,151.5],
                       [146.5,151]    ]

    message      = "HW_Tmax"
    time_ss      = [datetime(2019,12,1,0,0,0,0),
                    datetime(2020,1,1,0,0,0,0),
                    datetime(2020,2,1,0,0,0,0)]

    time_es      = [datetime(2020,1,1,0,0,0,0),
                    datetime(2020,2,1,0,0,0,0),
                    datetime(2020,3,1,0,0,0,0)]

    plot_spatial_map_Tmax(fire_path, file_name, land_ctl_path, land_sen_path, time_ss=time_ss, time_es=time_es, lat_names="lat",
                        lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon, reg_lats=reg_lats, reg_lons=reg_lons,
                        wrf_path=wrf_path, message=message, burn=burn)


    message        = "profile_transect"

    file_outs = [ "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Dec_lat-30_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Dec_lat-33_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Dec_lat-375_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Jan_lat-30_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Jan_lat-33_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Jan_lat-375_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Feb_lat-30_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Feb_lat-33_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Feb_lat-375_houly.nc"]

    plot_profile_wrf_wind(file_outs,message=message)
